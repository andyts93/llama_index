import concurrent.futures
import json
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Union, List, Generator
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.bridge.pydantic import Field
from webdav3.client import Client
import requests
import logging


class WebDAVReader(BaseReader):
    """
    WebDAV reader with Etag check
    - First checks folders Etag
    - Only scans modified folders
    - Sends multiple parallel HEAD requests for files
    """

    def __init__(
        self,
        webdav_options: dict,
        remote_path: str = "/",
        recursive: bool = True,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
            default=None, exclude=True
        ),
        required_exts: Optional[List[str]] = None,
        cache_file: Optional[str] = None,
        max_workers: int = 10,
        logger: Optional[logging.Logger] = None,
        folder_etag_propagated_to_root: bool = True,
        batch_etag_check: bool = True,
    ):
        self.client = Client(webdav_options)
        self.remote_path = remote_path
        self.recursive = recursive
        self.file_extractor = file_extractor
        self.required_exts = required_exts
        self.cache_file = cache_file or ".webdav_smart_cache.json"
        self.max_workers = max_workers
        self.cache = self._load_cache()
        self.folder_etag_propagated_to_root = folder_etag_propagated_to_root
        self.batch_etag_check = batch_etag_check

        # Required for direct HTTP requests
        self.base_url = webdav_options.get("webdav_hostname", "")
        self.auth = (
            webdav_options.get("webdav_login"),
            webdav_options.get("webdav_password"),
        )

        self.logger = logger or logging.getLogger(__name__)

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {"folders": {}, "files": {}}
        return {"folders": {}, "files": {}}

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_etag(self, remote_path: str) -> Optional[str]:
        """Get etag with HEAD request"""
        try:
            url = f"{self.base_url.rstrip('/')}/{remote_path}"
            response = requests.get(url, auth=self.auth, timeout=10)
            etag = response.headers.get("ETag", "").strip('"')
            return etag if etag else None
        except Exception:
            return None

    def _get_folder_etag(self, folder_path: str) -> Optional[str]:
        """Get folder etag, it should change on most of the servers"""
        try:
            info = self.client.info(folder_path)
            etag = info.get("etag", "").strip('"')
            return etag if etag else None
        except Exception:
            return None

    def _check_file_etag(self, file: dict) -> bool:
        """
        Check if a single file has changed
        :return: True if file has changed or is new
        """
        current_etag = file["etag"].strip('"') or self._get_etag(file["remote_path"])
        cached_etag = self.cache["files"].get(file["remote_path"], {}).get("etag")

        is_changed = (
            current_etag is None or cached_etag is None or current_etag != cached_etag
        )

        if is_changed:
            self.cache["files"][file["remote_path"]] = {
                "etag": current_etag,
                "last_check": datetime.now().isoformat(),
            }

        return is_changed

    def _check_files_etags_parallel(self, files: List[dict]) -> List[dict]:
        """
        Checks Etag of multiple files in parallel
        :param file_paths:
        :return: modified_files
        """

        def check_single_file(file: dict):
            current_etag = file["etag"].strip('"') or self._get_etag(
                file["remote_path"]
            )
            cached_etag = self.cache["files"].get(file["remote_path"], {}).get("etag")

            is_changed = (
                current_etag is None
                or cached_etag is None
                or current_etag != cached_etag
            )

            if is_changed:
                self.cache["files"][file["remote_path"]] = {
                    "etag": current_etag,
                    "last_check": datetime.now().isoformat(),
                }

            return file, is_changed

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            results = executor.map(check_single_file, files)

        return [file for file, changed in results if changed]

    def _download_and_parse_file(self, file: dict, tmp_dir: str) -> List[Document]:
        """
        Downloads and parses a single file

        :param file_path: Remote file path
        :param tmp_dir: Temporary directory for download
        :return: List of documents from the file
        """
        rel_path = (
            file["remote_path"].replace(self.remote_path.rstrip("/"), "").lstrip("/")
        )
        tmp_file = os.path.join(tmp_dir, rel_path)

        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

        try:
            self.client.download_sync(
                remote_path=file["remote_path"],
                local_path=tmp_file,
            )
            self.logger.debug(f"📥 {rel_path} downloaded")

            # Parse the single file
            simple_loader = SimpleDirectoryReader(
                input_files=[tmp_file],
                file_extractor=self.file_extractor,
                required_exts=self.required_exts,
            )

            documents = simple_loader.load_data()

            # Add metadata about the source
            for doc in documents:
                doc.metadata["webdav_path"] = file["remote_path"]
                doc.metadata["webdav_filename"] = os.path.basename(file["remote_path"])
                doc.metadata["creation_date"] = file["created"]
                doc.metadata["last_modified_date"] = file["modified"]

            self.logger.debug(f"✅ {rel_path} parsed ({len(documents)} docs)")

            # Clean up immediately
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception as e:
                self.logger.warning(f"Could not remove temp file {tmp_file}: {e}")

            return documents

        except Exception as e:
            self.logger.error(f"❌ {rel_path}: {e}")
            return []

    def _scan_and_yield_folder(
        self, folder_path: str, tmp_dir: str, depth: int = 0
    ) -> Generator[Document, None, None]:
        """
        Scans a folder and yields documents immediately as files are found
        Also yields from subfolders recursively

        :param folder_path: Folder to scan
        :param tmp_dir: Temporary directory for downloads
        :param depth: Recursion depth (for logging)
        :yield: Documents from files
        """
        # Check if folder has changed
        current_etag = self._get_folder_etag(folder_path)
        cached_etag = self.cache["folders"].get(folder_path, {}).get("etag")

        if (
            self.folder_etag_propagated_to_root
            and current_etag
            and current_etag == cached_etag
        ):
            self.logger.debug(f"  {'  ' * depth}⏭️  {folder_path} (not modified)")
            return

        self.logger.debug(f"  {'  ' * depth}📂 {folder_path} (scanning)")

        files_in_folder = []
        subfolders = []

        try:
            items = self.client.list(folder_path, get_info=True)
        except Exception as e:
            self.logger.error(f"Listing  {'  ' * depth}❌  {folder_path}: {e}")
            return

        # Separate files and folders
        for item in items:
            if item in [".", "..", ""]:
                continue
            if isinstance(item, dict):
                item_name = item.get("name")
                if not item_name:
                    item_name = (
                        item.get("path", "").replace(folder_path, "").lstrip("/")
                    )

                if not item_name or item_name in [".", ".."]:
                    continue

                remote_item = f"{folder_path.rstrip('/')}/{item_name}"
                is_dir = item.get("isdir", False)
            else:
                remote_item = f"{folder_path.rstrip('/')}/{item}"
                is_dir = self.client.is_dir(remote_item)

            if is_dir:
                if self.recursive:
                    subfolders.append(remote_item)
            else:
                if self.required_exts:
                    ext = os.path.splitext(remote_item)[1].lower()
                    if ext not in self.required_exts:
                        continue

                files_in_folder.append({**item, "remote_path": remote_item})

        # Update folder cache
        self.cache["folders"][folder_path] = {
            "etag": current_etag,
            "last_check": datetime.now().isoformat(),
        }

        # Process files in this folder
        if files_in_folder:
            if self.batch_etag_check and len(files_in_folder) > 1:
                # Check etags in parallel for efficiency
                self.logger.debug(
                    f"  {'  ' * depth}🔍 Checking {len(files_in_folder)} files"
                )
                changed_files = self._check_files_etags_parallel(files_in_folder)
            else:
                # Check one by one (useful for immediate yielding)
                changed_files = []
                for file in files_in_folder:
                    if self._check_file_etag(file):
                        changed_files.append(file)

            # Download and yield documents immediately
            for file in changed_files:
                documents = self._download_and_parse_file(file, tmp_dir)
                for doc in documents:
                    yield doc

        # Recursively process subfolders
        for subfolder in subfolders:
            yield from self._scan_and_yield_folder(subfolder, tmp_dir, depth + 1)

    def load_data(self, incremental: bool = True) -> List[Document]:
        """
        Legacy method for backward compatibility - loads all documents at once
        """
        return list(self.lazy_load_data(incremental=incremental))

    def lazy_load_data(
        self, incremental: bool = True
    ) -> Generator[Document, None, None]:
        """
        Yields documents one by one as folders are scanned and files are parsed
        No waiting for complete scan - immediate streaming

        :param incremental: If True, uses smart scan with etag checking
        :yield: Individual documents
        """
        self.logger.debug(
            f"🚀 Starting {'incremental' if incremental else 'full'} scan"
        )

        if not incremental:
            # Clear cache for full scan
            self.cache = {"folders": {}, "files": {}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            total_docs = 0

            # Start scanning and yielding immediately
            for doc in self._scan_and_yield_folder(self.remote_path, tmp_dir):
                total_docs += 1
                self._save_cache()
                yield doc

            self.logger.debug(f"✅ Yielded {total_docs} documents total")

    def get_stats(self) -> Dict:
        return {
            "cached_folders": len(self.cache.get("folders", {})),
            "cached_files": len(self.cache.get("files", {})),
            "cache_file": self.cache_file,
            "cache_exists": os.path.exists(self.cache_file),
        }

    def force_full_reindex(self):
        self.cache = {"folders": {}, "files": {}}
        self._save_cache()
        self.logger.debug(f"✅ Cache cleared")

    def clear_folder_cache(self, folder_path: str) -> None:
        if folder_path in self.cache["folders"]:
            del self.cache["folders"][folder_path]
            self._save_cache()
            self.logger.debug(f"✅ Folder {folder_path} cache deleted")
