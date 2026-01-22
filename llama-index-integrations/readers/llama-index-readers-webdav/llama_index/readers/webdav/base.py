import concurrent.futures
import json
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Union, List

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
            print(url)
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

    def _scan_folder_if_changed(
        self, folder_path: str, depth: int = 0
    ) -> tuple[List[str], List[str]]:
        """
        Scans folder only if its Etag is changed

        :param folder_path:
        :param depth:
        :return: (files_to_check, subfolders_to_check)
        """
        current_etag = self._get_folder_etag(folder_path)
        cached_etag = self.cache["folders"].get(folder_path, {}).get("etag")
        # folder_path = folder_path if folder_path.endswith('/') else folder_path + '/'

        if (
            self.folder_etag_propagated_to_root
            and current_etag
            and current_etag == cached_etag
        ):
            self.logger.debug(f"  {'  ' * depth}⏭️  {folder_path} (not modified)")
            return [], []

        self.logger.debug(f"  {'  ' * depth}⏭️  {folder_path} (modified or new)")

        files_to_check = []
        subfolders = []

        try:
            items = self.client.list(folder_path, get_info=True)
        except Exception as e:
            self.logger.error(f"Listing  {'  ' * depth}❌  {folder_path}: {e}")
            return [], []

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

                files_to_check.append(remote_item)

        # Update folder Etag cached
        self.cache["folders"][folder_path] = {
            "etag": current_etag,
            "last_check": datetime.now().isoformat(),
        }

        return files_to_check, subfolders

    def _check_files_etags_parallel(self, file_paths: List[str]) -> List[str]:
        """
        Checks Etag of multiple files in parallel
        :param file_paths:
        :return: modified_files
        """

        def check_single_file(path):
            current_etag = self._get_etag(path)
            cached_etag = self.cache["files"].get(path, {}).get("etag")

            is_changed = (
                current_etag is None
                or cached_etag is None
                or current_etag != cached_etag
            )

            if is_changed:
                self.cache["files"][path] = {
                    "etag": current_etag,
                    "last_check": datetime.now().isoformat(),
                }

            return path, is_changed

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            results = executor.map(check_single_file, file_paths)

        return [path for path, changed in results if changed]

    def _smart_scan(self) -> List[str]:
        """
        Smart scan:
        1. Checks folder's Etag
        2. If changed, scan content
        3. Recursively on subfolders
        4. For each file, Etag check
        :return: changed files
        """
        all_changed_files = []
        folders_to_scan = [self.remote_path]

        while folders_to_scan:
            folder = folders_to_scan.pop(0)
            files, subfolders = self._scan_folder_if_changed(folder)

            if files:
                self.logger.debug(f" -> Checking {len(files)} files in parallel")
                changed = self._check_files_etags_parallel(files)
                all_changed_files.extend(changed)
                self.logger.debug(f"   ✅ {len(changed)} modified files")

            folders_to_scan.extend(subfolders)

        return all_changed_files

    def _get_all_files_simple(self) -> List[str]:
        """Simple scan for first sync"""

        def scan_recursive(path):
            files = []
            items = self.client.list(path)

            for item in items:
                if item in [".", "..", ""]:
                    continue

                remote_item = f"{path.rstrip('/')}/{item}"

                if self.client.is_dir(remote_item):
                    if self.recursive:
                        files.extend(scan_recursive(remote_item))
                else:
                    if self.required_exts:
                        ext = os.path.splitext(remote_item)[1].lower()
                        if ext not in self.required_exts:
                            continue
                    files.append(remote_item)

            return files

        return scan_recursive(self.remote_path)

    def load_data(self, incremental: bool = True) -> List[Document]:
        if incremental and (self.cache["folders"] or self.cache["files"]):
            files_to_download = self._smart_scan()
        else:
            self.logger.debug("Complete scan")
            all_files = self._get_all_files_simple()

            self._check_files_etags_parallel(all_files)

            files_to_download = all_files

        self.logger.debug(f"🚀 Files to download: {len(files_to_download)}")

        if not files_to_download:
            self.logger.debug("✅ No changes detected")
            return []

        self.logger.debug("Downloading files")
        with tempfile.TemporaryDirectory() as tmp_dir:
            for file_path in files_to_download:
                rel_path = file_path.replace(self.remote_path.rstrip("/"), "").lstrip(
                    "/"
                )
                tmp_file = os.path.join(tmp_dir, rel_path)

                os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

                try:
                    self.client.download_sync(
                        remote_path=file_path,
                        local_path=tmp_file,
                    )
                    self.logger.debug(f"{rel_path} downloaded")
                except Exception as e:
                    self.logger.error(f"{rel_path}: {e}")

            simple_loader = SimpleDirectoryReader(
                tmp_dir,
                file_extractor=self.file_extractor,
                required_exts=self.required_exts,
                recursive=True,
            )

            documents = simple_loader.load_data()

        self._save_cache()
        self.logger.debug(f"✅ Loaded {len(documents)} documents")

        return documents

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
