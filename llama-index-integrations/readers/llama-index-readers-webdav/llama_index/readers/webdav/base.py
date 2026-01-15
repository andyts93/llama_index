import os
import tempfile
from typing import List, Optional, Dict, Union
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from webdav3.client import Client
from llama_index.core import SimpleDirectoryReader
from llama_index.core.bridge.pydantic import Field


class WebDAVReader(BaseReader):
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        remote_path: str = "/",
        recursive: bool = True,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = Field(
            default=None, exclude=True
        ),
        required_exts: Optional[List[str]] = None,
    ):
        self.client = Client(
            {
                "webdav_hostname": url,
                "webdav_login": username,
                "webdav_password": password,
            }
        )
        self.remote_path = remote_path
        self.recursive = recursive
        self.file_extractor = file_extractor
        self.required_exts = required_exts

    def load_data(self) -> List[Document]:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._download_recursive(self.remote_path, tmpdir)

            simple_loader = SimpleDirectoryReader(
                tmpdir,
                file_extractor=self.file_extractor,
                required_exts=self.required_exts,
                # file_metadata=get_metadata,
                # recursive=recursive,
            )
            return simple_loader.load_data()

    def _download_recursive(self, remote_path: str, local_path: str):
        items = self.client.list(remote_path)

        for item in items:
            if item in [".", ".."]:
                continue

            remote_item = f"{remote_path.rstrip('/')}/{item}"
            local_item = os.path.join(local_path, item)

            if self.client.is_dir(remote_item):
                os.makedirs(local_item, exist_ok=True)
                if self.recursive:
                    self._download_recursive(remote_item, local_item)
            else:
                self.client.download_sync(
                    remote_path=remote_item,
                    local_path=local_item,
                )
