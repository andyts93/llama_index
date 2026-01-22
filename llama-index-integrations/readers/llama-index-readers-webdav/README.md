# LlamaIndex Readers Integration: WebDav

## Overview

Simple WebDav reader allows to read files from a WebDav server both fully or incrementally.

## Installation

```
pip install llama-index-readers-webdav
```

## Usage

```python
from llama_index.readers.webdav import WebDAVReader
from llama_index.readers.file import (
    PDFReader,
)

# Initialize the server
parser = PDFReader()
file_extractor = {".pdf": parser}
reader = WebDAVReader(
    webdav_options={
        "webdav_hostname": "<URL>",
        "webdav_login": "<USERNAME>",
        "webdav_password": "<PASSWORD>",
    },  # pass any other webdav3 options https://pypi.org/project/webdavclient3/
    file_extractor=file_extractor,
    remote_path="/",
    folder_etag_propagated_to_root=False,  # If your webdav server propagates etag updates up to the `remote_path` folder enable it to skip the scan of folders with no updates
)

# Lazy load emails from the given mailbox
documents = reader.load_data(incremental=True)
```

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
