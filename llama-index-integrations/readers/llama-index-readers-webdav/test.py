from llama_index.readers.webdav import WebDAVReader
from llama_index.readers.file import (
    PDFReader,
)

if __name__ == "__main__":
    parser = PDFReader()
    file_extractor = {".pdf": parser}

    reader = WebDAVReader(
        url="http://localhost:8070/",
        username="test",
        password="test",
        file_extractor=file_extractor,
    )
    print(reader.load_data())
