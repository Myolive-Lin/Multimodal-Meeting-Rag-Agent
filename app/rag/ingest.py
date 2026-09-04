import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.tool.parsers import extract_frames, extract_image, extract_subtitles
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
SUBTITLE_EXTENSIONS = {'.srt', '.vtt', '.ass', '.ssa'}
TEXT_EXTENSIONS = {'.txt', '.md', '.log'}
def load_and_split(directory: str):
    documents = []
    raw_text_docs = []
    path = Path(directory)

    for filepath in path.rglob("*"):
        if not filepath.is_file():
            continue

        ext = filepath.suffix.lower()

        # 1. video
        if ext in VIDEO_EXTENSIONS:
            contents = extract_frames(str(filepath))
            for c in contents:
                documents.append(Document(page_content=c, metadata={"source": str(filepath), "type": "video"}))

        # 2. image
        elif ext in IMAGE_EXTENSIONS:
            contents = extract_image(str(filepath))
            for c in contents:
                documents.append(Document(page_content=c, metadata={"source": str(filepath), "type": "image"}))

        # 3. subtitle
        elif ext in SUBTITLE_EXTENSIONS:
            contents = extract_subtitles(str(filepath))
            for c in contents:
                documents.append(Document(page_content=c, metadata={"source": str(filepath), "type": "subtitle"}))

        elif ext in TEXT_EXTENSIONS:
            text = filepath.read_text(errors="ignore")
            raw_text_docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(filepath), "type": "text"}
                )
            )

    # only chunk raw text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    return documents + splitter.split_documents(raw_text_docs)