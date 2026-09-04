import os
from langchain_core.tools import tool
from app.rag.retriever import retrieve
from app.tool.parsers import extract_frames, extract_audio, extract_image
from app.rag.ingest import load_and_split
from app.rag.embed import get_embedder
from app.rag.store import insert_document

@tool
def query_documents(question: str, limit = 20, topk = 5) -> str:
    """Retrieve relevant document context for a user question."""
    return retrieve(question, limit=limit, topk=topk)

@tool
def ingest_media(filepath: str) -> str:
    """
    Ingest video, image, subtitle or text files into the vector database.

    Supported file types:
    - Video: .mp4, .avi, .mov, .mkv, .flv, .wmv
    - Image: .jpg, .jpeg, .png, .bmp, .tiff, .gif
    - Subtitle: .srt, .vtt, .ass, .ssa
    - Text: .txt, .md

    :param filepath: Path to a single file or directory containing media files
    :return: Summary of ingestion results
    """
    if os.path.isfile(filepath):
        directory = os.path.dirname(filepath)
    else:
        directory = filepath

    chunks = load_and_split(directory)
    embedder = get_embedder()

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk.page_content, normalize_embeddings=True).tolist()
        insert_document(
            source=chunk.metadata.get('source', filepath),
            chunk_index=i,
            content=chunk.page_content,
            embedding=embedding
        )


    return f"Successfully ingested {len(chunks)} document chunks from {filepath}"


@tool
def read_image(filepath: str) -> str:
    """
    Extract text content from an image using OCR.

    :param filepath: Path to the image file
    :return: Extracted text content with source information
    """
    return "".join(extract_image(filepath))

@tool
def read_audio(filepath: str) -> str:
    """
    Extract text content from audio using speech recognition (ASR).

    :param filepath: Path to the audio or video file
    :return: Transcribed text content with timestamps
    """
    return "".join(extract_audio(filepath))

@tool
def read_video(filepath: str) -> str:
    """
    Extract text content from video by combining OCR frame analysis and audio transcription.

    :param filepath: Path to the video file
    :return: Combined text content from frames and audio with timestamps
    """
    return "".join(extract_frames(filepath))
