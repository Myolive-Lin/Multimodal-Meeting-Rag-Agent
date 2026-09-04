import os
import re
import cv2
import pysubs2
import logging
import tempfile
import subprocess
from langchain_core.documents import Document
from faster_whisper import WhisperModel
from app.config import settings
from functools import lru_cache
from typing import List

# Todo: encapsulate one more, let agent have the ability to extract_subtitle, image, and understand video



logger = logging.getLogger(__name__)

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def extract_subtitles(sub_path: str, merge_window = 5.0) -> List[str]:
    """Parse subtitle files and merge adjacent subtitles
    :param sub_path: Path to the subtitle file
    :param merge_window: Window size in seconds to merge adjacent subtitles
    :return: List of Document objects
    """
    try:
        subs = pysubs2.load(sub_path)
    except Exception:
        return []

    contents = []
    start_time = None
    merged_text = []
    file_name = os.path.basename(sub_path)

    #subs:[start_time, end_time, text]
    for sub in subs:
        text = sub.text.strip()
        text = re.sub(r'\{.*?\}', '', text)

        if not text:
            continue

        sub_start = sub.start / 1000

        if start_time is None:
            start_time = sub_start
            merged_text = [text]
        elif sub_start - start_time <= merge_window:
            merged_text.append(text)
        else:
            content = f"Source:{file_name} {format_timestamp(start_time)}\nContent:{''.join(merged_text)}"
            contents.append(content)
            start_time = sub_start
            merged_text = [text]

    if merged_text:
        content = f"Source:{file_name} {format_timestamp(start_time)}\nContent:{''.join(merged_text)}"
        contents.append(content)

    return contents

@lru_cache(maxsize=1)
def get_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls= True, show_log = False)

@lru_cache(maxsize=1)
def get_whisper():
    device=  settings.device
    return WhisperModel(settings.whisper_model_size, device=device, compute_type='int8')


def extract_image(img_path:str) -> List[str]:
    try:
        ocr = get_ocr()
        result = ocr.ocr(img_path, cls = True) #cls angle classification
    except Exception:
        return []

    if not result or not result[0]:
        return []

    #line = [bbox,(text, confidence)]

    text = ''.join([line[1][0] for line in result[0]])
    file_name = os.path.basename(img_path)
    content = f"Source:{file_name}\n Content:{text}"
    return [content]

def extract_audio(video_path:str) -> List[str]:
    try:

        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
            temp_audio = f.name

        cmd = [
            "ffmpeg", "-y", '-i', video_path,
            '-acodec', 'pcm_s16le', "-ar", "16000", "-ac", "1",
            temp_audio
        ]
        subprocess.run(cmd, capture_output=True, check = True)

        model = get_whisper()
        segments, _ = model.transcribe(temp_audio, language = settings.language)

        os.remove(temp_audio)

    except Exception:
        return []

    contents = []
    file_name = os.path.basename(video_path)

    #[ segment.start, segment.end, segment.text]
    for segment in segments:
        content = f"Source:{file_name} {format_timestamp(segment.start)}\n Content:{segment.text.strip()}"
        contents.append(content)

    return contents

def extract_frames(video_path: str, frame_interval: float = 5.0) -> List[str]:
    """Extract textual content from video by sampling frames for OCR and transcribing audio using ASR."""
    contents = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(int(fps * frame_interval), 1)
        frame_count = 0

        ocr = get_ocr()
        file_name = os.path.basename(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                try:
                    result = ocr.ocr(frame, cls=True)
                    if result and result[0]:
                        text = "\n".join(line[1][0] for line in result[0])
                        timestamp = format_timestamp(frame_count / fps)

                        content = f"Source:{file_name} {timestamp}\nContent:{text}"
                        contents.append(content)

                except Exception as e:
                    logger.warning(f"OCR failed at frame {frame_count}: {e}")

            frame_count += 1

        cap.release()

    except Exception as e:
        logger.warning(f"Video processing failed: {e}")

    logger.info("Starting audio transcription (Whisper)...")
    audio_contents = extract_audio(video_path)
    contents.extend(audio_contents)

    logger.info(f"Finished processing video: {video_path}, total docs={len(contents)}")
    return contents
