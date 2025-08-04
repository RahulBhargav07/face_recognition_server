import logging
from fastapi import UploadFile
import os

SUPPORTED_FORMATS = {
    'image/jpeg', 'image/jpg', 'image/png', 
    'image/bmp', 'image/webp', 'image/gif'
}
SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler("face_recognition_api.log")]
    )

def validate_image(file: UploadFile) -> bool:
    if file.content_type not in SUPPORTED_FORMATS:
        return False
    ext = os.path.splitext(file.filename.lower())[1]
    return ext in SUPPORTED_EXTENSIONS
