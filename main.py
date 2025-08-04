from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
import os

from models.face_detector import FaceDetector
from services.embedding_service import EmbeddingService
from utils.helpers import validate_image, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_detector = None
embedding_service = None

@app.on_event("startup")
async def startup_event():
    global face_detector, embedding_service
    face_detector = FaceDetector()
    embedding_service = EmbeddingService()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down...")

@app.get("/")
async def root():
    return {"message": "Face Recognition API", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": face_detector is not None and embedding_service is not None
    }

@app.post("/generate_embedding")
async def generate_embedding(image: UploadFile = File(...)):
    if not validate_image(image):
        raise HTTPException(400, "Invalid image format")
    
    image_data = await image.read()
    if len(image_data) == 0:
        raise HTTPException(400, "Empty image file")

    faces = face_detector.detect_faces(image_data)
    if not faces:
        raise HTTPException(400, "No face detected")

    best_face = max(faces, key=lambda x: x['confidence'] * x['area'])
    if best_face['confidence'] < 0.6:
        raise HTTPException(400, "Low confidence")

    embedding = embedding_service.extract_embedding(best_face['face_data'])

    return {
        "embedding": embedding.tolist(),
        "confidence": best_face['confidence'],
        "bbox": best_face['bbox']
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Endpoint not found"})

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

