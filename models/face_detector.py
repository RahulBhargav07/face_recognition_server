import insightface
import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image_data: bytes):
        pil_image = Image.open(io.BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        faces = self.app.get(cv_image)

        results = []
        for face in faces:
            bbox = face.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            results.append({
                "face_data": face,
                "confidence": float(face.det_score),
                "area": float(area),
                "bbox": bbox.tolist(),
                "landmarks": face.kps.tolist() if hasattr(face, "kps") else None
            })
        return results
