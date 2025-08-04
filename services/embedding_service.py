import numpy as np
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info("Embedding service initialized")

    def extract_embedding(self, face_data) -> np.ndarray:
        embedding = face_data.normed_embedding
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        if embedding.shape[0] != 512:
            raise ValueError("Invalid embedding dimension")
        return normalize(embedding.reshape(1, -1), axis=1).flatten()
