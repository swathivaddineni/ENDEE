import os
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
        os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"

        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        vectors = self.model.encode(texts, show_progress_bar=False)
        return np.array(vectors).tolist()
