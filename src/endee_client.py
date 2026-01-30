import os
import json
import math
from typing import List, Dict, Any


class EndeeVectorDB:
    """
    Local Endee-like Vector DB (No Server Needed)
    """

    def __init__(self, base_url=None, index_name="docs_index", api_key=None):
        self.index_name = index_name

        self.db_dir = "local_db"
        os.makedirs(self.db_dir, exist_ok=True)

        self.file_path = os.path.join(self.db_dir, f"{self.index_name}.json")

        self.dimension = None
        self.items = []

        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.dimension = data.get("dimension")
                    self.items = data.get("items", [])
            except:
                self.dimension = None
                self.items = []

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension, "items": self.items}, f, ensure_ascii=False)

    def create_index(self, dim: int):
        if self.dimension is None:
            self.dimension = dim
            self._save()
        return True

    def upsert(self, items: List[Dict[str, Any]]):
        if not items:
            return {"status": "empty"}

        if self.dimension is None:
            self.dimension = len(items[0]["vector"])

        existing_ids = {it["id"] for it in self.items}

        for it in items:
            if it["id"] in existing_ids:
                for idx in range(len(self.items)):
                    if self.items[idx]["id"] == it["id"]:
                        self.items[idx] = it
                        break
            else:
                self.items.append(it)

        self._save()
        return {"status": "success", "count": len(items)}

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_vector: List[float], top_k=5):
        if not self.items:
            return {"matches": []}

        scored = []
        for it in self.items:
            score = self._cosine_similarity(query_vector, it["vector"])
            scored.append({"score": float(score), "metadata": it.get("metadata", {})})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": scored[:top_k]}
