import os
import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import faiss
import numpy as np


@dataclass
class IdentityMatch:
    id: int
    name: str
    confidence: float


class TwoTierFaceDatabase:
    """
    Two-Tier Decoupled Database Architecture:
    1. Tier 1 (FAISS IndexFlatIP): In-memory C++ SIMD vector matching (< 0.2ms)
    2. Tier 2 (SQLite with WAL mode): Instant zero-copy metadata lookup (< 0.02ms)
    """

    def __init__(
        self,
        index_path: str = "face_rec_module/data/faces.faiss",
        db_path: str = "face_rec_module/data/identities.db",
        dim: int = 512,
        threshold: float = 0.50,
    ):
        self.index_path = Path(index_path)
        self.db_path = Path(db_path)
        self.dim = dim
        self.threshold = threshold

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_sqlite()
        self._init_faiss()

    def _init_sqlite(self):
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA cache_size=-64000;")

        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Map each FAISS vector index (row_idx) to an identity id
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_map (
                    row_idx INTEGER PRIMARY KEY,
                    identity_id INTEGER NOT NULL,
                    FOREIGN KEY(identity_id) REFERENCES identities(id)
                );
                """
            )

    def _init_faiss(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def register_identity(self, name: str, embedding: np.ndarray) -> int:
        """Register a single vector for an identity."""
        return self.register_identity_multi_vector(name, [embedding])

    def register_identity_multi_vector(self, name: str, embeddings: List[np.ndarray]) -> int:
        """
        Multi-Vector Gallery Registration:
        Stores multiple angle vectors (center, left, right, up, down) in FAISS,
        all mapped to the same identity ID in SQLite.
        Enables robust recognition even with partial faces, extreme yaw/pitch profiles!
        """
        if not embeddings:
            raise ValueError("No embeddings provided.")

        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO identities (name) VALUES (?)", (name,))
            identity_id = cursor.lastrowid

            vecs = []
            start_row = self.index.ntotal
            for i, emb in enumerate(embeddings):
                vec = np.asarray(emb, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(vec)
                vecs.append(vec)
                cursor.execute(
                    "INSERT INTO vector_map (row_idx, identity_id) VALUES (?, ?)",
                    (start_row + i, identity_id),
                )

        stacked_vecs = np.vstack(vecs)
        self.index.add(stacked_vecs)
        faiss.write_index(self.index, str(self.index_path))
        return identity_id

    def search(self, query_embedding: np.ndarray) -> Optional[IdentityMatch]:
        if self.index.ntotal == 0:
            return None

        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        similarities, indices = self.index.search(vec, 1)
        best_score = float(similarities[0][0])
        best_rowid = int(indices[0][0])

        if best_rowid < 0 or best_score < self.threshold:
            return None

        cursor = self.conn.cursor()
        # Lookup identity_id from vector_map
        cursor.execute("SELECT identity_id FROM vector_map WHERE row_idx = ? LIMIT 1;", (best_rowid,))
        map_row = cursor.fetchone()
        if map_row:
            target_id = map_row[0]
        else:
            # Fallback for backwards compatibility
            target_id = best_rowid + 1

        cursor.execute("SELECT name FROM identities WHERE id = ? LIMIT 1;", (target_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return IdentityMatch(id=target_id, name=row[0], confidence=best_score)

    def get_total_identities(self) -> int:
        return self.index.ntotal

    def list_identities(self) -> List[Tuple[int, str, str]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, created_at FROM identities ORDER BY id ASC;")
        return cursor.fetchall()
