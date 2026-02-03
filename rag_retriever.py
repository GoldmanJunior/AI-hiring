"""
Retriever RAG avec recherche vectorielle FAISS.
Recherche les chunks les plus pertinents pour une requête.
"""

import os
import pickle
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import yaml

from entity_normalizer import EntityNormalizer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Résultat d'une recherche RAG."""
    chunk_id: int
    text: str
    score: float
    table_name: str
    row_id: int
    localities: list[str]
    parties: list[str]
    source_type: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "table_name": self.table_name,
            "row_id": self.row_id,
            "localities": self.localities,
            "parties": self.parties,
            "source_type": self.source_type
        }


class RAGRetriever:
    """
    Retriever RAG utilisant FAISS pour la recherche vectorielle.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le retriever.

        Args:
            config_path: Chemin vers la configuration
        """
        self.config = self._load_config(config_path)

        # Configuration
        embedding_config = self.config.get("embedding", {})
        self.model_name = embedding_config.get(
            "model_name",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        faiss_config = self.config.get("faiss", {})
        self.index_path = faiss_config.get("index_path", "rag_index/vectors.faiss")
        self.metadata_path = faiss_config.get("metadata_path", "rag_index/metadata.pkl")

        retrieval_config = self.config.get("retrieval", {})
        self.top_k = retrieval_config.get("top_k", 5)
        self.min_confidence = retrieval_config.get("min_confidence", 0.5)

        # Charger le modèle d'embedding
        logger.info(f"Chargement du modèle d'embedding: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Charger l'index et les métadonnées
        self.index = self._load_index()
        self.metadata, self.chunks = self._load_metadata()

        # Normaliser d'entités
        self.normalizer = EntityNormalizer(config_path)

        logger.info(
            f"RAGRetriever initialisé: {self.index.ntotal} vecteurs, "
            f"{len(self.chunks)} chunks"
        )

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config non trouvée: {config_path}")
            return {}

    def _load_index(self) -> faiss.Index:
        """Charge l'index FAISS."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Index FAISS non trouvé: {self.index_path}. "
                "Exécutez d'abord l'indexation avec rag_indexer.py"
            )

        index = faiss.read_index(self.index_path)
        logger.info(f"Index FAISS chargé: {index.ntotal} vecteurs")
        return index

    def _load_metadata(self) -> tuple[list[dict], list[str]]:
        """Charge les métadonnées et chunks."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Métadonnées non trouvées: {self.metadata_path}. "
                "Exécutez d'abord l'indexation avec rag_indexer.py"
            )

        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)

        metadata = data.get("metadata", [])
        chunks = data.get("chunks", [])

        logger.info(f"Métadonnées chargées: {len(metadata)} entrées")
        return metadata, chunks

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode une requête en vecteur.

        Args:
            query: Requête texte

        Returns:
            Vecteur normalisé
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        return embedding.astype(np.float32)

    def _filter_by_entities(
        self,
        results: list[RetrievalResult],
        query_entities: dict[str, list[str]]
    ) -> list[RetrievalResult]:
        """
        Booste les résultats qui matchent les entités de la requête.

        Args:
            results: Résultats de la recherche
            query_entities: Entités extraites de la requête

        Returns:
            Résultats re-scorés
        """
        query_localities = set(query_entities.get("localities", []))
        query_parties = set(query_entities.get("parties", []))

        if not query_localities and not query_parties:
            return results

        boosted_results = []
        for result in results:
            result_localities = set(result.localities)
            result_parties = set(result.parties)

            # Calculer le boost basé sur le match d'entités
            locality_match = len(query_localities & result_localities) > 0
            party_match = len(query_parties & result_parties) > 0

            boost = 1.0
            if locality_match:
                boost += 0.2
            if party_match:
                boost += 0.2

            # Créer un nouveau résultat avec score boosté
            boosted_result = RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(min(result.score * boost, 1.0)),  # Plafonner à 1.0
                table_name=result.table_name,
                row_id=result.row_id,
                localities=result.localities,
                parties=result.parties,
                source_type=result.source_type
            )
            boosted_results.append(boosted_result)

        # Re-trier par score
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        normalize_query: bool = True
    ) -> list[RetrievalResult]:
        """
        Recherche les chunks les plus pertinents.

        Args:
            query: Requête en langage naturel
            top_k: Nombre de résultats (défaut: config)
            min_confidence: Score minimum (défaut: config)
            normalize_query: Normaliser les entités dans la requête

        Returns:
            Liste de RetrievalResult triés par pertinence
        """
        top_k = top_k or self.top_k
        min_confidence = min_confidence if min_confidence is not None else self.min_confidence

        # Normaliser la requête si demandé
        processed_query = query
        query_entities = {"localities": [], "parties": []}

        if normalize_query:
            query_entities = self.normalizer.extract_entities(query)
            processed_query = self.normalizer.normalize_query(query)
            if processed_query != query:
                logger.info(f"Requête normalisée: '{query}' -> '{processed_query}'")

        # Encoder la requête
        query_vector = self._encode_query(processed_query)

        # Recherche dans FAISS
        scores, indices = self.index.search(query_vector, top_k * 2)  # Récupérer plus pour filtrage

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            # Convertir le score (inner product après normalisation = similarité cosinus)
            # Score est déjà entre -1 et 1, on le normalise en 0-1
            # Convertir en float Python natif (FAISS retourne numpy.float32)
            normalized_score = float((score + 1) / 2)

            if normalized_score < min_confidence:
                continue

            meta = self.metadata[idx]
            result = RetrievalResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                score=normalized_score,
                table_name=meta["table_name"],
                row_id=meta["row_id"],
                localities=meta.get("localities", []),
                parties=meta.get("parties", []),
                source_type=meta.get("source_type", "unknown")
            )
            results.append(result)

        # Filtrer/booster par entités
        if query_entities["localities"] or query_entities["parties"]:
            results = self._filter_by_entities(results, query_entities)

        # Limiter aux top_k
        results = results[:top_k]

        top_score = results[0].score if results else 0
        logger.info(
            f"Recherche '{query[:50]}...': {len(results)} résultats "
            f"(top score: {top_score:.3f})"
        )

        return results

    def search_with_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> dict:
        """
        Recherche et retourne les résultats avec contexte formaté.

        Args:
            query: Requête en langage naturel
            top_k: Nombre de résultats

        Returns:
            Dictionnaire avec résultats et contexte
        """
        results = self.search(query, top_k)

        # Formater le contexte pour le LLM
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}] (confiance: {r.score:.2f})\n{r.text}"
            )

        context = "\n\n".join(context_parts)

        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "context": context,
            "num_results": len(results)
        }


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        retriever = RAGRetriever()

        test_queries = [
            "Résultats à Tiapoum",
            "Qui a gagné à Abidjan ?",
            "Performance du RHDP",
            "Score des indépendants à Bouaké",
            "Candidats élus dans le nord",
        ]

        print("=" * 60)
        print("TEST DU RETRIEVER RAG")
        print("=" * 60)

        for query in test_queries:
            print(f"\n--- Query: {query} ---")
            results = retriever.search(query, top_k=3)

            for i, r in enumerate(results, 1):
                print(f"\n  [{i}] Score: {r.score:.3f}")
                print(f"      Type: {r.source_type}")
                print(f"      Localités: {r.localities}")
                print(f"      Partis: {r.parties}")
                print(f"      Texte: {r.text[:150]}...")

    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Exécutez d'abord: python rag_indexer.py")
