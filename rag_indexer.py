"""
Indexeur RAG avec FAISS et embeddings multilingues.
Crée des chunks textuels à partir de la base SQLite et les indexe.
"""

import os
import pickle
import sqlite3
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import yaml

from entity_normalizer import EntityNormalizer

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Métadonnées d'un chunk indexé."""
    chunk_id: int
    table_name: str
    row_id: int
    text: str
    localities: list[str]
    parties: list[str]
    source_type: str  # "circonscription" ou "candidat"

    def to_dict(self) -> dict:
        return asdict(self)


class RAGIndexer:
    """
    Indexeur RAG pour créer des embeddings et un index FAISS
    à partir des données de la base SQLite.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise l'indexeur.

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
        self.dimension = embedding_config.get("dimension", 384)
        self.batch_size = embedding_config.get("batch_size", 32)

        faiss_config = self.config.get("faiss", {})
        self.index_path = faiss_config.get("index_path", "rag_index/vectors.faiss")
        self.metadata_path = faiss_config.get("metadata_path", "rag_index/metadata.pkl")

        db_config = self.config.get("database", {})
        self.db_path = db_config.get("path", "etl/elections.db")

        # Charger le modèle d'embedding
        logger.info(f"Chargement du modèle d'embedding: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Normaliser d'entités
        self.normalizer = EntityNormalizer(config_path)

        # Stockage des chunks et métadonnées
        self.chunks: list[str] = []
        self.metadata: list[ChunkMetadata] = []

        logger.info("RAGIndexer initialisé")

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config non trouvée: {config_path}")
            return {}

    def _connect_db(self) -> sqlite3.Connection:
        """Établit la connexion à la base."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Base de données non trouvée: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_circonscription_chunk(self, row: dict) -> tuple[str, ChunkMetadata]:
        """
        Crée un chunk textuel pour une circonscription.

        Args:
            row: Dictionnaire avec les données de la circonscription

        Returns:
            Tuple (texte du chunk, métadonnées)
        """
        # Construire le texte descriptif
        region = row.get("region") or "Non spécifiée"
        nom = row.get("nom") or "Inconnue"
        inscrits = row.get("inscrits") or 0
        votants = row.get("votants") or 0
        participation = row.get("taux_participation") or 0
        exprimes = row.get("exprimes") or 0
        nuls = row.get("nuls") or 0

        text = (
            f"Circonscription {nom} dans la région {region}. "
            f"Inscrits: {inscrits:,}, Votants: {votants:,}, "
            f"Taux de participation: {participation:.2f}%. "
            f"Suffrages exprimés: {exprimes:,}, Bulletins nuls: {nuls:,}."
        )

        # Normaliser les entités
        localities = []
        if nom:
            normalized_loc = self.normalizer.normalize_locality(nom)
            if normalized_loc:
                localities.append(normalized_loc)
        if region:
            normalized_reg = self.normalizer.normalize_locality(region)
            if normalized_reg:
                localities.append(normalized_reg)

        metadata = ChunkMetadata(
            chunk_id=len(self.chunks),
            table_name="circonscriptions",
            row_id=row.get("id"),
            text=text,
            localities=localities,
            parties=[],
            source_type="circonscription"
        )

        return text, metadata

    def _create_candidat_chunk(
        self,
        row: dict,
        circonscription: Optional[dict] = None
    ) -> tuple[str, ChunkMetadata]:
        """
        Crée un chunk textuel pour un candidat.

        Args:
            row: Dictionnaire avec les données du candidat
            circonscription: Infos de la circonscription associée

        Returns:
            Tuple (texte du chunk, métadonnées)
        """
        nom = row.get("nom") or "Inconnu"
        parti = row.get("parti") or "Sans parti"
        score = row.get("score") or 0
        pourcentage = row.get("pourcentage") or 0
        elu = "ÉLU" if row.get("elu") else "Non élu"

        circ_nom = "Inconnue"
        region = "Non spécifiée"
        if circonscription:
            circ_nom = circonscription.get("nom", "Inconnue")
            region = circonscription.get("region", "Non spécifiée")

        text = (
            f"Candidat {nom} du parti {parti} dans la circonscription {circ_nom} "
            f"(région {region}). "
            f"Score: {score:,} voix ({pourcentage:.2f}%). "
            f"Statut: {elu}."
        )

        # Normaliser les entités
        localities = []
        if circ_nom:
            normalized_loc = self.normalizer.normalize_locality(circ_nom)
            if normalized_loc:
                localities.append(normalized_loc)
        if region:
            normalized_reg = self.normalizer.normalize_locality(region)
            if normalized_reg:
                localities.append(normalized_reg)

        parties = []
        if parti:
            normalized_party = self.normalizer.normalize_party(parti)
            if normalized_party:
                parties.append(normalized_party)

        metadata = ChunkMetadata(
            chunk_id=len(self.chunks),
            table_name="candidats",
            row_id=row.get("id"),
            text=text,
            localities=localities,
            parties=parties,
            source_type="candidat"
        )

        return text, metadata

    def _create_aggregated_chunk(
        self,
        circonscription: dict,
        candidats: list[dict]
    ) -> tuple[str, ChunkMetadata]:
        """
        Crée un chunk agrégé pour une circonscription avec tous ses candidats.

        Args:
            circonscription: Données de la circonscription
            candidats: Liste des candidats de cette circonscription

        Returns:
            Tuple (texte du chunk, métadonnées)
        """
        nom = circonscription.get("nom", "Inconnue")
        region = circonscription.get("region", "Non spécifiée")
        participation = circonscription.get("taux_participation", 0)

        # Construire le résumé des candidats
        candidats_text = []
        parties = []
        for c in sorted(candidats, key=lambda x: x.get("score", 0) or 0, reverse=True):
            c_nom = c.get("nom", "Inconnu")
            c_parti = c.get("parti", "?")
            c_score = c.get("score", 0) or 0
            c_pct = c.get("pourcentage", 0) or 0
            c_elu = " [ÉLU]" if c.get("elu") else ""

            candidats_text.append(f"{c_nom} ({c_parti}): {c_pct:.2f}%{c_elu}")

            normalized_party = self.normalizer.normalize_party(c_parti)
            if normalized_party and normalized_party not in parties:
                parties.append(normalized_party)

        text = (
            f"Résultats électoraux à {nom} ({region}). "
            f"Participation: {participation:.2f}%. "
            f"Candidats: {'; '.join(candidats_text[:10])}."  # Limiter à 10
        )

        # Normaliser les localités
        localities = []
        normalized_loc = self.normalizer.normalize_locality(nom)
        if normalized_loc:
            localities.append(normalized_loc)
        normalized_reg = self.normalizer.normalize_locality(region)
        if normalized_reg and normalized_reg not in localities:
            localities.append(normalized_reg)

        metadata = ChunkMetadata(
            chunk_id=len(self.chunks),
            table_name="agregation",
            row_id=circonscription.get("id", 0),
            text=text,
            localities=localities,
            parties=parties,
            source_type="agregation"
        )

        return text, metadata

    def build_chunks(self):
        """Construit tous les chunks à partir de la base de données."""
        logger.info("Construction des chunks...")

        conn = self._connect_db()

        try:
            # 1. Chunks pour chaque circonscription
            cursor = conn.execute("SELECT * FROM circonscriptions")
            circonscriptions = {row["id"]: dict(row) for row in cursor.fetchall()}

            for circ_id, circ_data in circonscriptions.items():
                text, metadata = self._create_circonscription_chunk(circ_data)
                self.chunks.append(text)
                self.metadata.append(metadata)

            logger.info(f"Chunks circonscriptions: {len(circonscriptions)}")

            # 2. Chunks pour chaque candidat
            cursor = conn.execute("SELECT * FROM candidats")
            candidats_by_circ: dict[int, list[dict]] = {}

            for row in cursor.fetchall():
                candidat_data = dict(row)
                circ_id = candidat_data.get("circonscription_id")

                # Chunk individuel pour le candidat
                circ_data = circonscriptions.get(circ_id, {})
                text, metadata = self._create_candidat_chunk(candidat_data, circ_data)
                self.chunks.append(text)
                self.metadata.append(metadata)

                # Grouper par circonscription pour chunks agrégés
                if circ_id not in candidats_by_circ:
                    candidats_by_circ[circ_id] = []
                candidats_by_circ[circ_id].append(candidat_data)

            logger.info(f"Chunks candidats: {len(self.chunks) - len(circonscriptions)}")

            # 3. Chunks agrégés par circonscription
            for circ_id, candidats in candidats_by_circ.items():
                circ_data = circonscriptions.get(circ_id, {"id": circ_id})
                text, metadata = self._create_aggregated_chunk(circ_data, candidats)
                self.chunks.append(text)
                self.metadata.append(metadata)

            logger.info(f"Chunks agrégés: {len(candidats_by_circ)}")
            logger.info(f"Total chunks: {len(self.chunks)}")

        finally:
            conn.close()

    def build_index(self) -> faiss.Index:
        """
        Construit l'index FAISS à partir des chunks.

        Returns:
            Index FAISS
        """
        if not self.chunks:
            raise ValueError("Aucun chunk à indexer. Appelez build_chunks() d'abord.")

        logger.info(f"Création des embeddings pour {len(self.chunks)} chunks...")

        # Créer les embeddings par batch
        embeddings = self.model.encode(
            self.chunks,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normaliser les vecteurs (pour similarité cosinus avec IndexFlatIP)
        faiss.normalize_L2(embeddings)

        # Créer l'index FAISS
        logger.info(f"Construction de l'index FAISS (dimension: {self.dimension})...")
        index = faiss.IndexFlatIP(self.dimension)  # Inner Product après normalisation = cosinus
        index.add(embeddings.astype(np.float32))

        logger.info(f"Index créé avec {index.ntotal} vecteurs")
        return index

    def save(self, index: faiss.Index):
        """
        Sauvegarde l'index et les métadonnées.

        Args:
            index: Index FAISS à sauvegarder
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Sauvegarder l'index FAISS
        faiss.write_index(index, self.index_path)
        logger.info(f"Index sauvegardé: {self.index_path}")

        # Sauvegarder les métadonnées
        metadata_dicts = [m.to_dict() for m in self.metadata]
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                "metadata": metadata_dicts,
                "chunks": self.chunks
            }, f)
        logger.info(f"Métadonnées sauvegardées: {self.metadata_path}")

    def index_database(self):
        """Pipeline complet d'indexation."""
        logger.info("=" * 50)
        logger.info("Démarrage de l'indexation RAG")
        logger.info("=" * 50)

        # 1. Construire les chunks
        self.build_chunks()

        # 2. Construire l'index
        index = self.build_index()

        # 3. Sauvegarder
        self.save(index)

        logger.info("=" * 50)
        logger.info("Indexation terminée avec succès!")
        logger.info(f"  - Chunks indexés: {len(self.chunks)}")
        logger.info(f"  - Index: {self.index_path}")
        logger.info(f"  - Métadonnées: {self.metadata_path}")
        logger.info("=" * 50)


# Script d'exécution standalone
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    indexer = RAGIndexer()
    indexer.index_database()
