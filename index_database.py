#!/usr/bin/env python
"""
Script d'indexation initiale pour le système RAG.
Crée l'index FAISS et les métadonnées à partir de la base SQLite.

Usage:
    python index_database.py
    python index_database.py --config custom_config.yaml
    python index_database.py --force  # Réindexer même si l'index existe
"""

import argparse
import logging
import os
import sys

from rag_indexer import RAGIndexer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    missing = []

    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")

    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    try:
        from unidecode import unidecode
    except ImportError:
        missing.append("unidecode")

    try:
        from rapidfuzz import fuzz
    except ImportError:
        missing.append("rapidfuzz")

    if missing:
        print("Dépendances manquantes:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstallez-les avec:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def check_database(config_path: str) -> bool:
    """Vérifie que la base de données existe."""
    import yaml

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    db_path = config.get("database", {}).get("path", "etl/elections.db")

    if not os.path.exists(db_path):
        print(f"Base de données non trouvée: {db_path}")
        print("\nAssurez-vous d'avoir:")
        print("  1. Extrait les données du PDF (extract_pdf_table.py)")
        print("  2. Nettoyé les CSV (etl/clean_csv.py)")
        print("  3. Chargé les données (etl/load_data.py)")
        return False

    return True


def check_index_exists(config_path: str) -> bool:
    """Vérifie si l'index existe déjà."""
    import yaml

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    faiss_config = config.get("faiss", {})
    index_path = faiss_config.get("index_path", "rag_index/vectors.faiss")
    metadata_path = faiss_config.get("metadata_path", "rag_index/metadata.pkl")

    return os.path.exists(index_path) and os.path.exists(metadata_path)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Indexation de la base de données pour le système RAG"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forcer la réindexation même si l'index existe"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Vérifier les prérequis sans indexer"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("INDEXATION RAG - Base de données électorale")
    print("=" * 60)

    # Vérifier les dépendances
    print("\n[1/4] Vérification des dépendances...")
    if not check_dependencies():
        sys.exit(1)
    print("  ✓ Toutes les dépendances sont installées")

    # Vérifier la base de données
    print("\n[2/4] Vérification de la base de données...")
    if not check_database(args.config):
        sys.exit(1)
    print("  ✓ Base de données trouvée")

    # Vérifier si l'index existe
    print("\n[3/4] Vérification de l'index existant...")
    if check_index_exists(args.config):
        if args.force:
            print("  ⚠ Index existant sera écrasé (--force)")
        elif args.check_only:
            print("  ✓ Index existant trouvé")
            print("\n✓ Tous les prérequis sont satisfaits!")
            return
        else:
            print("  ✓ Index existant trouvé")
            print("\nL'index existe déjà. Utilisez --force pour réindexer.")
            return
    else:
        print("  ⚠ Aucun index trouvé - création nécessaire")

    if args.check_only:
        print("\n⚠ Indexation nécessaire. Relancez sans --check-only")
        return

    # Indexation
    print("\n[4/4] Indexation en cours...")
    print("  (Cette opération peut prendre quelques minutes)")
    print()

    try:
        indexer = RAGIndexer(args.config)
        indexer.index_database()

        print("\n" + "=" * 60)
        print("✓ INDEXATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)
        print("\nVous pouvez maintenant utiliser le système hybride:")
        print("  python hybrid_router.py")

    except Exception as e:
        logger.error(f"Erreur lors de l'indexation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
