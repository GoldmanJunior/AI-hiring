"""
Normalisation d'entités avec fuzzy matching.
Gère les typos, accents, et aliases pour localités et partis.
"""

import json
import logging
import os
import re
from typing import Optional

from unidecode import unidecode
from rapidfuzz import fuzz, process

import yaml

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """
    Normaliseur d'entités pour localités et partis politiques.
    Utilise fuzzy matching avec rapidfuzz pour gérer les typos.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le normaliseur.

        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)

        # Charger les dictionnaires d'aliases
        norm_config = self.config.get("normalization", {})
        self.fuzzy_threshold = norm_config.get("fuzzy_threshold", 85)

        localities_path = norm_config.get(
            "localities_aliases",
            "aliases/localities_aliases.json"
        )
        parties_path = norm_config.get(
            "parties_aliases",
            "aliases/parties_aliases.json"
        )

        self.localities_aliases = self._load_aliases(localities_path)
        self.parties_aliases = self._load_aliases(parties_path)

        # Construire les index inversés (alias -> canonical)
        self.localities_index = self._build_index(self.localities_aliases)
        self.parties_index = self._build_index(self.parties_aliases)

        # Liste de tous les noms connus pour fuzzy matching
        self.all_localities = list(self.localities_index.keys())
        self.all_parties = list(self.parties_index.keys())

        logger.info(
            f"EntityNormalizer initialisé: {len(self.localities_aliases)} localités, "
            f"{len(self.parties_aliases)} partis"
        )

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config non trouvée: {config_path}")
            return {}

    def _load_aliases(self, path: str) -> dict[str, list[str]]:
        """Charge un dictionnaire d'aliases depuis JSON."""
        if not os.path.exists(path):
            logger.warning(f"Fichier d'aliases non trouvé: {path}")
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Filtrer les commentaires
                return {k: v for k, v in data.items() if not k.startswith("_")}
        except json.JSONDecodeError as e:
            logger.error(f"Erreur JSON dans {path}: {e}")
            return {}

    def _build_index(self, aliases: dict[str, list[str]]) -> dict[str, str]:
        """
        Construit un index inversé: alias -> forme canonique.

        Args:
            aliases: Dictionnaire {canonical: [alias1, alias2, ...]}

        Returns:
            Dictionnaire {alias_normalized: canonical}
        """
        index = {}
        for canonical, alias_list in aliases.items():
            # Ajouter la forme canonique elle-même
            index[self._normalize_basic(canonical)] = canonical

            # Ajouter tous les aliases
            for alias in alias_list:
                index[self._normalize_basic(alias)] = canonical

        return index

    def _normalize_basic(self, text: str) -> str:
        """
        Normalisation basique: lowercase, unidecode, strip.

        Args:
            text: Texte à normaliser

        Returns:
            Texte normalisé
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower().strip()

        # Supprimer les accents
        text = unidecode(text)

        # Supprimer les caractères spéciaux sauf tirets et apostrophes
        text = re.sub(r"[^\w\s'-]", "", text)

        # Normaliser les espaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _fuzzy_match(
        self,
        text: str,
        candidates: list[str],
        threshold: int = None
    ) -> Optional[str]:
        """
        Trouve la meilleure correspondance fuzzy.

        Args:
            text: Texte à matcher
            candidates: Liste de candidats
            threshold: Seuil de similarité (0-100)

        Returns:
            Meilleure correspondance ou None
        """
        if not text or not candidates:
            return None

        threshold = threshold or self.fuzzy_threshold

        # Utiliser rapidfuzz pour trouver la meilleure correspondance
        result = process.extractOne(
            text,
            candidates,
            scorer=fuzz.WRatio,
            score_cutoff=threshold
        )

        if result:
            match, score, _ = result
            logger.debug(f"Fuzzy match: '{text}' -> '{match}' (score: {score})")
            return match

        return None

    def normalize_locality(self, name: str) -> Optional[str]:
        """
        Normalise un nom de localité.

        Args:
            name: Nom de localité (potentiellement mal orthographié)

        Returns:
            Forme canonique ou None si non trouvé
        """
        if not name:
            return None

        normalized = self._normalize_basic(name)

        # Recherche exacte dans l'index
        if normalized in self.localities_index:
            return self.localities_index[normalized]

        # Fuzzy matching
        match = self._fuzzy_match(normalized, self.all_localities)
        if match:
            return self.localities_index.get(match)

        logger.debug(f"Localité non reconnue: '{name}'")
        return None

    def normalize_party(self, name: str) -> Optional[str]:
        """
        Normalise un nom de parti politique.

        Args:
            name: Nom de parti (potentiellement mal orthographié)

        Returns:
            Forme canonique ou None si non trouvé
        """
        if not name:
            return None

        normalized = self._normalize_basic(name)

        # Recherche exacte dans l'index
        if normalized in self.parties_index:
            return self.parties_index[normalized]

        # Fuzzy matching
        match = self._fuzzy_match(normalized, self.all_parties)
        if match:
            return self.parties_index.get(match)

        logger.debug(f"Parti non reconnu: '{name}'")
        return None

    def extract_entities(self, query: str) -> dict[str, list[str]]:
        """
        Extrait et normalise les entités d'une requête.

        Args:
            query: Requête en langage naturel

        Returns:
            Dictionnaire {type: [entités normalisées]}
        """
        entities = {
            "localities": [],
            "parties": []
        }

        # Tokeniser la requête
        words = re.split(r'\s+', query)

        # Essayer de matcher des séquences de mots (1 à 3 mots)
        for i in range(len(words)):
            for length in range(3, 0, -1):
                if i + length > len(words):
                    continue

                phrase = " ".join(words[i:i + length])

                # Essayer comme localité
                locality = self.normalize_locality(phrase)
                if locality and locality not in entities["localities"]:
                    entities["localities"].append(locality)

                # Essayer comme parti
                party = self.normalize_party(phrase)
                if party and party not in entities["parties"]:
                    entities["parties"].append(party)

        return entities

    def normalize_query(self, query: str) -> str:
        """
        Normalise une requête en remplaçant les entités par leurs formes canoniques.

        Args:
            query: Requête originale

        Returns:
            Requête avec entités normalisées
        """
        normalized_query = query

        # Extraire les entités
        entities = self.extract_entities(query)

        # Remplacer les localités
        for locality in entities["localities"]:
            # Trouver le texte original qui a matché
            for alias in self.localities_aliases.get(locality, [locality]):
                if alias.lower() in query.lower():
                    # Remplacer par la forme canonique (en préservant la casse)
                    pattern = re.compile(re.escape(alias), re.IGNORECASE)
                    normalized_query = pattern.sub(locality, normalized_query)
                    break

        # Remplacer les partis
        for party in entities["parties"]:
            for alias in self.parties_aliases.get(party, [party]):
                if alias.lower() in query.lower():
                    pattern = re.compile(re.escape(alias), re.IGNORECASE)
                    normalized_query = pattern.sub(party, normalized_query)
                    break

        if normalized_query != query:
            logger.info(f"Query normalisée: '{query}' -> '{normalized_query}'")

        return normalized_query

    def get_all_localities(self) -> list[str]:
        """Retourne toutes les formes canoniques de localités."""
        return list(self.localities_aliases.keys())

    def get_all_parties(self) -> list[str]:
        """Retourne toutes les formes canoniques de partis."""
        return list(self.parties_aliases.keys())


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    normalizer = EntityNormalizer()

    print("=" * 60)
    print("TEST DE NORMALISATION D'ENTITÉS")
    print("=" * 60)

    # Test localités
    test_localities = [
        "Tiapoum",      # Exact
        "Tiapum",       # Typo
        "ABIDJAN",      # Majuscules
        "Cote d Ivoire", # Sans accent
        "bouaké",       # Accent
        "Youpougon",    # Typo
        "inconnu123",   # Non trouvé
    ]

    print("\n--- Localités ---")
    for loc in test_localities:
        result = normalizer.normalize_locality(loc)
        print(f"  '{loc}' -> '{result}'")

    # Test partis
    test_parties = [
        "RHDP",         # Exact
        "R.H.D.P",      # Avec points
        "pdci-rda",     # Lowercase
        "independant",  # Sans accent
        "INDÉPENDANT",  # Avec accent
        "Front Populaire Ivoirien",  # Nom complet
        "inconnu",      # Non trouvé
    ]

    print("\n--- Partis ---")
    for party in test_parties:
        result = normalizer.normalize_party(party)
        print(f"  '{party}' -> '{result}'")

    # Test extraction d'entités
    test_queries = [
        "Résultats à Tiapum",
        "Performance du R.H.D.P à Abidjan",
        "Qui a gagné à bouaké ?",
        "Score de l'independant dans le nord",
    ]

    print("\n--- Extraction d'entités ---")
    for query in test_queries:
        entities = normalizer.extract_entities(query)
        print(f"  '{query}'")
        print(f"    Localités: {entities['localities']}")
        print(f"    Partis: {entities['parties']}")

    # Test normalisation de requête
    print("\n--- Normalisation de requêtes ---")
    for query in test_queries:
        normalized = normalizer.normalize_query(query)
        print(f"  '{query}'")
        print(f"  -> '{normalized}'")
