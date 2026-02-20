"""
Classification d'intent par mots-clés (SANS LLM).
Détermine si une question doit être routée vers SQL ou RAG.
"""

import re
import logging
from dataclasses import dataclass
from typing import Literal

import yaml

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Résultat de la classification d'intent."""
    intent: Literal["sql", "rag"]
    score_sql: float
    score_rag: float
    confidence: float
    matched_keywords: dict[str, list[str]]


class IntentClassifier:
    """
    Classifieur d'intent basé sur des mots-clés et patterns.
    Aucun appel LLM - classification purement règle-based.
    """

    # Mots-clés déclencheurs SQL (analytiques, agrégation, comparaison)
    SQL_KEYWORDS = {
        "high": [
            # Analytiques
            "combien", "total", "somme", "moyenne", "count", "nombre",
            "classement", "top", "ranking", "maximum", "minimum",
            "sum", "avg", "max", "min", "count",
            # Comparaison
            "supérieur", "inférieur", "entre", "comparer",
            "plus que", "moins que", "greater", "less than",
            "plus de", "moins de", "au moins", "au plus",
            # Agrégation
            "par région", "par parti", "groupe par", "grouper par",
            "ensemble des", "tous les", "toutes les",
            "group by", "order by",
            # Statistiques
            "pourcentage", "taux", "ratio", "proportion",
            "statistique", "statistiques", "stats",
        ],
        "medium": [
            # Requêtes de liste
            "liste", "lister", "énumérer", "afficher",
            "quels sont", "quelles sont",
            # Filtres
            "filtrer", "sélectionner", "uniquement",
            "seulement", "où", "when", "where",
            # Tri
            "trier", "ordonner", "croissant", "décroissant",
            "meilleur", "pire", "premier", "dernier",
        ],
        "low": [
            # Verbes analytiques généraux
            "calculer", "déterminer", "évaluer",
            "mesurer", "quantifier",
        ]
    }

    # Mots-clés déclencheurs RAG (recherche, narratif, lookup)
    RAG_KEYWORDS = {
        "high": [
            # Recherche spécifique
            "qui est", "qui a", "qui sont",
            "résultats à", "résultats de", "résultat à", "résultat de",
            "performance à", "performance de",
            "gagné à", "perdu à", "élu à", "élue à",
            "score de", "voix de", "votes de",
            # Narratif
            "explique", "expliquer", "décris", "décrire",
            "comment", "pourquoi", "raconte",
            # Lookup par nom
            "informations sur", "info sur", "détails sur",
            "parle-moi de", "dis-moi sur",
        ],
        "medium": [
            # Recherche géographique
            "dans", "à", "au", "aux", "en",
            "région de", "localité de", "commune de",
            "circonscription de",
            # Recherche par entité
            "candidat", "parti", "liste",
            "député", "élu", "élue",
        ],
        "low": [
            # Questions ouvertes
            "quel", "quelle", "quoi",
            "où", "quand",
        ]
    }

    # Patterns regex pour détection avancée
    SQL_PATTERNS = [
        r"combien\s+de\s+\w+",  # "combien de candidats"
        r"top\s+\d+",  # "top 10"
        r"les\s+\d+\s+(premiers?|meilleurs?)",  # "les 5 premiers"
        r"plus\s+de\s+\d+",  # "plus de 50%"
        r"moins\s+de\s+\d+",  # "moins de 10"
        r"entre\s+\d+\s+et\s+\d+",  # "entre 20 et 50"
        r"par\s+(région|parti|commune|circonscription)",  # "par région"
        r"(total|somme|moyenne)\s+des?",  # "total des voix"
    ]

    RAG_PATTERNS = [
        r"résultats?\s+(à|de|du|dans)\s+\w+",  # "résultats à Tiapoum"
        r"qui\s+a\s+(gagné|perdu|obtenu)",  # "qui a gagné"
        r"(performance|score|voix)\s+(de|du|à)\s+\w+",  # "score de RHDP"
        r"élu[es]?\s+(à|de|dans)\s+\w+",  # "élu à Abidjan"
        r"candidat[es]?\s+(de|du|à)\s+\w+",  # "candidat de RHDP"
    ]

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le classifieur.

        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.weights = self.config.get("intent", {}).get("weights", {
            "high": 3.0,
            "medium": 1.5,
            "low": 0.5
        })
        self.sql_threshold = self.config.get("intent", {}).get("sql_threshold", 0.6)

        logger.info("IntentClassifier initialisé (classification par mots-clés)")

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config non trouvée: {config_path}, utilisation des défauts")
            return {}

    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour la comparaison."""
        text = text.lower().strip()
        # Supprimer la ponctuation sauf les apostrophes
        text = re.sub(r"[^\w\s'-]", " ", text)
        # Normaliser les espaces
        text = re.sub(r"\s+", " ", text)
        return text

    def _calculate_keyword_score(
        self,
        text: str,
        keywords: dict[str, list[str]]
    ) -> tuple[float, list[str]]:
        """
        Calcule le score basé sur les mots-clés trouvés.

        Args:
            text: Texte normalisé
            keywords: Dictionnaire de mots-clés par niveau

        Returns:
            Tuple (score, liste des mots-clés matchés)
        """
        score = 0.0
        matched = []

        for level, keyword_list in keywords.items():
            weight = self.weights.get(level, 1.0)
            for keyword in keyword_list:
                if keyword.lower() in text:
                    score += weight
                    matched.append(keyword)

        return score, matched

    def _calculate_pattern_score(
        self,
        text: str,
        patterns: list[str]
    ) -> tuple[float, list[str]]:
        """
        Calcule le score basé sur les patterns regex.

        Args:
            text: Texte normalisé
            patterns: Liste de patterns regex

        Returns:
            Tuple (score, liste des patterns matchés)
        """
        score = 0.0
        matched = []

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += self.weights["high"]  # Patterns = haute confiance
                matched.append(pattern)

        return score, matched

    def classify(self, question: str) -> IntentResult:
        """
        Classifie l'intent d'une question.

        Args:
            question: Question en langage naturel

        Returns:
            IntentResult avec l'intent déterminé et les scores
        """
        text = self._normalize_text(question)

        # Calculer les scores SQL
        sql_kw_score, sql_kw_matched = self._calculate_keyword_score(
            text, self.SQL_KEYWORDS
        )
        sql_pat_score, sql_pat_matched = self._calculate_pattern_score(
            text, self.SQL_PATTERNS
        )
        score_sql = sql_kw_score + sql_pat_score

        # Calculer les scores RAG
        rag_kw_score, rag_kw_matched = self._calculate_keyword_score(
            text, self.RAG_KEYWORDS
        )
        rag_pat_score, rag_pat_matched = self._calculate_pattern_score(
            text, self.RAG_PATTERNS
        )
        score_rag = rag_kw_score + rag_pat_score

        # Normaliser les scores
        total_score = score_sql + score_rag
        if total_score > 0:
            normalized_sql = score_sql / total_score
            normalized_rag = score_rag / total_score
        else:
            # Aucun mot-clé trouvé, défaut vers RAG (recherche générale)
            normalized_sql = 0.4
            normalized_rag = 0.6

        # Déterminer l'intent
        if normalized_sql >= self.sql_threshold:
            intent = "sql"
            confidence = normalized_sql
        else:
            intent = "rag"
            confidence = normalized_rag

        result = IntentResult(
            intent=intent,
            score_sql=round(normalized_sql, 3),
            score_rag=round(normalized_rag, 3),
            confidence=round(confidence, 3),
            matched_keywords={
                "sql": sql_kw_matched + sql_pat_matched,
                "rag": rag_kw_matched + rag_pat_matched
            }
        )

        logger.info(
            f"Classification: '{question[:50]}...' -> {intent} "
            f"(sql={result.score_sql}, rag={result.score_rag})"
        )

        return result

    def explain(self, question: str) -> str:
        """
        Explique la classification d'une question.

        Args:
            question: Question à expliquer

        Returns:
            Explication textuelle
        """
        result = self.classify(question)

        explanation = f"""
Classification de: "{question}"

Intent déterminé: {result.intent.upper()}
Confiance: {result.confidence * 100:.1f}%

Scores:
  - SQL: {result.score_sql * 100:.1f}%
  - RAG: {result.score_rag * 100:.1f}%

Mots-clés SQL détectés: {', '.join(result.matched_keywords['sql']) or 'aucun'}
Mots-clés RAG détectés: {', '.join(result.matched_keywords['rag']) or 'aucun'}

Seuil SQL: {self.sql_threshold * 100:.1f}%
Décision: {'SQL (score >= seuil)' if result.intent == 'sql' else 'RAG (score < seuil)'}
"""
        return explanation.strip()


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    classifier = IntentClassifier()

    test_questions = [
        # SQL attendu
        "Combien de partis ont plus de 10% ?",
        "Classement top 5 localités par participation",
        "Quelle est la moyenne des scores par région ?",
        "Total des voix pour le RHDP",
        "Les 10 premiers candidats par score",

        # RAG attendu
        "Résultats à Tiapoum",
        "Qui a gagné à Abidjan ?",
        "Performance du RHDP dans le nord",
        "Parle-moi du candidat Konan",
        "Score de l'indépendant à Bouaké",
    ]

    print("=" * 60)
    print("TEST DU CLASSIFIEUR D'INTENT")
    print("=" * 60)

    for q in test_questions:
        result = classifier.classify(q)
        print(f"\n'{q}'")
        print(f"  -> {result.intent.upper()} (conf: {result.confidence:.2f})")
        print(f"     SQL: {result.score_sql:.2f}, RAG: {result.score_rag:.2f}")
