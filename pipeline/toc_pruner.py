import sys
import pathlib
_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging
import sqlite3
import os
from dataclasses import dataclass
from typing import Optional

import yaml

from pipeline.clarification_generator import DisambiguatedQuestion, ClarificationTree
from retrievers.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)


@dataclass
class PruningResult:
    """Résultat de l'élagage d'une DQ."""
    dq: DisambiguatedQuestion
    is_valid: bool
    route: str  # "sql" ou "rag"
    pruning_reason: Optional[str]
    relevance_score: float  # 0.0-1.0
    schema_compatible: bool
    sql_feasible: bool
    rag_feasible: bool


@dataclass
class PrunedTree:
    """Arbre élagué avec DQs validées et routées."""
    original_query: str
    valid_dqs: list[PruningResult]
    pruned_dqs: list[PruningResult]
    pruning_metadata: dict

    def get_sql_dqs(self) -> list[PruningResult]:
        """Retourne les DQs routées vers SQL."""
        return [dq for dq in self.valid_dqs if dq.route == "sql"]

    def get_rag_dqs(self) -> list[PruningResult]:
        """Retourne les DQs routées vers RAG."""
        return [dq for dq in self.valid_dqs if dq.route == "rag"]


class ToCPruner:
    """
    Élague et valide les clarifications générées.
    Vérifie la faisabilité SQL et RAG de chaque DQ.
    """

    KNOWN_SCHEMA = {
        "circonscriptions": [
            "id", "region", "code", "nom", "nb_bv", "inscrits", "votants",
            "taux_participation", "nuls", "exprimes", "bulletins_blancs",
            "taux_bulletins_blancs"
        ],
        "candidats": [
            "id", "circonscription_id", "region", "parti", "nom",
            "score", "pourcentage", "elu"
        ]
    }

    SQL_METRICS = [
        "count", "sum", "avg", "max", "min", "total",
        "nombre", "combien", "moyenne", "somme", "maximum", "minimum",
        "classement", "top", "ranking", "pourcentage", "taux"
    ]

    RAG_CONCEPTS = [
        "explique", "décris", "pourquoi", "comment", "contexte",
        "historique", "analyse", "comparaison", "tendance",
        # Recherche par localité/entité
        "résultats", "resultats", "résultat", "resultat",
        "performance", "performances", "situation",
        "dans", "à", "au", "aux", "de", "du",
        "informations", "info", "détails", "details"
    ]

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le pruner.

        Args:
            config_path: Chemin vers la configuration
        """
        self.config = self._load_config(config_path)

        toc_config = self.config.get("toc", {})
        self.min_confidence = toc_config.get("min_confidence", 0.3)
        self.min_relevance = toc_config.get("min_relevance", 0.4)

        db_config = self.config.get("database", {})
        self.db_path = db_config.get("path", "etl/elections.db")

        self.intent_classifier = IntentClassifier(config_path)

        self._load_real_schema()

        logger.info("ToCPruner initialisé")

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _load_real_schema(self):
        """Charge le schéma réel de la base de données."""
        if not os.path.exists(self.db_path):
            logger.warning(f"Base de données non trouvée: {self.db_path}")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info('{table}')")
                columns = [row[1] for row in cursor.fetchall()]
                self.KNOWN_SCHEMA[table] = columns

            conn.close()
            logger.info(f"Schéma chargé: {list(self.KNOWN_SCHEMA.keys())}")

        except Exception as e:
            logger.error(f"Erreur chargement schéma: {e}")

    def _check_schema_compatibility(self, dq: DisambiguatedQuestion) -> tuple[bool, list[str]]:
        """
        Vérifie si la DQ est compatible avec le schéma.

        Returns:
            Tuple (is_compatible, list_of_issues)
        """
        issues = []
        question_lower = dq.explicit_question.lower()

        impossible_metrics = ["age", "sexe", "profession", "date_naissance", "email", "telephone"]
        for metric in impossible_metrics:
            if metric in question_lower:
                issues.append(f"Métrique non disponible: {metric}")

        return len(issues) == 0, issues

    def _check_sql_feasibility(self, dq: DisambiguatedQuestion) -> tuple[bool, str]:
        """
        Vérifie si la DQ peut être répondue par SQL.

        Returns:
            Tuple (is_feasible, reason)
        """
        question_lower = dq.explicit_question.lower()

        has_sql_metric = any(metric in question_lower for metric in self.SQL_METRICS)

        numeric_cols = ["score", "pourcentage", "inscrits", "votants", "taux", "nb_bv", "nuls", "exprimes"]
        has_numeric_ref = any(col in question_lower for col in numeric_cols)

        is_listing = any(word in question_lower for word in ["liste", "lister", "quels", "quelles", "combien"])

        if has_sql_metric or has_numeric_ref or is_listing:
            return True, "Contient des métriques/colonnes SQL"

        return False, "Pas de métriques SQL détectées"

    def _check_rag_feasibility(self, dq: DisambiguatedQuestion) -> tuple[bool, str]:
        """
        Vérifie si la DQ peut être répondue par RAG.
        Par défaut, RAG est toujours considéré comme faisable (fallback générique).

        Returns:
            Tuple (is_feasible, reason)
        """
        question_lower = dq.explicit_question.lower()

        has_rag_concept = any(concept in question_lower for concept in self.RAG_CONCEPTS)
        has_entity_search = bool(dq.entities.get("localities") or dq.entities.get("parties"))
        is_wh_question = any(word in question_lower for word in ["qui", "où", "quoi", "quel", "quelle"])

        known_regions = ["bagoue", "poro", "tchologo", "gbeke", "abidjan", "bouake",
                        "yamoussoukro", "korhogo", "daloa", "san-pedro", "gagnoa"]
        has_region_mention = any(region in question_lower for region in known_regions)

        if has_rag_concept or has_entity_search or is_wh_question or has_region_mention:
            return True, "Contient des concepts/entités RAG"

        # RAG est toujours faisable comme fallback (recherche sémantique générique)
        return True, "RAG disponible comme fallback sémantique"

    def _calculate_relevance(self, dq: DisambiguatedQuestion) -> float:
        """
        Calcule le score de pertinence d'une DQ par rapport à la requête originale.

        Returns:
            Score de pertinence (0.0-1.0)
        """
        score = dq.confidence

        if dq.entities.get("localities"):
            score += 0.1
        if dq.entities.get("parties"):
            score += 0.1
        if dq.entities.get("metrics"):
            score += 0.1

        if len(dq.assumptions) > 3:
            score -= 0.1 * (len(dq.assumptions) - 3)

        return max(0.0, min(1.0, score))

    def _determine_route(self, dq: DisambiguatedQuestion) -> str:
        """
        Détermine la route (SQL ou RAG) pour une DQ.
        Utilise le classifier d'intent existant.

        Returns:
            "sql" ou "rag"
        """
        intent_result = self.intent_classifier.classify(dq.explicit_question)
        return intent_result.intent

    def prune_dq(self, dq: DisambiguatedQuestion) -> PruningResult:
        """
        Évalue et potentiellement élague une DQ.

        Args:
            dq: DQ à évaluer

        Returns:
            PruningResult avec les résultats de validation
        """
        schema_ok, schema_issues = self._check_schema_compatibility(dq)
        sql_ok, sql_reason = self._check_sql_feasibility(dq)
        rag_ok, rag_reason = self._check_rag_feasibility(dq)

        relevance = self._calculate_relevance(dq)
        route = self._determine_route(dq)

        is_valid = True
        pruning_reason = None

        if dq.confidence < self.min_confidence:
            is_valid = False
            pruning_reason = f"Confiance trop basse: {dq.confidence:.2f} < {self.min_confidence}"

        elif relevance < self.min_relevance * 0.5:  # Plus tolérant
            is_valid = False
            pruning_reason = f"Pertinence trop basse: {relevance:.2f} < {self.min_relevance * 0.5}"

        elif not schema_ok:
            is_valid = False
            pruning_reason = f"Incompatible avec le schéma: {', '.join(schema_issues)}"

        # Note: On ne rejette plus sur "ni SQL ni RAG" car RAG est toujours disponible

        if route == "sql" and not sql_ok:
            route = "rag"
            logger.debug(f"DQ {dq.dq_id}: basculé SQL -> RAG")
        elif route == "rag" and not rag_ok and sql_ok:
            route = "sql"
            logger.debug(f"DQ {dq.dq_id}: basculé RAG -> SQL")

        return PruningResult(
            dq=dq,
            is_valid=is_valid,
            route=route,
            pruning_reason=pruning_reason,
            relevance_score=relevance,
            schema_compatible=schema_ok,
            sql_feasible=sql_ok,
            rag_feasible=rag_ok
        )

    def prune(self, tree: ClarificationTree) -> PrunedTree:
        """
        Élague un arbre de clarifications complet.

        Args:
            tree: Arbre de clarifications à élaguer

        Returns:
            PrunedTree avec DQs validées et élaguées
        """
        logger.info(f"Élagage de {len(tree.disambiguated_questions)} DQs...")

        valid_dqs = []
        pruned_dqs = []

        for dq in tree.disambiguated_questions:
            result = self.prune_dq(dq)

            if result.is_valid:
                valid_dqs.append(result)
                logger.debug(f"DQ valide: {dq.dq_id} -> {result.route}")
            else:
                pruned_dqs.append(result)
                logger.debug(f"DQ élaguée: {dq.dq_id} - {result.pruning_reason}")

        valid_dqs = self._deduplicate(valid_dqs)
        valid_dqs.sort(key=lambda x: x.relevance_score, reverse=True)

        pruned_tree = PrunedTree(
            original_query=tree.original_query,
            valid_dqs=valid_dqs,
            pruned_dqs=pruned_dqs,
            pruning_metadata={
                "total_dqs": len(tree.disambiguated_questions),
                "valid_count": len(valid_dqs),
                "pruned_count": len(pruned_dqs),
                "sql_routes": len([d for d in valid_dqs if d.route == "sql"]),
                "rag_routes": len([d for d in valid_dqs if d.route == "rag"])
            }
        )

        logger.info(
            f"Élagage terminé: {len(valid_dqs)} valides, {len(pruned_dqs)} élaguées "
            f"(SQL: {pruned_tree.pruning_metadata['sql_routes']}, "
            f"RAG: {pruned_tree.pruning_metadata['rag_routes']})"
        )

        return pruned_tree

    def _deduplicate(self, results: list[PruningResult]) -> list[PruningResult]:
        """
        Déduplique les DQs similaires.
        Garde celle avec la meilleure pertinence.
        """
        seen_questions = {}

        for result in results:
            normalized = result.dq.explicit_question.lower().strip()

            if normalized not in seen_questions:
                seen_questions[normalized] = result
            elif result.relevance_score > seen_questions[normalized].relevance_score:
                seen_questions[normalized] = result

        return list(seen_questions.values())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from clarification_generator import ClarificationGenerator

    generator = ClarificationGenerator()
    pruner = ToCPruner()

    test_query = "Résultats du RHDP à Abidjan"

    print("=" * 60)
    print("TEST DU PRUNER ToC")
    print("=" * 60)
    print(f"Query: {test_query}")

    tree = generator.generate(test_query)
    print(f"\nDQs générées: {len(tree.disambiguated_questions)}")

    pruned_tree = pruner.prune(tree)

    print(f"\n--- DQs Valides ({len(pruned_tree.valid_dqs)}) ---")
    for pr in pruned_tree.valid_dqs:
        print(f"  [{pr.dq.dq_id}] Route: {pr.route}, Relevance: {pr.relevance_score:.2f}")
        print(f"    Question: {pr.dq.explicit_question}")

    print(f"\n--- DQs Élaguées ({len(pruned_tree.pruned_dqs)}) ---")
    for pr in pruned_tree.pruned_dqs:
        print(f"  [{pr.dq.dq_id}] Raison: {pr.pruning_reason}")
