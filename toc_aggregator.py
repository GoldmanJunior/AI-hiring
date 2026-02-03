"""
ToC Aggregator - Agrège les résultats de multiples DQs.
Combine les résultats SQL et RAG en une réponse cohérente.
"""

import os
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from groq import Groq
from dotenv import load_dotenv
import yaml

from toc_pruner import PruningResult, PrunedTree
from citation_manager import Citation, CitedResponse

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class DQResult:
    """Résultat d'une DQ individuelle."""
    dq_id: str
    route: str
    question: str
    answer: str
    success: bool
    sources: list[Citation]
    sql_query: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AggregatedResponse:
    """Réponse agrégée de toutes les DQs."""
    original_query: str
    final_answer: str
    interpretations: list[DQResult]
    sql_facts: list[str]
    rag_insights: list[str]
    sources: list[dict]
    confidence: float
    method: str  # "toc" for Tree of Clarifications
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "final_answer": self.final_answer,
            "interpretations": [
                {
                    "dq_id": i.dq_id,
                    "route": i.route,
                    "question": i.question,
                    "answer": i.answer,
                    "success": i.success
                }
                for i in self.interpretations
            ],
            "sql_facts": self.sql_facts,
            "rag_insights": self.rag_insights,
            "sources": self.sources,
            "confidence": self.confidence,
            "method": self.method,
            "metadata": self.metadata
        }


class ToCAggregator:
    """
    Agrège les résultats de multiples DQs en une réponse cohérente.
    Utilise un LLM pour synthétiser les informations.
    """

    AGGREGATION_PROMPT = """Tu es un assistant expert en analyse de données électorales.

CONTEXTE:
L'utilisateur a posé une question qui a été interprétée de plusieurs façons.
Chaque interprétation a été répondue séparément (via SQL ou RAG).

QUESTION ORIGINALE:
"{original_query}"

RÉSULTATS PAR INTERPRÉTATION:
{interpretations}

TÂCHE:
Synthétise toutes ces réponses en une réponse unique, complète et cohérente.

RÈGLES:
1. Commence par la réponse la plus pertinente/directe
2. Intègre les informations complémentaires des autres interprétations
3. Distingue clairement les FAITS (issus de SQL) des ANALYSES (issues de RAG)
4. Si des interprétations sont contradictoires, mentionne-le
5. Sois concis mais complet
6. Utilise des listes ou tableaux si approprié
7. Mentionne les sources de manière naturelle

FORMAT:
Réponds directement à l'utilisateur de manière naturelle et informative.
Ne mentionne pas le processus interne de désambiguïsation."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise l'agrégateur.

        Args:
            config_path: Chemin vers la configuration
        """
        self.config = self._load_config(config_path)

        # Configuration Groq
        groq_config = self.config.get("groq", {})
        self.model = groq_config.get("model", "llama-3.3-70b-versatile")
        self.temperature = groq_config.get("temperature", 0.3)

        # Client Groq
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
        else:
            self.groq_client = None
            logger.warning("GROQ_API_KEY non définie - agrégation LLM désactivée")

        logger.info("ToCAggregator initialisé")

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _format_interpretations(self, results: list[DQResult]) -> str:
        """Formate les résultats pour le prompt d'agrégation."""
        parts = []

        for i, result in enumerate(results, 1):
            status = "✓" if result.success else "✗"
            route_label = "SQL (données)" if result.route == "sql" else "RAG (contexte)"

            part = f"""
--- Interprétation {i} [{result.dq_id}] ({route_label}) {status} ---
Question: {result.question}
Réponse: {result.answer if result.success else f"Erreur: {result.error}"}
"""
            if result.sql_query:
                part += f"SQL exécuté: {result.sql_query}\n"

            parts.append(part)

        return "\n".join(parts)

    def _extract_facts_and_insights(self, results: list[DQResult]) -> tuple[list[str], list[str]]:
        """
        Extrait les faits SQL et insights RAG des résultats.

        Returns:
            Tuple (sql_facts, rag_insights)
        """
        sql_facts = []
        rag_insights = []

        for result in results:
            if not result.success:
                continue

            if result.route == "sql":
                # Extraire des faits concis de la réponse SQL
                sql_facts.append(f"{result.question}: {result.answer[:200]}")
            else:
                # Extraire des insights de la réponse RAG
                rag_insights.append(f"{result.question}: {result.answer[:200]}")

        return sql_facts, rag_insights

    def _collect_sources(self, results: list[DQResult]) -> list[dict]:
        """Collecte et déduplique les sources de tous les résultats."""
        all_sources = []
        seen = set()

        for result in results:
            for source in result.sources:
                # Créer une clé unique pour déduplication
                key = f"{source.table_name}:{source.row_id}"
                if key not in seen:
                    seen.add(key)
                    all_sources.append(source.to_dict())

        return all_sources

    def _calculate_confidence(self, results: list[DQResult]) -> float:
        """Calcule le score de confiance global."""
        if not results:
            return 0.0

        successful = sum(1 for r in results if r.success)
        return successful / len(results)

    def _synthesize_with_llm(
        self,
        original_query: str,
        results: list[DQResult]
    ) -> str:
        """
        Utilise le LLM pour synthétiser les résultats.

        Returns:
            Réponse synthétisée
        """
        if not self.groq_client:
            return self._synthesize_simple(results)

        interpretations_text = self._format_interpretations(results)

        prompt = self.AGGREGATION_PROMPT.format(
            original_query=original_query,
            interpretations=interpretations_text
        )

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Erreur synthèse LLM: {e}")
            return self._synthesize_simple(results)

    def _synthesize_simple(self, results: list[DQResult]) -> str:
        """
        Synthèse simple sans LLM (fallback).

        Returns:
            Réponse combinée simple
        """
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return "Aucune réponse n'a pu être générée pour cette question."

        parts = []
        for result in successful_results:
            parts.append(f"**{result.question}**\n{result.answer}")

        return "\n\n".join(parts)

    def aggregate(
        self,
        original_query: str,
        results: list[DQResult]
    ) -> AggregatedResponse:
        """
        Agrège les résultats de multiples DQs.

        Args:
            original_query: Question originale
            results: Liste des résultats de chaque DQ

        Returns:
            AggregatedResponse avec la réponse finale
        """
        logger.info(f"Agrégation de {len(results)} résultats...")

        # Extraire faits et insights
        sql_facts, rag_insights = self._extract_facts_and_insights(results)

        # Collecter les sources
        sources = self._collect_sources(results)

        # Calculer la confiance
        confidence = self._calculate_confidence(results)

        # Synthétiser la réponse finale
        if len(results) == 1:
            # Un seul résultat, pas besoin de synthèse
            final_answer = results[0].answer if results[0].success else f"Erreur: {results[0].error}"
        else:
            # Plusieurs résultats, synthétiser avec LLM
            final_answer = self._synthesize_with_llm(original_query, results)

        response = AggregatedResponse(
            original_query=original_query,
            final_answer=final_answer,
            interpretations=results,
            sql_facts=sql_facts,
            rag_insights=rag_insights,
            sources=sources,
            confidence=confidence,
            method="toc",
            metadata={
                "num_interpretations": len(results),
                "successful": sum(1 for r in results if r.success),
                "sql_routes": sum(1 for r in results if r.route == "sql"),
                "rag_routes": sum(1 for r in results if r.route == "rag")
            }
        )

        logger.info(
            f"Agrégation terminée: confiance={confidence:.2f}, "
            f"SQL={response.metadata['sql_routes']}, RAG={response.metadata['rag_routes']}"
        )

        return response

    def format_response(self, response: AggregatedResponse) -> str:
        """
        Formate la réponse agrégée pour affichage.

        Args:
            response: Réponse agrégée

        Returns:
            Texte formaté
        """
        lines = [
            f"=== Réponse (confiance: {response.confidence*100:.0f}%) ===\n",
            response.final_answer,
        ]

        if response.sql_facts:
            lines.append("\n--- Faits (SQL) ---")
            for fact in response.sql_facts[:5]:
                lines.append(f"• {fact[:100]}...")

        if response.rag_insights:
            lines.append("\n--- Analyses (RAG) ---")
            for insight in response.rag_insights[:5]:
                lines.append(f"• {insight[:100]}...")

        if response.sources:
            lines.append(f"\n--- Sources ({len(response.sources)}) ---")
            for source in response.sources[:5]:
                lines.append(f"• {source.get('table_name', '?')}: {source.get('excerpt', '')[:50]}...")

        return "\n".join(lines)


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    aggregator = ToCAggregator()

    # Simuler des résultats
    from citation_manager import Citation

    test_results = [
        DQResult(
            dq_id="DQ1",
            route="sql",
            question="Combien de candidats RHDP ont été élus ?",
            answer="155 candidats RHDP ont été élus.",
            success=True,
            sources=[Citation(
                source_type="sql",
                table_name="candidats",
                row_id=None,
                excerpt="COUNT(*) = 155",
                confidence=1.0,
                localities=[],
                parties=["rhdp"]
            )],
            sql_query="SELECT COUNT(*) FROM candidats WHERE parti='RHDP' AND elu=1"
        ),
        DQResult(
            dq_id="DQ2",
            route="rag",
            question="Quelles sont les performances du RHDP à Abidjan ?",
            answer="Le RHDP a dominé à Abidjan avec des scores élevés dans plusieurs communes, notamment Yopougon et Abobo.",
            success=True,
            sources=[Citation(
                source_type="rag",
                table_name="agregation",
                row_id=42,
                excerpt="Résultats à Yopougon: RHDP 68%...",
                confidence=0.85,
                localities=["abidjan", "yopougon"],
                parties=["rhdp"]
            )],
        ),
    ]

    print("=" * 60)
    print("TEST DE L'AGRÉGATEUR ToC")
    print("=" * 60)

    response = aggregator.aggregate(
        "Résultats du RHDP",
        test_results
    )

    print(aggregator.format_response(response))
