"""
Gestionnaire de citations et provenance des données.
Formate les sources pour traçabilité et confiance.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional, Literal

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Une citation/source pour une réponse."""
    source_type: str  # "sql" ou "rag"
    table_name: str
    row_id: Optional[int]
    excerpt: str
    confidence: float
    localities: list[str]
    parties: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CitedResponse:
    """Réponse avec citations."""
    answer: str
    method: Literal["sql", "rag", "hybrid"]
    success: bool
    sources: list[Citation]
    query: str
    sql_query: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "method": self.method,
            "success": self.success,
            "sources": [s.to_dict() for s in self.sources],
            "query": self.query,
            "sql_query": self.sql_query,
            "error": self.error
        }


class CitationManager:
    """
    Gestionnaire de citations pour tracer la provenance des données.
    """

    def __init__(self):
        """Initialise le gestionnaire."""
        logger.info("CitationManager initialisé")

    def create_sql_citation(
        self,
        table_name: str,
        row_data: dict,
        confidence: float = 1.0
    ) -> Citation:
        """
        Crée une citation pour un résultat SQL.

        Args:
            table_name: Nom de la table source
            row_data: Données de la ligne
            confidence: Score de confiance (1.0 pour SQL)

        Returns:
            Citation formatée
        """
        # Extraire les entités si présentes
        localities = []
        parties = []

        if "region" in row_data and row_data["region"]:
            localities.append(str(row_data["region"]))
        if "nom" in row_data and row_data["nom"]:
            # Pourrait être une localité (circonscription) ou un candidat
            pass
        if "parti" in row_data and row_data["parti"]:
            parties.append(str(row_data["parti"]))

        # Créer un extrait lisible
        excerpt_parts = []
        for key, value in list(row_data.items())[:5]:  # Limiter à 5 champs
            if value is not None:
                excerpt_parts.append(f"{key}: {value}")
        excerpt = ", ".join(excerpt_parts)

        return Citation(
            source_type="sql",
            table_name=table_name,
            row_id=row_data.get("id"),
            excerpt=excerpt[:200],  # Limiter la longueur
            confidence=confidence,
            localities=localities,
            parties=parties
        )

    def create_rag_citation(
        self,
        retrieval_result: dict
    ) -> Citation:
        """
        Crée une citation pour un résultat RAG.

        Args:
            retrieval_result: Résultat du retriever RAG

        Returns:
            Citation formatée
        """
        # Convertir le score numpy en float Python natif
        score = retrieval_result.get("score", 0.0)
        if hasattr(score, 'item'):
            score = score.item()
        else:
            score = float(score)

        return Citation(
            source_type="rag",
            table_name=retrieval_result.get("table_name", "unknown"),
            row_id=retrieval_result.get("row_id"),
            excerpt=retrieval_result.get("text", "")[:200],
            confidence=score,
            localities=retrieval_result.get("localities", []),
            parties=retrieval_result.get("parties", [])
        )

    def create_sql_response(
        self,
        query: str,
        sql_query: str,
        data: list[dict],
        columns: list[str],
        success: bool = True,
        error: Optional[str] = None
    ) -> CitedResponse:
        """
        Crée une réponse citée pour un résultat SQL.

        Args:
            query: Question originale
            sql_query: Requête SQL exécutée
            data: Données retournées
            columns: Noms des colonnes
            success: Succès de la requête
            error: Message d'erreur si échec

        Returns:
            CitedResponse avec sources
        """
        if not success:
            return CitedResponse(
                answer=f"Erreur lors de l'exécution: {error}",
                method="sql",
                success=False,
                sources=[],
                query=query,
                sql_query=sql_query,
                error=error
            )

        # Créer les citations pour chaque ligne (limité)
        sources = []
        table_name = self._extract_table_from_sql(sql_query)

        for row in data[:10]:  # Limiter à 10 sources
            citation = self.create_sql_citation(table_name, row)
            sources.append(citation)

        # Formater la réponse
        if not data:
            answer = "Aucun résultat trouvé pour cette requête."
        elif len(data) == 1:
            # Résultat unique - formater de manière lisible
            answer = self._format_single_result(data[0], columns)
        else:
            # Plusieurs résultats - format tabulaire
            answer = self._format_multiple_results(data, columns)

        return CitedResponse(
            answer=answer,
            method="sql",
            success=True,
            sources=sources,
            query=query,
            sql_query=sql_query
        )

    def create_rag_response(
        self,
        query: str,
        answer: str,
        retrieval_results: list[dict],
        success: bool = True,
        error: Optional[str] = None
    ) -> CitedResponse:
        """
        Crée une réponse citée pour un résultat RAG.

        Args:
            query: Question originale
            answer: Réponse générée par le LLM
            retrieval_results: Résultats du retriever
            success: Succès de la recherche
            error: Message d'erreur si échec

        Returns:
            CitedResponse avec sources
        """
        if not success:
            return CitedResponse(
                answer=f"Erreur lors de la recherche: {error}",
                method="rag",
                success=False,
                sources=[],
                query=query,
                error=error
            )

        # Créer les citations
        sources = []
        for result in retrieval_results:
            citation = self.create_rag_citation(result)
            sources.append(citation)

        return CitedResponse(
            answer=answer,
            method="rag",
            success=True,
            sources=sources,
            query=query
        )

    def _extract_table_from_sql(self, sql: str) -> str:
        """Extrait le nom de la table principale d'une requête SQL."""
        import re

        # Pattern pour FROM table_name
        match = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
        if match:
            return match.group(1)
        return "unknown"

    def _format_single_result(self, row: dict, columns: list[str]) -> str:
        """Formate un résultat unique de manière lisible."""
        parts = []
        for col in columns:
            value = row.get(col)
            if value is not None:
                # Formater les nombres
                if isinstance(value, float):
                    value = f"{value:.2f}"
                elif isinstance(value, int) and value > 1000:
                    value = f"{value:,}"
                parts.append(f"{col}: {value}")

        return "\n".join(parts)

    def _format_multiple_results(
        self,
        data: list[dict],
        columns: list[str],
        max_rows: int = 20
    ) -> str:
        """Formate plusieurs résultats en tableau."""
        lines = []

        # En-tête
        header = " | ".join(columns[:6])  # Limiter les colonnes
        lines.append(header)
        lines.append("-" * len(header))

        # Données
        for row in data[:max_rows]:
            values = []
            for col in columns[:6]:
                value = row.get(col, "")
                if value is None:
                    value = ""
                elif isinstance(value, float):
                    value = f"{value:.2f}"
                elif isinstance(value, int) and value > 1000:
                    value = f"{value:,}"
                else:
                    value = str(value)[:30]  # Tronquer
                values.append(value)
            lines.append(" | ".join(values))

        if len(data) > max_rows:
            lines.append(f"... et {len(data) - max_rows} autres résultats")

        return "\n".join(lines)

    def format_sources_text(self, sources: list[Citation]) -> str:
        """
        Formate les sources en texte lisible.

        Args:
            sources: Liste de citations

        Returns:
            Texte formaté des sources
        """
        if not sources:
            return "Aucune source disponible."

        lines = ["Sources:"]
        for i, source in enumerate(sources, 1):
            conf_pct = source.confidence * 100
            lines.append(
                f"  [{i}] {source.table_name} (confiance: {conf_pct:.0f}%)"
            )
            if source.localities:
                lines.append(f"      Localités: {', '.join(source.localities)}")
            if source.parties:
                lines.append(f"      Partis: {', '.join(source.parties)}")
            if source.excerpt:
                lines.append(f"      Extrait: {source.excerpt[:100]}...")

        return "\n".join(lines)


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = CitationManager()

    # Test SQL citation
    print("=== Test Citation SQL ===")
    sql_data = {
        "id": 1,
        "nom": "KOFFI AKA",
        "parti": "RHDP",
        "score": 12500,
        "pourcentage": 65.3,
        "region": "PORO"
    }
    citation = manager.create_sql_citation("candidats", sql_data)
    print(citation.to_dict())

    # Test SQL response
    print("\n=== Test Réponse SQL ===")
    response = manager.create_sql_response(
        query="Combien de candidats RHDP ?",
        sql_query="SELECT COUNT(*) as count FROM candidats WHERE parti = 'RHDP'",
        data=[{"count": 155}],
        columns=["count"]
    )
    print(response.to_dict())

    # Test RAG citation
    print("\n=== Test Citation RAG ===")
    rag_result = {
        "table_name": "agregation",
        "row_id": 42,
        "text": "Résultats à Korhogo: RHDP 78%, PDCI 15%...",
        "score": 0.89,
        "localities": ["korhogo"],
        "parties": ["rhdp", "pdci-rda"]
    }
    citation = manager.create_rag_citation(rag_result)
    print(citation.to_dict())
