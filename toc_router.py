"""
ToC Router - Routeur principal Tree of Clarifications.
Orchestre la génération, le routage, l'exécution et l'agrégation.
"""

import os
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import yaml

# Imports des composants ToC
from clarification_generator import ClarificationGenerator, ClarificationTree
from toc_pruner import ToCPruner, PrunedTree, PruningResult
from toc_aggregator import ToCAggregator, DQResult, AggregatedResponse

# Imports des systèmes existants
from sql_analyzer import SQLAnalyzer
from rag_retriever import RAGRetriever
from citation_manager import CitationManager, Citation

logger = logging.getLogger(__name__)
load_dotenv()


class ToCRouter:
    """
    Routeur Tree of Clarifications.

    Pipeline:
    1. Génération de DQs (clarifications)
    2. Élagage et validation
    3. Routage vers SQL ou RAG
    4. Exécution parallèle
    5. Agrégation des résultats
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le routeur ToC.

        Args:
            config_path: Chemin vers la configuration
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path

        # Configuration ToC
        toc_config = self.config.get("toc", {})
        self.parallel_execution = toc_config.get("parallel_execution", True)
        self.max_workers = toc_config.get("max_workers", 3)
        self.fallback_to_simple = toc_config.get("fallback_to_simple", True)

        # Initialiser les composants ToC
        logger.info("Initialisation des composants ToC...")

        self.clarification_generator = ClarificationGenerator(config_path)
        self.pruner = ToCPruner(config_path)
        self.aggregator = ToCAggregator(config_path)

        # Initialiser les systèmes d'exécution
        db_path = self.config.get("database", {}).get("path", "etl/elections.db")
        self.sql_analyzer = SQLAnalyzer(db_path)
        self.citation_manager = CitationManager()

        # RAG (optionnel)
        try:
            self.rag_retriever = RAGRetriever(config_path)
            self.rag_available = True
        except FileNotFoundError:
            logger.warning("RAG non disponible")
            self.rag_retriever = None
            self.rag_available = False

        # Client Groq pour génération RAG
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=api_key) if api_key else None

        logger.info(f"ToCRouter initialisé (RAG: {'activé' if self.rag_available else 'désactivé'})")

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _execute_sql(self, pruning_result: PruningResult) -> DQResult:
        """
        Exécute une DQ via SQL.

        Args:
            pruning_result: DQ validée avec route SQL

        Returns:
            DQResult avec la réponse SQL
        """
        dq = pruning_result.dq
        logger.info(f"Exécution SQL: {dq.dq_id}")

        try:
            result = self.sql_analyzer.query(dq.explicit_question)

            if result.success:
                # Créer les citations
                sources = []
                for row in result.data[:10]:
                    sources.append(Citation(
                        source_type="sql",
                        table_name=self._extract_table_from_sql(result.sql_query),
                        row_id=row.get("id"),
                        excerpt=str(row)[:150],
                        confidence=1.0,
                        localities=[],
                        parties=[]
                    ))

                # Formater la réponse
                answer = self._format_sql_answer(result.data, result.columns)

                return DQResult(
                    dq_id=dq.dq_id,
                    route="sql",
                    question=dq.explicit_question,
                    answer=answer,
                    success=True,
                    sources=sources,
                    sql_query=result.sql_query
                )
            else:
                return DQResult(
                    dq_id=dq.dq_id,
                    route="sql",
                    question=dq.explicit_question,
                    answer="",
                    success=False,
                    sources=[],
                    sql_query=result.sql_query,
                    error=result.error
                )

        except Exception as e:
            logger.error(f"Erreur SQL {dq.dq_id}: {e}")
            return DQResult(
                dq_id=dq.dq_id,
                route="sql",
                question=dq.explicit_question,
                answer="",
                success=False,
                sources=[],
                error=str(e)
            )

    def _execute_rag(self, pruning_result: PruningResult) -> DQResult:
        """
        Exécute une DQ via RAG.

        Args:
            pruning_result: DQ validée avec route RAG

        Returns:
            DQResult avec la réponse RAG
        """
        dq = pruning_result.dq
        logger.info(f"Exécution RAG: {dq.dq_id}")

        if not self.rag_available:
            return DQResult(
                dq_id=dq.dq_id,
                route="rag",
                question=dq.explicit_question,
                answer="",
                success=False,
                sources=[],
                error="RAG non disponible"
            )

        try:
            # Recherche vectorielle
            search_results = self.rag_retriever.search_with_context(dq.explicit_question)

            if search_results["num_results"] == 0:
                return DQResult(
                    dq_id=dq.dq_id,
                    route="rag",
                    question=dq.explicit_question,
                    answer="",
                    success=False,
                    sources=[],
                    error="Aucun résultat RAG"
                )

            # Générer la réponse avec LLM
            answer = self._generate_rag_answer(dq.explicit_question, search_results["context"])

            # Créer les citations
            sources = []
            for result in search_results["results"]:
                sources.append(Citation(
                    source_type="rag",
                    table_name=result.get("table_name", "unknown"),
                    row_id=result.get("row_id"),
                    excerpt=result.get("text", "")[:150],
                    confidence=result.get("score", 0.5),
                    localities=result.get("localities", []),
                    parties=result.get("parties", [])
                ))

            return DQResult(
                dq_id=dq.dq_id,
                route="rag",
                question=dq.explicit_question,
                answer=answer,
                success=True,
                sources=sources
            )

        except Exception as e:
            logger.error(f"Erreur RAG {dq.dq_id}: {e}")
            return DQResult(
                dq_id=dq.dq_id,
                route="rag",
                question=dq.explicit_question,
                answer="",
                success=False,
                sources=[],
                error=str(e)
            )

    def _generate_rag_answer(self, question: str, context: str) -> str:
        """Génère une réponse RAG avec le LLM."""
        if not self.groq_client:
            return f"Contexte trouvé mais génération LLM non disponible.\n\n{context[:500]}"

        prompt = f"""Tu es un assistant expert en données électorales.

CONTEXTE:
{context}

QUESTION: {question}

Réponds de manière concise et précise en utilisant uniquement les informations du contexte."""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.get("groq", {}).get("model", "llama-3.3-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Erreur génération RAG: {e}")
            return f"Erreur de génération. Contexte:\n{context[:300]}"

    def _format_sql_answer(self, data: list[dict], columns: list[str]) -> str:
        """Formate les résultats SQL en réponse lisible."""
        if not data:
            return "Aucun résultat trouvé."

        if len(data) == 1 and len(columns) == 1:
            # Résultat agrégé simple (COUNT, SUM, etc.)
            value = data[0].get(columns[0])
            if isinstance(value, (int, float)):
                return f"{columns[0]}: {value:,}" if isinstance(value, int) else f"{columns[0]}: {value:.2f}"
            return f"{columns[0]}: {value}"

        # Plusieurs résultats - format liste
        lines = []
        for i, row in enumerate(data[:20], 1):
            parts = []
            for col in columns[:5]:
                value = row.get(col)
                if value is not None:
                    if isinstance(value, float):
                        parts.append(f"{col}: {value:.2f}")
                    elif isinstance(value, int) and value > 1000:
                        parts.append(f"{col}: {value:,}")
                    else:
                        parts.append(f"{col}: {value}")
            lines.append(f"{i}. " + ", ".join(parts))

        result = "\n".join(lines)
        if len(data) > 20:
            result += f"\n... et {len(data) - 20} autres résultats"

        return result

    def _extract_table_from_sql(self, sql: str) -> str:
        """Extrait le nom de la table d'une requête SQL."""
        import re
        match = re.search(r'\bFROM\s+(\w+)', sql or "", re.IGNORECASE)
        return match.group(1) if match else "unknown"

    def _execute_dq(self, pruning_result: PruningResult) -> DQResult:
        """
        Exécute une DQ selon sa route.

        Args:
            pruning_result: DQ validée

        Returns:
            DQResult
        """
        if pruning_result.route == "sql":
            return self._execute_sql(pruning_result)
        else:
            return self._execute_rag(pruning_result)

    def query(
        self,
        question: str,
        explain: bool = False
    ) -> AggregatedResponse:
        """
        Traite une question avec le pipeline ToC complet.

        Args:
            question: Question utilisateur
            explain: Si True, inclut des détails de debug

        Returns:
            AggregatedResponse avec la réponse finale
        """
        logger.info(f"ToC Query: '{question[:50]}...'")

        # STEP 1: Génération des clarifications
        logger.info("STEP 1: Génération des DQs...")
        clarification_tree = self.clarification_generator.generate(question)

        if explain:
            logger.info(f"  DQs générées: {len(clarification_tree.disambiguated_questions)}")
            for dq in clarification_tree.disambiguated_questions:
                logger.info(f"    [{dq.dq_id}] {dq.interpretation}")

        # STEP 2: Élagage et routage
        logger.info("STEP 2: Élagage et routage...")
        pruned_tree = self.pruner.prune(clarification_tree)

        if explain:
            logger.info(f"  Valides: {len(pruned_tree.valid_dqs)}, Élaguées: {len(pruned_tree.pruned_dqs)}")

        # Vérifier qu'on a des DQs valides
        if not pruned_tree.valid_dqs:
            logger.warning("Aucune DQ valide, fallback vers requête simple")
            if self.fallback_to_simple:
                return self._fallback_simple(question)
            else:
                return AggregatedResponse(
                    original_query=question,
                    final_answer="Impossible de répondre à cette question avec les données disponibles.",
                    interpretations=[],
                    sql_facts=[],
                    rag_insights=[],
                    sources=[],
                    confidence=0.0,
                    method="toc",
                    metadata={"error": "Aucune interprétation valide"}
                )

        # STEP 3 & 4: Exécution (parallèle ou séquentielle)
        logger.info(f"STEP 3-4: Exécution de {len(pruned_tree.valid_dqs)} DQs...")
        results = []

        if self.parallel_execution and len(pruned_tree.valid_dqs) > 1:
            # Exécution parallèle
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._execute_dq, pr): pr
                    for pr in pruned_tree.valid_dqs
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        pr = futures[future]
                        logger.error(f"Erreur exécution {pr.dq.dq_id}: {e}")
                        results.append(DQResult(
                            dq_id=pr.dq.dq_id,
                            route=pr.route,
                            question=pr.dq.explicit_question,
                            answer="",
                            success=False,
                            sources=[],
                            error=str(e)
                        ))
        else:
            # Exécution séquentielle
            for pr in pruned_tree.valid_dqs:
                result = self._execute_dq(pr)
                results.append(result)

        # STEP 5 & 6: Agrégation
        logger.info("STEP 5-6: Agrégation...")
        response = self.aggregator.aggregate(question, results)

        # Ajouter les métadonnées de debug si demandé
        if explain:
            response.metadata.update({
                "dqs_generated": len(clarification_tree.disambiguated_questions),
                "dqs_pruned": len(pruned_tree.pruned_dqs),
                "dqs_executed": len(results),
                "pruning_reasons": [
                    {"dq_id": pr.dq.dq_id, "reason": pr.pruning_reason}
                    for pr in pruned_tree.pruned_dqs
                ]
            })

        logger.info(f"ToC terminé: confiance={response.confidence:.2f}")
        return response

    def _fallback_simple(self, question: str) -> AggregatedResponse:
        """
        Fallback vers exécution simple (sans ToC).

        Args:
            question: Question originale

        Returns:
            AggregatedResponse
        """
        logger.info("Fallback vers exécution simple...")

        # Essayer SQL d'abord
        sql_result = self.sql_analyzer.query(question)

        if sql_result.success:
            answer = self._format_sql_answer(sql_result.data, sql_result.columns)
            return AggregatedResponse(
                original_query=question,
                final_answer=answer,
                interpretations=[DQResult(
                    dq_id="FALLBACK_SQL",
                    route="sql",
                    question=question,
                    answer=answer,
                    success=True,
                    sources=[],
                    sql_query=sql_result.sql_query
                )],
                sql_facts=[answer[:200]],
                rag_insights=[],
                sources=[],
                confidence=0.8,
                method="simple_sql",
                metadata={"fallback": True}
            )

        # Essayer RAG
        if self.rag_available:
            try:
                search_results = self.rag_retriever.search_with_context(question)
                if search_results["num_results"] > 0:
                    answer = self._generate_rag_answer(question, search_results["context"])
                    return AggregatedResponse(
                        original_query=question,
                        final_answer=answer,
                        interpretations=[DQResult(
                            dq_id="FALLBACK_RAG",
                            route="rag",
                            question=question,
                            answer=answer,
                            success=True,
                            sources=[]
                        )],
                        sql_facts=[],
                        rag_insights=[answer[:200]],
                        sources=[],
                        confidence=0.6,
                        method="simple_rag",
                        metadata={"fallback": True}
                    )
            except Exception as e:
                logger.error(f"Erreur RAG fallback: {e}")

        # Échec total
        return AggregatedResponse(
            original_query=question,
            final_answer="Impossible de répondre à cette question.",
            interpretations=[],
            sql_facts=[],
            rag_insights=[],
            sources=[],
            confidence=0.0,
            method="failed",
            metadata={"fallback": True, "error": "SQL et RAG ont échoué"}
        )

    def close(self):
        """Ferme les connexions."""
        self.sql_analyzer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Mode interactif
def interactive_mode():
    """Lance le mode interactif ToC."""
    print("=" * 60)
    print("TREE OF CLARIFICATIONS ROUTER")
    print("=" * 60)
    print("Commandes:")
    print("  'quit' ou 'exit' - Quitter")
    print("  'explain:' prefix - Mode debug détaillé")
    print("=" * 60)

    with ToCRouter() as router:
        while True:
            try:
                question = input("\nQuestion: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Au revoir!")
                    break

                # Mode explain
                explain = False
                if question.lower().startswith('explain:'):
                    explain = True
                    question = question[8:].strip()

                # Exécuter
                response = router.query(question, explain=explain)

                # Afficher
                print("\n" + router.aggregator.format_response(response))

                if explain:
                    print(f"\n--- Métadonnées ---")
                    import json
                    print(json.dumps(response.metadata, indent=2, ensure_ascii=False))

            except KeyboardInterrupt:
                print("\nInterruption. Au revoir!")
                break
            except Exception as e:
                print(f"Erreur: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    interactive_mode()
