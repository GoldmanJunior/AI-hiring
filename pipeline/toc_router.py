import sys
import pathlib
_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import time
import logging
import contextvars
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import yaml
import observability as obs

from pipeline.clarification_generator import ClarificationGenerator, ClarificationTree
from pipeline.toc_pruner import ToCPruner, PrunedTree, PruningResult
from pipeline.toc_aggregator import ToCAggregator, DQResult, AggregatedResponse

from retrievers.sql_analyzer import SQLAnalyzer
from retrievers.rag_retriever import RAGRetriever
from retrievers.citation_manager import CitationManager, Citation

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

        toc_config = self.config.get("toc", {})
        self.parallel_execution = toc_config.get("parallel_execution", True)
        self.max_workers = toc_config.get("max_workers", 3)
        self.fallback_to_simple = toc_config.get("fallback_to_simple", True)

        logger.info("Initialisation des composants ToC...")

        self.clarification_generator = ClarificationGenerator(config_path)
        self.pruner = ToCPruner(config_path)
        self.aggregator = ToCAggregator(config_path)

        db_path = self.config.get("database", {}).get("path", "etl/elections.db")
        self.sql_analyzer = SQLAnalyzer(db_path)
        self.citation_manager = CitationManager()

        try:
            self.rag_retriever = RAGRetriever(config_path)
            self.rag_available = True
        except FileNotFoundError:
            logger.warning("RAG non disponible")
            self.rag_retriever = None
            self.rag_available = False

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

        _span, _t0 = obs.new_span(
            "sql_dq_execution",
            metadata={"dq_id": dq.dq_id, "question": dq.explicit_question[:120]},
        )

        try:
            result = self.sql_analyzer.query(dq.explicit_question)

            if result.success:
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

                answer = self._format_sql_answer(result.data, result.columns)

                obs.end_span(_span, _t0, metadata={
                    "sql_query": result.sql_query,
                    "row_count": result.row_count,
                    "success": True,
                })

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
                obs.end_span(_span, _t0, metadata={
                    "sql_query": result.sql_query,
                    "success": False,
                }, error=result.error)

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
            obs.end_span(_span, _t0, error=str(e))
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

        _span, _t0 = obs.new_span(
            "rag_dq_execution",
            metadata={"dq_id": dq.dq_id, "question": dq.explicit_question[:120]},
        )

        if not self.rag_available:
            obs.end_span(_span, _t0, error="RAG non disponible")
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
            _retrieval_t0 = time.monotonic()
            search_results = self.rag_retriever.search_with_context(dq.explicit_question)
            _retrieval_ms = int((time.monotonic() - _retrieval_t0) * 1000)

            if search_results["num_results"] == 0:
                obs.end_span(_span, _t0, metadata={"num_results": 0}, error="Aucun résultat RAG")
                return DQResult(
                    dq_id=dq.dq_id,
                    route="rag",
                    question=dq.explicit_question,
                    answer="",
                    success=False,
                    sources=[],
                    error="Aucun résultat RAG"
                )

            _scores = [r.get("score", 0.0) for r in search_results["results"]]
            _retrieval_span, _r0 = obs.new_span("rag_retrieval", metadata={
                "dq_id": dq.dq_id,
                "top_k": search_results["num_results"],
                "top_score": round(_scores[0], 4) if _scores else 0,
                "avg_score": round(sum(_scores) / len(_scores), 4) if _scores else 0,
                "doc_ids": [r.get("chunk_id") for r in search_results["results"]],
                "retrieval_latency_ms": _retrieval_ms,
            })
            obs.end_span(_retrieval_span, _r0)

            answer = self._generate_rag_answer(dq.explicit_question, search_results["context"])

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

            obs.end_span(_span, _t0, metadata={"num_results": search_results["num_results"], "success": True})
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
            obs.end_span(_span, _t0, error=str(e))
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

        _model = self.config.get("groq", {}).get("model", "llama-3.3-70b-versatile")
        prompt = f"""Tu es un assistant expert en données électorales.

CONTEXTE:
{context}

QUESTION: {question}

Réponds de manière concise et précise en utilisant uniquement les informations du contexte."""

        _messages = [{"role": "user", "content": prompt}]

        try:
            _t0 = time.monotonic()
            response = self.groq_client.chat.completions.create(
                model=_model,
                messages=_messages,
                temperature=0.3,
                max_tokens=512
            )
            _duration_ms = int((time.monotonic() - _t0) * 1000)
            output = response.choices[0].message.content.strip()

            obs.record_generation(
                name="rag_answer_generation",
                model=_model,
                input_messages=_messages,
                output=output,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                temperature=0.3,
                duration_ms=_duration_ms,
            )

            return output
        except Exception as e:
            logger.error(f"Erreur génération RAG: {e}")
            obs.record_generation(
                name="rag_answer_generation",
                model=_model,
                input_messages=_messages,
                output="",
                error=str(e),
            )
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

        clarification_tree = self.clarification_generator.generate(question)

        if explain:
            logger.info(f"  DQs générées: {len(clarification_tree.disambiguated_questions)}")
            for dq in clarification_tree.disambiguated_questions:
                logger.info(f"    [{dq.dq_id}] {dq.interpretation}")

        _prune_span, _p0 = obs.new_span("pruning_routing", metadata={"dqs_generated": len(clarification_tree.disambiguated_questions)})
        pruned_tree = self.pruner.prune(clarification_tree)
        obs.end_span(_prune_span, _p0, metadata={
            "valid_dqs": len(pruned_tree.valid_dqs),
            "pruned_dqs": len(pruned_tree.pruned_dqs),
            "sql_routes": pruned_tree.pruning_metadata.get("sql_routes", 0),
            "rag_routes": pruned_tree.pruning_metadata.get("rag_routes", 0),
        })

        if explain:
            logger.info(f"  Valides: {len(pruned_tree.valid_dqs)}, Élaguées: {len(pruned_tree.pruned_dqs)}")

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

        results = []

        if self.parallel_execution and len(pruned_tree.valid_dqs) > 1:
            # Each task gets its own copy of the current context so that
            # _active_trace propagates into worker threads independently.
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for pr in pruned_tree.valid_dqs:
                    _ctx = contextvars.copy_context()
                    futures[executor.submit(_ctx.run, self._execute_dq, pr)] = pr
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
            for pr in pruned_tree.valid_dqs:
                result = self._execute_dq(pr)
                results.append(result)

        response = self.aggregator.aggregate(question, results)

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

                explain = False
                if question.lower().startswith('explain:'):
                    explain = True
                    question = question[8:].strip()

                response = router.query(question, explain=explain)

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
