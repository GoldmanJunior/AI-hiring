"""
API FastAPI pour le système hybride SQL + RAG + ToC.
Expose les endpoints pour interroger la base de données électorale.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Imports des modules du système
from schemas import (
    QueryRequest, QueryResponse,
    HealthResponse, SchemaResponse, StatsResponse,
    IntentResponse, ErrorResponse
)
from toc_router import ToCRouter
from intent_classifier import IntentClassifier
from sql_analyzer import SQLAnalyzer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== GLOBAL STATE =====

class AppState:
    """État global de l'application."""
    toc_router: Optional[ToCRouter] = None
    intent_classifier: Optional[IntentClassifier] = None
    sql_analyzer: Optional[SQLAnalyzer] = None


state = AppState()


# ===== LIFESPAN =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    logger.info("Initialisation de l'API...")

    try:
        # Initialiser les composants
        state.toc_router = ToCRouter()
        state.intent_classifier = IntentClassifier()
        state.sql_analyzer = SQLAnalyzer("etl/elections.db")

        logger.info("API initialisée avec succès")
    except Exception as e:
        logger.error(f"Erreur d'initialisation: {e}")
        raise

    yield

    # Shutdown
    logger.info("Arrêt de l'API...")
    if state.toc_router:
        state.toc_router.close()
    if state.sql_analyzer:
        state.sql_analyzer.close()


# ===== APPLICATION =====

app = FastAPI(
    title="Elections API",
    description="""
API d'interrogation des données électorales via le système ToC (Tree of Clarifications).

## Fonctionnalités

- **ToC (Tree of Clarifications)**: Génère plusieurs interprétations d'une question ambiguë
- **Routage intelligent**: Chaque interprétation est routée vers SQL ou RAG automatiquement
- **SQL**: Requêtes analytiques sur la base de données
- **RAG**: Recherche sémantique dans les documents indexés
- **Agrégation**: Synthèse des résultats en une réponse cohérente

## Sécurité

Seules les requêtes SELECT sont autorisées. Aucune modification de données n'est possible.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== EXCEPTION HANDLERS =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "code": f"HTTP_{exc.status_code}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Erreur non gérée: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Erreur interne du serveur", "detail": str(exc), "code": "INTERNAL_ERROR"}
    )


# ===== ENDPOINTS =====

@app.get("/", tags=["General"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "Elections API - Système hybride SQL + RAG + ToC",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Vérifie l'état de santé de l'API."""
    components = {
        "toc_router": state.toc_router is not None,
        "sql_analyzer": state.sql_analyzer is not None,
        "rag_available": state.toc_router.rag_available if state.toc_router else False
    }

    all_ok = all(components.values())

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        version="1.0.0",
        components=components
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_toc(request: QueryRequest):
    """
    Interroge le système avec le pipeline ToC (Tree of Clarifications).

    Le système:
    1. Génère plusieurs interprétations de la question
    2. Route chaque interprétation vers SQL ou RAG
    3. Exécute les requêtes en parallèle
    4. Agrège les résultats en une réponse cohérente
    """
    if not state.toc_router:
        raise HTTPException(status_code=503, detail="Service non disponible")

    try:
        logger.info(f"Query ToC: {request.question[:50]}...")

        response = state.toc_router.query(request.question, explain=request.explain)

        return QueryResponse(
            success=response.confidence > 0,
            original_query=response.original_query,
            answer=response.final_answer,
            method=response.method,
            confidence=response.confidence,
            interpretations=[
                {
                    "dq_id": i.dq_id,
                    "route": i.route,
                    "question": i.question,
                    "answer": i.answer,
                    "success": i.success,
                    "sql_query": i.sql_query,
                    "error": i.error
                }
                for i in response.interpretations
            ],
            sql_facts=response.sql_facts,
            rag_insights=response.rag_insights,
            sources=response.sources,
            metadata=response.metadata
        )

    except Exception as e:
        logger.error(f"Erreur query ToC: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/intent", response_model=IntentResponse, tags=["Analysis"])
async def classify_intent(
    question: str = Query(..., min_length=1, max_length=1000)
):
    """
    Classifie l'intent d'une question (SQL ou RAG).

    Utilise uniquement des mots-clés, sans appel LLM.
    """
    if not state.intent_classifier:
        raise HTTPException(status_code=503, detail="Service non disponible")

    result = state.intent_classifier.classify(question)

    return IntentResponse(
        question=question,
        intent=result.intent,
        score_sql=result.score_sql,
        score_rag=result.score_rag,
        confidence=result.confidence,
        matched_keywords=result.matched_keywords
    )


@app.get("/schema", response_model=SchemaResponse, tags=["Database"])
async def get_schema():
    """Retourne le schéma de la base de données."""
    if not state.sql_analyzer:
        raise HTTPException(status_code=503, detail="Service non disponible")

    tables = state.sql_analyzer.get_tables()
    schema_text = state.sql_analyzer.get_schema()

    return SchemaResponse(
        tables=tables,
        schema_text=schema_text
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Retourne les statistiques du système."""
    stats = {
        "database": {},
        "rag": {},
        "toc": {}
    }

    if state.sql_analyzer:
        stats["database"] = {
            "tables": state.sql_analyzer.get_tables(),
            "connected": True
        }

    if state.toc_router:
        stats["rag"] = {
            "available": state.toc_router.rag_available,
            "index_size": state.toc_router.rag_retriever.index.ntotal if state.toc_router.rag_available else 0
        }
        stats["toc"] = {
            "parallel_execution": state.toc_router.parallel_execution,
            "max_workers": state.toc_router.max_workers
        }

    return StatsResponse(**stats)


@app.get("/tables/{table_name}/sample", tags=["Database"])
async def get_table_sample(
    table_name: str,
    limit: int = Query(default=10, ge=1, le=100)
):
    """Retourne un échantillon de données d'une table."""
    if not state.sql_analyzer:
        raise HTTPException(status_code=503, detail="Service non disponible")

    tables = state.sql_analyzer.get_tables()
    if table_name not in tables:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' non trouvée")

    result = state.sql_analyzer.query(f"Affiche {limit} lignes de la table {table_name}")

    if result.success:
        return {
            "table": table_name,
            "columns": result.columns,
            "data": result.data,
            "count": len(result.data)
        }
    else:
        raise HTTPException(status_code=500, detail=result.error)


# ===== MAIN =====

def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="API Elections - SQL + RAG + ToC")
    parser.add_argument("--host", default="0.0.0.0", help="Host (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (défaut: 8000)")
    parser.add_argument("--reload", action="store_true", help="Mode développement avec reload")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de workers")

    args = parser.parse_args()

    print("=" * 60)
    print("ELECTIONS API - SQL + RAG + ToC")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
