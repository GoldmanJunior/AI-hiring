"""
Schémas Pydantic pour l'API FastAPI.
Définition des modèles de requête et réponse.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ===== ENUMS =====

class RouteMethod(str, Enum):
    """Méthode de routage."""
    SQL = "sql"
    RAG = "rag"
    TOC = "toc"


# ===== REQUEST MODELS =====

class QueryRequest(BaseModel):
    """Requête de question via ToC (Tree of Clarifications)."""
    question: str = Field(..., min_length=1, max_length=1000, description="Question en langage naturel")
    explain: bool = Field(default=False, description="Inclure les détails de debug")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Nombre de résultats RAG")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Quels sont les résultats du RHDP à Abidjan ?",
                "explain": False,
                "top_k": 5
            }
        }


# ===== RESPONSE MODELS =====

class SourceModel(BaseModel):
    """Source d'une réponse."""
    source_type: str
    table_name: str
    row_id: Optional[int] = None
    excerpt: str
    confidence: float
    localities: list[str] = []
    parties: list[str] = []


class InterpretationModel(BaseModel):
    """Interprétation d'une question (DQ)."""
    dq_id: str
    route: str
    question: str
    answer: str
    success: bool
    sql_query: Optional[str] = None
    error: Optional[str] = None


class QueryResponse(BaseModel):
    """Réponse à une requête."""
    success: bool
    original_query: str
    answer: str
    method: str
    confidence: float
    interpretations: list[InterpretationModel] = []
    sql_facts: list[str] = []
    rag_insights: list[str] = []
    sources: list[dict] = []
    metadata: dict = {}

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "original_query": "Résultats du RHDP",
                "answer": "Le RHDP a obtenu 155 élus avec un total de 1,642,831 voix.",
                "method": "toc",
                "confidence": 0.85,
                "interpretations": [],
                "sql_facts": ["155 candidats RHDP élus"],
                "rag_insights": ["Forte performance dans le nord"],
                "sources": [],
                "metadata": {}
            }
        }


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    version: str
    components: dict


class SchemaResponse(BaseModel):
    """Réponse du schéma de base."""
    tables: list[str]
    schema_text: str


class StatsResponse(BaseModel):
    """Statistiques du système."""
    database: dict
    rag: dict
    toc: dict


class IntentResponse(BaseModel):
    """Réponse de classification d'intent."""
    question: str
    intent: str
    score_sql: float
    score_rag: float
    confidence: float
    matched_keywords: dict


class ErrorResponse(BaseModel):
    """Réponse d'erreur."""
    error: str
    detail: Optional[str] = None
    code: str = "UNKNOWN_ERROR"
