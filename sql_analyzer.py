"""
Module SQL Analyzer - Interrogation de base SQLite via LLM Groq.
Sécurisé : SEULES les requêtes SELECT sont autorisées.
"""

import sqlite3
import re
import logging
import os
from typing import Optional, Any
from dataclasses import dataclass

from groq import Groq
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLSecurityError(Exception):
    """Exception levée quand une requête SQL non autorisée est détectée."""
    pass


class SQLExecutionError(Exception):
    """Exception levée lors d'une erreur d'exécution SQL."""
    pass


class LLMError(Exception):
    """Exception levée lors d'une erreur avec l'API Groq."""
    pass


@dataclass
class QueryResult:
    """Résultat d'une requête SQL."""
    success: bool
    data: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    sql_query: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convertit le résultat en dictionnaire."""
        return {
            "success": self.success,
            "data": self.data,
            "columns": self.columns,
            "row_count": self.row_count,
            "sql_query": self.sql_query,
            "error": self.error
        }


class SQLValidator:
    """Validateur de sécurité pour les requêtes SQL."""

    # Mots-clés SQL interdits (modification de données)
    FORBIDDEN_KEYWORDS = [
        r'\bUPDATE\b',
        r'\bDELETE\b',
        r'\bINSERT\b',
        r'\bDROP\b',
        r'\bALTER\b',
        r'\bCREATE\b',
        r'\bTRUNCATE\b',
        r'\bREPLACE\b',
        r'\bMERGE\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bATTACH\b',
        r'\bDETACH\b',
        r'\bPRAGMA\b',
        r'\bVACUUM\b',
        r'\bREINDEX\b',
    ]

    # Pattern pour détecter les commentaires SQL (potentielle injection)
    COMMENT_PATTERNS = [
        r'--',
        r'/\*',
        r'\*/',
    ]

    @classmethod
    def validate(cls, sql: str) -> tuple[bool, str]:
        """
        Valide qu'une requête SQL est sécurisée (SELECT uniquement).

        Args:
            sql: La requête SQL à valider

        Returns:
            Tuple (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Requête SQL vide"

        # Normaliser la requête
        sql_normalized = sql.strip().upper()

        # Vérifier que la requête commence par SELECT ou WITH (CTE)
        if not (sql_normalized.startswith('SELECT') or sql_normalized.startswith('WITH')):
            return False, "Seules les requêtes SELECT sont autorisées"

        # Vérifier l'absence de mots-clés interdits
        for pattern in cls.FORBIDDEN_KEYWORDS:
            if re.search(pattern, sql_normalized, re.IGNORECASE):
                keyword = pattern.replace(r'\b', '').upper()
                return False, f"Mot-clé interdit détecté: {keyword}"

        # Vérifier l'absence de commentaires SQL (injection potentielle)
        for pattern in cls.COMMENT_PATTERNS:
            if pattern in sql:
                return False, f"Commentaires SQL non autorisés: {pattern}"

        # Vérifier l'absence de points-virgules multiples (injection)
        if sql.count(';') > 1:
            return False, "Requêtes multiples non autorisées"

        # Vérifier l'absence de UNION avec sous-requêtes dangereuses
        if 'UNION' in sql_normalized:
            # Autoriser UNION mais vérifier qu'il n'y a pas de requêtes dangereuses après
            parts = re.split(r'\bUNION\b', sql_normalized, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if part and not (part.startswith('SELECT') or part.startswith('ALL SELECT') or part.startswith('(')):
                    # Vérifier si c'est juste "ALL" suivi de SELECT
                    if not re.match(r'^ALL\s+SELECT', part):
                        return False, "UNION avec requête non-SELECT détecté"

        return True, ""

    @classmethod
    def sanitize(cls, sql: str) -> str:
        """
        Nettoie une requête SQL.

        Args:
            sql: La requête SQL à nettoyer

        Returns:
            La requête nettoyée
        """
        # Supprimer les espaces multiples
        sql = re.sub(r'\s+', ' ', sql.strip())

        # Supprimer le point-virgule final si présent
        sql = sql.rstrip(';')

        return sql


class SchemaExtractor:
    """Extracteur de schéma de base de données SQLite."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def get_tables(self) -> list[str]:
        """Récupère la liste des tables."""
        cursor = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> list[dict[str, str]]:
        """Récupère le schéma d'une table."""
        cursor = self.connection.execute(f"PRAGMA table_info('{table_name}')")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "name": row[1],
                "type": row[2],
                "nullable": not row[3],
                "primary_key": bool(row[5])
            })
        return columns

    def get_table_sample(self, table_name: str, limit: int = 3) -> list[dict]:
        """Récupère un échantillon de données d'une table."""
        cursor = self.connection.execute(f"SELECT * FROM '{table_name}' LIMIT {limit}")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def get_full_schema(self, include_samples: bool = True) -> str:
        """
        Génère une description complète du schéma de la base.

        Args:
            include_samples: Inclure des exemples de données

        Returns:
            Description textuelle du schéma
        """
        tables = self.get_tables()
        schema_parts = []

        for table in tables:
            columns = self.get_table_schema(table)

            # Description de la table
            table_desc = f"TABLE: {table}\n"
            table_desc += "COLONNES:\n"

            for col in columns:
                pk = " (PRIMARY KEY)" if col["primary_key"] else ""
                nullable = " (nullable)" if col["nullable"] else " (NOT NULL)"
                table_desc += f"  - {col['name']}: {col['type']}{pk}{nullable}\n"

            # Échantillon de données
            if include_samples:
                try:
                    samples = self.get_table_sample(table)
                    if samples:
                        table_desc += "EXEMPLE DE DONNÉES:\n"
                        for sample in samples:
                            table_desc += f"  {sample}\n"
                except Exception:
                    pass

            schema_parts.append(table_desc)

        return "\n".join(schema_parts)


class SQLAnalyzer:
    """
    Analyseur SQL utilisant Groq LLM pour convertir des questions
    en langage naturel en requêtes SQL sécurisées.
    """

    SYSTEM_PROMPT = """Tu es un expert SQL. Tu convertis des questions initialement en langage naturel en requêtes SQL.

RÈGLES STRICTES:
1. Tu dois UNIQUEMENT générer des requêtes SELECT
2. Tu ne dois JAMAIS générer UPDATE, DELETE, INSERT, DROP, ALTER, CREATE, TRUNCATE
3. Retourne UNIQUEMENT la requête SQL, sans explication, sans markdown, sans ```
4. Si la question ne peut pas être traduite en SELECT, réponds: "IMPOSSIBLE"
5. Utilise le schéma de base de données fourni
6. Génère du SQL compatible SQLite
7. Utilise des alias clairs pour les colonnes calculées
8. Pour les agrégations, utilise GROUP BY approprié
9. Limite les résultats à 100 lignes par défaut sauf si demandé autrement, on ne doit jamais recuperer toutes les lignes ou accepter des reqûetes malicieuses
10. REFUSER les requêtes inappropriées ou dangereuses, EXPLIQUER pourquoi elles sont refusées sans ajouter autres choses
11.Si une requête ne retrouve rien, reponds "nothing found in the dataset",tu dois seulement repondre en te referant à la base de données fournie

SCHÉMA DE LA BASE DE DONNÉES:
{schema}

IMPORTANT: Réponds UNIQUEMENT avec la requête SQL, rien d'autre."""

    def __init__(
        self,
        db_path: str,
        groq_api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile"
    ):
        """
        Initialise l'analyseur SQL.

        Args:
            db_path: Chemin vers la base de données SQLite
            groq_api_key: Clé API Groq (optionnel si dans .env)
            model: Modèle Groq à utiliser
        """
        # Charger les variables d'environnement
        load_dotenv()

        # Initialiser la connexion à la base de données
        self.db_path = db_path
        self.connection = self._connect_db()

        # Initialiser le client Groq
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Clé API Groq requise (paramètre ou variable GROQ_API_KEY)")

        self.groq_client = Groq(api_key=api_key)
        self.model = model

        # Extraire le schéma
        self.schema_extractor = SchemaExtractor(self.connection)
        self.schema = self.schema_extractor.get_full_schema()

        logger.info(f"SQLAnalyzer initialisé avec la base: {db_path}")
        logger.info(f"Modèle Groq: {model}")

    def _connect_db(self) -> sqlite3.Connection:
        """Établit la connexion à la base de données."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Base de données non trouvée: {self.db_path}")

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        logger.info(f"Connexion établie à: {self.db_path}")
        return connection

    def _generate_sql(self, question: str) -> str:
        """
        Utilise Groq LLM pour générer une requête SQL.

        Args:
            question: Question en langage naturel

        Returns:
            Requête SQL générée
        """
        system_prompt = self.SYSTEM_PROMPT.format(schema=self.schema)

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,  # Basse température pour plus de précision
                max_tokens=1024
            )

            sql = response.choices[0].message.content.strip()

            # Nettoyer le SQL (enlever les backticks markdown si présents)
            sql = re.sub(r'^```sql?\s*', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\s*```$', '', sql)
            sql = sql.strip()

            logger.info(f"SQL généré: {sql}")
            return sql

        except Exception as e:
            logger.error(f"Erreur Groq API: {e}")
            raise LLMError(f"Erreur lors de la génération SQL: {e}")

    def _execute_sql(self, sql: str) -> QueryResult:
        """
        Exécute une requête SQL validée.

        Args:
            sql: Requête SQL à exécuter

        Returns:
            QueryResult avec les résultats
        """
        try:
            cursor = self.connection.execute(sql)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            # Convertir en liste de dictionnaires
            data = [dict(zip(columns, row)) for row in rows]

            return QueryResult(
                success=True,
                data=data,
                columns=columns,
                row_count=len(data),
                sql_query=sql
            )

        except sqlite3.Error as e:
            logger.error(f"Erreur SQL: {e}")
            return QueryResult(
                success=False,
                data=[],
                columns=[],
                row_count=0,
                sql_query=sql,
                error=str(e)
            )

    def query(self, question: str) -> QueryResult:
        """
        Interroge la base de données avec une question en langage naturel.

        Args:
            question: Question en langage naturel

        Returns:
            QueryResult avec les résultats ou l'erreur
        """
        logger.info(f"Question reçue: {question}")

        # Étape 1: Générer le SQL via Groq
        try:
            sql = self._generate_sql(question)
        except LLMError as e:
            return QueryResult(
                success=False,
                data=[],
                columns=[],
                row_count=0,
                sql_query="",
                error=str(e)
            )

        # Vérifier si le LLM a indiqué que c'est impossible
        if sql.upper() == "IMPOSSIBLE":
            return QueryResult(
                success=False,
                data=[],
                columns=[],
                row_count=0,
                sql_query="",
                error="Cette question ne peut pas être convertie en requête SELECT"
            )

        # Étape 2: Nettoyer le SQL
        sql = SQLValidator.sanitize(sql)

        # Étape 3: Valider la sécurité
        is_valid, error_message = SQLValidator.validate(sql)
        if not is_valid:
            logger.warning(f"Requête rejetée pour raison de sécurité: {error_message}")
            return QueryResult(
                success=False,
                data=[],
                columns=[],
                row_count=0,
                sql_query=sql,
                error=f"Requête rejetée: {error_message}"
            )

        # Étape 4: Exécuter la requête
        return self._execute_sql(sql)

    def get_schema(self) -> str:
        """Retourne le schéma de la base de données."""
        return self.schema

    def get_tables(self) -> list[str]:
        """Retourne la liste des tables."""
        return self.schema_extractor.get_tables()

    def close(self):
        """Ferme la connexion à la base de données."""
        if self.connection:
            self.connection.close()
            logger.info("Connexion fermée")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Fonction utilitaire pour utilisation rapide
def quick_query(
    db_path: str,
    question: str,
    groq_api_key: Optional[str] = None
) -> dict:
    """
    Fonction utilitaire pour une requête rapide.

    Args:
        db_path: Chemin vers la base de données
        question: Question en langage naturel
        groq_api_key: Clé API Groq (optionnel si dans .env)

    Returns:
        Dictionnaire avec les résultats
    """
    with SQLAnalyzer(db_path, groq_api_key) as analyzer:
        result = analyzer.query(question)
        return result.to_dict()
