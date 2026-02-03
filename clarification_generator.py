"""
Clarification Generator - Génère des interprétations multiples d'une requête.
Partie du système Tree of Clarifications (ToC).
"""

import os
import logging
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

from groq import Groq
from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class DisambiguatedQuestion:
    """Une question désambiguïsée (DQ)."""
    dq_id: str
    original_query: str
    interpretation: str
    explicit_question: str
    confidence: float
    entities: dict[str, list[str]]  # {"localities": [...], "parties": [...], "metrics": [...]}
    assumptions: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClarificationTree:
    """Arbre de clarifications."""
    original_query: str
    disambiguated_questions: list[DisambiguatedQuestion]
    generation_metadata: dict

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "disambiguated_questions": [dq.to_dict() for dq in self.disambiguated_questions],
            "generation_metadata": self.generation_metadata
        }


class ClarificationGenerator:
    """
    Génère des interprétations multiples (DQs) d'une requête utilisateur.
    Utilise un LLM pour l'expansion mais structure rigoureusement la sortie.
    """

    SYSTEM_PROMPT = """Tu es un expert en analyse de requêtes utilisateur pour un système de base de données électorale.

CONTEXTE:
- Base de données SQLite contenant des résultats électoraux (Côte d'Ivoire)
- Tables disponibles:
  * circonscriptions: id, region, code, nom, nb_bv, inscrits, votants, taux_participation, nuls, exprimes, bulletins_blancs, taux_bulletins_blancs
  * candidats: id, circonscription_id, region, parti, nom, score, pourcentage, elu

TÂCHE:
Génère 2 à 5 interprétations DISTINCTES et PLAUSIBLES de la requête utilisateur.
Chaque interprétation doit être:
- Auto-suffisante (explicite, sans ambiguïté)
- Répondable avec les données disponibles
- Distincte des autres interprétations

RÈGLES STRICTES:
1. Ne génère JAMAIS de questions impossibles à répondre avec les données disponibles
2. Préfère la diversité des interprétations
3. Identifie les entités (localités, partis, métriques) dans chaque interprétation
4. Liste les hypothèses faites pour chaque interprétation
5. Attribue un score de confiance (0.0-1.0) basé sur la probabilité que cette interprétation soit l'intention réelle

FORMAT DE SORTIE (JSON strict):
{
  "disambiguated_questions": [
    {
      "dq_id": "DQ1",
      "interpretation": "Description brève de l'interprétation",
      "explicit_question": "Question reformulée explicitement",
      "confidence": 0.8,
      "entities": {
        "localities": ["abidjan", "bouake"],
        "parties": ["rhdp", "pdci-rda"],
        "metrics": ["score", "pourcentage"]
      },
      "assumptions": ["L'utilisateur veut les résultats du premier tour", "..."]
    }
  ]
}

Réponds UNIQUEMENT avec le JSON, sans texte additionnel."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le générateur de clarifications.

        Args:
            config_path: Chemin vers la configuration
        """
        self.config = self._load_config(config_path)

        # Configuration
        toc_config = self.config.get("toc", {})
        self.min_dqs = toc_config.get("min_disambiguations", 2)
        self.max_dqs = toc_config.get("max_disambiguations", 5)

        groq_config = self.config.get("groq", {})
        self.model = groq_config.get("model", "llama-3.3-70b-versatile")
        self.temperature = toc_config.get("generation_temperature", 0.7)

        # Client Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY non définie")
        self.groq_client = Groq(api_key=api_key)

        logger.info(f"ClarificationGenerator initialisé (DQs: {self.min_dqs}-{self.max_dqs})")

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _extract_basic_entities(self, query: str) -> dict[str, list[str]]:
        """
        Extraction basique d'entités depuis une requête.
        Utilisé comme fallback quand le LLM échoue.
        """
        query_lower = query.lower()

        # Régions/localités connues
        known_localities = [
            "bagoue", "poro", "tchologo", "gbeke", "abidjan", "bouake",
            "yamoussoukro", "korhogo", "daloa", "san-pedro", "gagnoa",
            "yopougon", "abobo", "cocody", "marcory", "treichville",
            "divo", "man", "bondoukou", "odienne", "ferkessedougou"
        ]

        # Partis connus
        known_parties = [
            "rhdp", "pdci", "fpi", "independant", "indépendant",
            "udpci", "ppa-ci", "eds"
        ]

        localities = [loc for loc in known_localities if loc in query_lower]
        parties = [p for p in known_parties if p in query_lower]

        return {
            "localities": localities,
            "parties": parties,
            "metrics": []
        }

    def _parse_llm_response(self, response_text: str) -> list[dict]:
        """Parse la réponse JSON du LLM."""
        if not response_text:
            logger.warning("Réponse LLM vide")
            return []

        # Nettoyer la réponse
        text = response_text.strip()

        # Enlever les backticks markdown si présents
        text = re.sub(r'^```json?\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Essayer de trouver le JSON dans la réponse (parfois le LLM ajoute du texte avant/après)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)

        try:
            data = json.loads(text)
            return data.get("disambiguated_questions", [])
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON: {e}")
            logger.warning(f"Réponse brute (premiers 500 chars): {response_text[:500]}")

            # Tenter de parser comme liste directe
            try:
                list_match = re.search(r'\[[\s\S]*\]', text)
                if list_match:
                    data = json.loads(list_match.group(0))
                    if isinstance(data, list):
                        return data
            except json.JSONDecodeError:
                pass

            return []

    def _validate_dq(self, dq_data: dict, original_query: str) -> Optional[DisambiguatedQuestion]:
        """Valide et convertit un DQ depuis les données JSON."""
        required_fields = ["dq_id", "interpretation", "explicit_question"]

        for field in required_fields:
            if field not in dq_data:
                logger.warning(f"DQ invalide - champ manquant: {field}")
                return None

        # Valider la confiance
        confidence = dq_data.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            confidence = 0.5

        # Extraire les entités
        entities = dq_data.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}
        entities.setdefault("localities", [])
        entities.setdefault("parties", [])
        entities.setdefault("metrics", [])

        # Extraire les hypothèses
        assumptions = dq_data.get("assumptions", [])
        if not isinstance(assumptions, list):
            assumptions = []

        return DisambiguatedQuestion(
            dq_id=str(dq_data["dq_id"]),
            original_query=original_query,
            interpretation=str(dq_data["interpretation"]),
            explicit_question=str(dq_data["explicit_question"]),
            confidence=float(confidence),
            entities=entities,
            assumptions=assumptions
        )

    def generate(self, query: str) -> ClarificationTree:
        """
        Génère un arbre de clarifications pour une requête.

        Args:
            query: Requête utilisateur originale

        Returns:
            ClarificationTree avec les DQs générées
        """
        logger.info(f"Génération de clarifications pour: '{query[:50]}...'")

        # Construire le prompt
        user_prompt = f"""REQUÊTE UTILISATEUR:
"{query}"

Génère {self.min_dqs} à {self.max_dqs} interprétations distinctes de cette requête.
Retourne UNIQUEMENT le JSON structuré."""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=2048
            )

            response_text = response.choices[0].message.content
            logger.debug(f"Réponse LLM reçue ({len(response_text) if response_text else 0} chars)")
            dqs_data = self._parse_llm_response(response_text)

            # Valider et convertir les DQs
            valid_dqs = []
            for dq_data in dqs_data:
                dq = self._validate_dq(dq_data, query)
                if dq:
                    valid_dqs.append(dq)

            # S'assurer qu'on a au moins une DQ
            if not valid_dqs:
                # Extraire des entités basiques de la requête
                entities = self._extract_basic_entities(query)
                # Créer une DQ par défaut
                valid_dqs.append(DisambiguatedQuestion(
                    dq_id="DQ_DEFAULT",
                    original_query=query,
                    interpretation="Interprétation littérale de la requête",
                    explicit_question=query,
                    confidence=1.0,
                    entities=entities,
                    assumptions=["Interprétation directe sans désambiguïsation"]
                ))
                logger.info(f"DQ par défaut créée avec entités: {entities}")

            # Trier par confiance
            valid_dqs.sort(key=lambda x: x.confidence, reverse=True)

            # Limiter au maximum
            valid_dqs = valid_dqs[:self.max_dqs]

            tree = ClarificationTree(
                original_query=query,
                disambiguated_questions=valid_dqs,
                generation_metadata={
                    "model": self.model,
                    "num_generated": len(valid_dqs),
                    "temperature": self.temperature
                }
            )

            logger.info(f"Généré {len(valid_dqs)} DQs")
            return tree

        except Exception as e:
            logger.error(f"Erreur génération: {e}")
            # Extraire des entités basiques de la requête
            entities = self._extract_basic_entities(query)
            # Retourner un arbre minimal
            return ClarificationTree(
                original_query=query,
                disambiguated_questions=[
                    DisambiguatedQuestion(
                        dq_id="DQ_FALLBACK",
                        original_query=query,
                        interpretation="Interprétation par défaut (erreur de génération)",
                        explicit_question=query,
                        confidence=1.0,
                        entities=entities,
                        assumptions=["Fallback suite à erreur"]
                    )
                ],
                generation_metadata={"error": str(e)}
            )

    def generate_simple(self, query: str) -> list[str]:
        """
        Version simplifiée - retourne juste les questions explicites.

        Args:
            query: Requête utilisateur

        Returns:
            Liste de questions explicites
        """
        tree = self.generate(query)
        return [dq.explicit_question for dq in tree.disambiguated_questions]


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = ClarificationGenerator()

    test_queries = [
        "Résultats à Abidjan",
        "Qui a gagné ?",
        "Performance du RHDP",
        "Candidats avec plus de 50%",
    ]

    print("=" * 60)
    print("TEST DU GÉNÉRATEUR DE CLARIFICATIONS")
    print("=" * 60)

    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        tree = generator.generate(query)

        for dq in tree.disambiguated_questions:
            print(f"\n  [{dq.dq_id}] (conf: {dq.confidence:.2f})")
            print(f"  Interprétation: {dq.interpretation}")
            print(f"  Question: {dq.explicit_question}")
            print(f"  Entités: {dq.entities}")
