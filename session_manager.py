"""
Gestionnaire de session pour mémoriser le contexte utilisateur.
Stocke les sélections et préférences pendant la session.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Contexte d'une session utilisateur."""
    session_id: str
    created_at: float
    last_activity: float

    # Entités mémorisées
    selected_localities: list[str] = field(default_factory=list)
    selected_parties: list[str] = field(default_factory=list)
    selected_candidates: list[str] = field(default_factory=list)

    # Historique des questions (pour le contexte)
    query_history: list[dict] = field(default_factory=list)

    # Préférences
    preferred_route: Optional[str] = None  # "sql" ou "rag"

    def add_selection(self, entity_type: str, value: str):
        """Ajoute une sélection à la session."""
        if entity_type == "locality":
            if value not in self.selected_localities:
                self.selected_localities.append(value)
        elif entity_type == "party":
            if value not in self.selected_parties:
                self.selected_parties.append(value)
        elif entity_type == "candidate":
            if value not in self.selected_candidates:
                self.selected_candidates.append(value)
        self.last_activity = time.time()

    def add_query(self, question: str, answer: str, entities: dict):
        """Ajoute une requête à l'historique."""
        self.query_history.append({
            "question": question,
            "answer": answer[:200],  # Tronquer pour économiser la mémoire
            "entities": entities,
            "timestamp": time.time()
        })
        # Garder seulement les 10 dernières requêtes
        if len(self.query_history) > 10:
            self.query_history = self.query_history[-10:]
        self.last_activity = time.time()

    def get_context_prompt(self) -> str:
        """Génère un prompt de contexte pour le LLM."""
        context_parts = []

        if self.selected_localities:
            context_parts.append(f"Localités sélectionnées: {', '.join(self.selected_localities)}")

        if self.selected_parties:
            context_parts.append(f"Partis sélectionnés: {', '.join(self.selected_parties)}")

        if self.selected_candidates:
            context_parts.append(f"Candidats sélectionnés: {', '.join(self.selected_candidates)}")

        if self.query_history:
            last_query = self.query_history[-1]
            context_parts.append(f"Dernière question: {last_query['question']}")

        if context_parts:
            return "CONTEXTE DE SESSION:\n" + "\n".join(context_parts)
        return ""

    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "session_id": self.session_id,
            "selected_localities": self.selected_localities,
            "selected_parties": self.selected_parties,
            "selected_candidates": self.selected_candidates,
            "query_count": len(self.query_history),
            "last_activity": self.last_activity
        }


class SessionManager:
    """
    Gestionnaire de sessions en mémoire.
    """

    def __init__(self, session_ttl: int = 3600):
        """
        Initialise le gestionnaire.

        Args:
            session_ttl: Durée de vie des sessions en secondes (défaut: 1 heure)
        """
        self.session_ttl = session_ttl
        self._sessions: dict[str, SessionContext] = {}
        self._lock = Lock()

        logger.info(f"SessionManager initialisé (TTL: {session_ttl}s)")

    def create_session(self) -> str:
        """Crée une nouvelle session."""
        session_id = str(uuid.uuid4())
        now = time.time()

        with self._lock:
            self._sessions[session_id] = SessionContext(
                session_id=session_id,
                created_at=now,
                last_activity=now
            )

        logger.debug(f"Session créée: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Récupère une session existante."""
        with self._lock:
            self._cleanup_expired()

            session = self._sessions.get(session_id)
            if session:
                session.last_activity = time.time()
                return session
            return None

    def get_or_create_session(self, session_id: Optional[str]) -> SessionContext:
        """Récupère ou crée une session."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        # Créer une nouvelle session
        new_id = self.create_session()
        return self._sessions[new_id]

    def update_session(
        self,
        session_id: str,
        question: str,
        answer: str,
        entities: dict
    ):
        """Met à jour une session avec les résultats d'une requête."""
        session = self.get_session(session_id)
        if not session:
            return

        # Ajouter la requête à l'historique
        session.add_query(question, answer, entities)

        # Mémoriser les entités mentionnées
        for locality in entities.get("localities", []):
            session.add_selection("locality", locality)

        for party in entities.get("parties", []):
            session.add_selection("party", party)

        logger.debug(f"Session mise à jour: {session_id}")

    def clear_session(self, session_id: str):
        """Efface une session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug(f"Session effacée: {session_id}")

    def _cleanup_expired(self):
        """Nettoie les sessions expirées."""
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_activity > self.session_ttl
        ]

        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.debug(f"Sessions expirées nettoyées: {len(expired)}")

    def get_stats(self) -> dict:
        """Retourne les statistiques des sessions."""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "session_ttl": self.session_ttl
            }


# Instance globale
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Retourne l'instance singleton du gestionnaire de sessions."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Test standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    manager = SessionManager()

    # Test création session
    print("=== Test Session ===")
    session_id = manager.create_session()
    print(f"Session créée: {session_id}")

    # Test récupération
    session = manager.get_session(session_id)
    print(f"Session récupérée: {session.to_dict()}")

    # Test mise à jour
    manager.update_session(
        session_id,
        question="Résultats à Tiapoum ?",
        answer="Le RHDP a obtenu 65% des voix à Tiapoum.",
        entities={"localities": ["tiapoum"], "parties": ["rhdp"]}
    )

    session = manager.get_session(session_id)
    print(f"Après mise à jour: {session.to_dict()}")
    print(f"Contexte prompt: {session.get_context_prompt()}")

    # Test stats
    print(f"\nStats: {manager.get_stats()}")
