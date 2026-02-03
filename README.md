# Elections CI - Système d'Interrogation Intelligente

## Statut du Projet

> **PROJET EN COURS DE DEVELOPPEMENT - PREMIERE SOUMISSION**
>
> Ce projet constitue une première soumission et n'est pas encore terminé.
>
> **Travaux restants:**
> - Lecture d'articles scientifiques pour améliorer l'extraction des données tabulaires depuis les PDFs
> - Finalisation du Level 4 (optimisations et améliorations)
> - Tests approfondis et validation des résultats et interface graphique

---

## Description

Système d'interrogation intelligente des données électorales de Côte d'Ivoire utilisant une architecture hybride **SQL + RAG** (Retrieval-Augmented Generation) avec un routeur **ToC (Tree of Clarifications)** pour la désambiguïsation des requêtes.

Le système permet de poser des questions en langage naturel sur les résultats électoraux et obtient des réponses précises en combinant:
- **SQL** pour les requêtes analytiques (comptages, moyennes, classements)
- **RAG** pour les recherches sémantiques (contexte, comparaisons, descriptions)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI (api.py)                         │
│                      Point d'entrée REST                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ToC Router (toc_router.py)                   │
│              Orchestrateur Tree of Clarifications               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Clarification │    │  ToC Pruner   │    │ ToC Aggregator│
│   Generator   │    │ (toc_pruner)  │    │(toc_aggregator│
│(clarification_│    │   Élagage &   │    │  Synthèse des │
│  generator)   │    │   Routage     │    │   résultats   │
└───────────────┘    └───────┬───────┘    └───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────┐           ┌─────────────────────┐
│   SQL Analyzer      │           │   RAG Retriever     │
│  (sql_analyzer.py)  │           │  (rag_retriever.py) │
│                     │           │                     │
│  ┌───────────────┐  │           │  ┌───────────────┐  │
│  │  Groq LLM     │  │           │  │ FAISS Index   │  │
│  │  (Llama 3)    │  │           │  │ + Embeddings  │  │
│  └───────────────┘  │           │  └───────────────┘  │
│                     │           │                     │
│  ┌───────────────┐  │           │  ┌───────────────┐  │
│  │   SQLite DB   │  │           │  │Entity Normaliz│  │
│  │ elections.db  │  │           │  │    + Fuzzy    │  │
│  └───────────────┘  │           │  └───────────────┘  │
└─────────────────────┘           └─────────────────────┘
```

---

## Pipeline ToC (Tree of Clarifications)

Le système utilise une approche innovante pour traiter les questions ambiguës:

```
Question Utilisateur
        │
        ▼
┌─────────────────────────────────────┐
│  STEP 1: Génération des DQs         │
│  (Disambiguation Questions)         │
│  - Génère 2-5 interprétations       │
│  - Extrait les entités (lieux,      │
│    partis, métriques)               │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  STEP 2: Élagage & Routage          │
│  - Classification intent (keywords) │
│  - Validation faisabilité SQL/RAG   │
│  - Élimination des DQs invalides    │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  STEP 3-4: Exécution Parallèle      │
│  - SQL: Génération + Exécution      │
│  - RAG: Recherche vectorielle +     │
│         Synthèse LLM                │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  STEP 5-6: Agrégation               │
│  - Fusion des résultats SQL/RAG     │
│  - Génération réponse finale        │
│  - Calcul score de confiance        │
└─────────────────┬───────────────────┘
                  ▼
        Réponse Structurée
        (answer, sources, confidence)
```

---

## Technologies Utilisées

### Backend & API
| Technologie | Version | Usage |
|-------------|---------|-------|
| Python | 3.11+ | Langage principal |
| FastAPI | 0.104+ | Framework API REST |
| Uvicorn | 0.24+ | Serveur ASGI |
| Pydantic | 2.0+ | Validation des données |

### LLM & Embeddings
| Technologie | Usage |
|-------------|-------|
| Groq API | Inférence LLM (Llama 3 70B) |
| Sentence-Transformers | Embeddings multilingual |
| `paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embedding |

### Base de Données & Recherche
| Technologie | Usage |
|-------------|-------|
| SQLite | Stockage des données électorales |
| FAISS | Index vectoriel pour RAG |
| RapidFuzz | Fuzzy matching entités |

### Extraction PDF
| Technologie | Usage |
|-------------|-------|
| Camelot | Extraction tableaux PDF |
| pdfplumber | Alternative extraction |
| OpenCV | Traitement images PDF |

---

## Structure du Projet

```
Ai hiring/
├── api.py                    # Point d'entrée FastAPI
├── schemas.py                # Modèles Pydantic
├── config.yaml               # Configuration centrale
├── requirements.txt          # Dépendances Python
├── .env                      # Variables d'environnement (à créer)
├── .env.example              # Exemple de configuration
│
├── # === PIPELINE ToC ===
├── toc_router.py             # Orchestrateur principal
├── clarification_generator.py # Génération des interprétations
├── toc_pruner.py             # Élagage et routage
├── toc_aggregator.py         # Agrégation des résultats
│
├── # === COMPOSANTS ===
├── sql_analyzer.py           # Conversion NL -> SQL + exécution
├── rag_retriever.py          # Recherche vectorielle FAISS
├── rag_indexer.py            # Création de l'index RAG
├── index_database.py         # Indexation de la BD
├── intent_classifier.py      # Classification SQL/RAG (keywords)
├── entity_normalizer.py      # Normalisation entités (fuzzy)
├── citation_manager.py       # Gestion des sources/citations
│
├── # === ETL ===
├── extract_pdf_table.py      # Extraction tableaux PDF
├── etl/
│   ├── clean_csv.py          # Nettoyage données CSV
│   ├── load_data.py          # Chargement dans SQLite
│   └── elections.db          # Base de données SQLite
│
├── # === DONNÉES ===
├── aliases/
│   ├── localities_aliases.json  # Alias localités
│   └── parties_aliases.json     # Alias partis politiques
│
└── rag_index/                # Index FAISS (généré)
    ├── vectors.faiss
    └── metadata.pkl
```

---

## Installation

### 1. Cloner le projet

```bash
git clone <repository_url>
cd "Ai hiring"
```

### 2. Créer l'environnement virtuel

```bash
python -m venv env

# Windows
env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine:

```env
# Clé API Groq (obligatoire)
GROQ_API_KEY=votre_cle_api_groq

# Optionnel
GROQ_MODEL=llama-3.3-70b-versatile
```

> Obtenir une clé API Groq: https://console.groq.com/

---

## Guide d'Exécution - Étape par Étape

### Étape 1: Extraction des données PDF (si nécessaire)

Si vous avez un nouveau PDF de résultats électoraux:

```bash
python extract_pdf_table.py
```

Cela génère un fichier CSV avec les données extraites.

### Étape 2: Nettoyage des données CSV

```bash
python etl/clean_csv.py
```

Nettoie et formate les données pour le chargement.

### Étape 3: Chargement dans la base SQLite

```bash
python etl/load_data.py
```

Crée/remplit la base `etl/elections.db` avec les tables:
- `circonscriptions` - Données par circonscription
- `candidats` - Données par candidat

### Étape 4: Création de l'index RAG

```bash
python index_database.py
```

Génère l'index FAISS dans `rag_index/`:
- `vectors.faiss` - Vecteurs d'embedding
- `metadata.pkl` - Métadonnées des chunks

### Étape 5: Démarrer l'API

```bash
uvicorn api:app --reload --host localhost --port 8000
```

L'API est accessible sur `http://localhost:8000`

---

## Utilisation de l'API

### Documentation Interactive

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil |
| GET | `/health` | État de santé du système |
| **POST** | **`/query`** | **Interrogation principale (ToC)** |
| GET | `/intent?question=...` | Classification d'intent |
| GET | `/schema` | Schéma de la base de données |
| GET | `/stats` | Statistiques du système |
| GET | `/tables/{name}/sample` | Échantillon d'une table |

### Exemple de Requête

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les résultats du RHDP à Abidjan ?",
    "explain": true
  }'
```

### Format de Réponse

```json
{
  "success": true,
  "original_query": "Quels sont les résultats du RHDP à Abidjan ?",
  "answer": "Le RHDP a obtenu...",
  "method": "toc",
  "confidence": 0.95,
  "interpretations": [
    {
      "dq_id": "DQ1",
      "route": "sql",
      "question": "Quel est le score total du RHDP...",
      "answer": "...",
      "success": true
    }
  ],
  "sql_facts": ["..."],
  "rag_insights": ["..."],
  "sources": [...]
}
```

---

## Configuration

Le fichier `config.yaml` permet de personnaliser:

```yaml
database:
  path: "etl/elections.db"

embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

faiss:
  index_path: "rag_index/vectors.faiss"
  metadata_path: "rag_index/metadata.pkl"

retrieval:
  top_k: 5
  min_confidence: 0.5

toc:
  min_disambiguations: 2
  max_disambiguations: 5
  parallel_execution: true

groq:
  model: "llama-3.3-70b-versatile"
  temperature: 0.1
  max_tokens: 1024
```

---

## Sécurité

- **Requêtes SQL**: Seules les requêtes `SELECT` sont autorisées
- **Validation**: Toutes les requêtes sont validées avant exécution
- **Pas d'injection**: Le système refuse les requêtes dangereuses (DROP, DELETE, UPDATE, etc.)

---

## Améliorations Futures (Level 4)

- [ ] Amélioration de l'extraction PDF avec techniques avancées
- [ ] Cache des résultats fréquents
- [ ] Métriques et monitoring
- [ ] Tests unitaires et d'intégration
- [ ]observavilité
- [ ] Interface utilisateur web


