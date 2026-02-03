"""
Chargement des données nettoyées dans la base SQLite.
Utilise les CSV nettoyés générés par clean_csv.py
"""

import sqlite3
import pandas as pd
import os

# Chemins
DATA_DIR = "C:\\Users\\Goldman_Junior\\Documents\\Ai hiring\\data\\cleaned"
DB_PATH = "C:\\Users\\Goldman_Junior\\Documents\\Ai hiring\\etl\\elections.db"

# Supprimer l'ancienne base si elle existe
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# Connexion à la base de données
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")

# Créer les tables avec les bons types
conn.executescript("""
    CREATE TABLE IF NOT EXISTS circonscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region TEXT,
        code INTEGER NOT NULL UNIQUE,
        nom TEXT NOT NULL,
        nb_bv INTEGER,
        inscrits INTEGER,
        votants INTEGER,
        taux_participation REAL,
        nuls INTEGER,
        exprimes INTEGER,
        bulletins_blancs INTEGER,
        taux_bulletins_blancs REAL
    );

    CREATE TABLE IF NOT EXISTS candidats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        circonscription_id INTEGER NOT NULL,
        region TEXT,
        parti TEXT,
        nom TEXT NOT NULL,
        score INTEGER,
        pourcentage REAL,
        elu INTEGER DEFAULT 0,
        FOREIGN KEY (circonscription_id) REFERENCES circonscriptions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_candidats_circ ON candidats(circonscription_id);
    CREATE INDEX IF NOT EXISTS idx_candidats_parti ON candidats(parti);
    CREATE INDEX IF NOT EXISTS idx_candidats_elu ON candidats(elu);
""")

# Charger les CSV nettoyés
df_circ = pd.read_csv(f"{DATA_DIR}/all_circonscriptions.csv")
df_cand = pd.read_csv(f"{DATA_DIR}/all_candidats.csv")

print(f"Chargement de {len(df_circ)} circonscriptions et {len(df_cand)} candidats...")

# Convertir les types
df_circ["inscrits"] = df_circ["inscrits"].astype("Int64")
df_circ["votants"] = df_circ["votants"].astype("Int64")
df_circ["nb_bv"] = df_circ["nb_bv"].astype("Int64")
df_circ["nuls"] = df_circ["nuls"].astype("Int64")
df_circ["exprimes"] = df_circ["exprimes"].astype("Int64")
df_circ["bulletins_blancs"] = df_circ["bulletins_blancs"].astype("Int64")

# Insérer les circonscriptions
df_circ.to_sql("circonscriptions", conn, if_exists="append", index=False)

# Créer le mapping code -> id
cursor = conn.execute("SELECT id, code FROM circonscriptions")
map_circ = {code: id for id, code in cursor.fetchall()}

# Ajouter circonscription_id aux candidats
df_cand["circonscription_id"] = df_cand["code_circ"].map(map_circ)
df_cand = df_cand.drop(columns=["code_circ"])

# Insérer les candidats
df_cand.to_sql("candidats", conn, if_exists="append", index=False)

conn.commit()
conn.close()

print("Chargement terminé avec succès!")
print(f"Base de données: {DB_PATH}")
