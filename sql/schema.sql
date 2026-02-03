PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS circonscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    nom TEXT,
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
    nom TEXT NOT NULL,
    parti TEXT,
    score INTEGER,
    pourcentage REAL,
    elu BOOLEAN DEFAULT 0,
    FOREIGN KEY (circonscription_id)
        REFERENCES circonscriptions(id)
);
