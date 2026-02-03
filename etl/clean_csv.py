"""
Script de nettoyage des CSV extraits du PDF des résultats électoraux.
Corrige les problèmes de structure : régions sur plusieurs lignes, colonnes décalées, etc.
"""

import pandas as pd
import glob
import re

def clean_region_name(text):
    """Nettoie les noms de région qui sont sur plusieurs lignes."""
    if pd.isna(text):
        return None
    # Supprimer les retours à la ligne et joindre les caractères
    cleaned = str(text).replace("\n", "").replace("\r", "").strip()
    # Insérer des tirets entre les mots si nécessaire (ex: ASSAITYBENGA -> ASSAIT-YBENGA)
    return cleaned

def clean_number(text):
    """Nettoie les nombres (supprime espaces)."""
    if pd.isna(text):
        return None
    s = str(text).replace(" ", "").replace("\n", "").strip()
    if s.replace("-", "").isdigit():
        return int(s)
    return None

def clean_percentage(text):
    """Nettoie les pourcentages."""
    if pd.isna(text):
        return None
    s = str(text).replace("%", "").replace(",", ".").replace(" ", "").strip()
    try:
        return float(s)
    except ValueError:
        return None

def clean_text(text):
    """Nettoie le texte général."""
    if pd.isna(text):
        return None
    return str(text).replace("\n", " ").replace("\r", "").strip()

def process_csv(input_path: str, output_path: str):
    """Traite un fichier CSV et le nettoie."""

    # Lire le CSV brut
    df = pd.read_csv(input_path, header=None, dtype=str)

    # Identifier les colonnes selon la structure du PDF
    # Colonnes: REGION, CODE, CIRCONSCRIPTION, NB_BV, INSCRITS, VOTANTS, TAUX_PART,
    #           NULS, EXPRIMES, BULL_BLANCS_NB, BULL_BLANCS_PCT, PARTI, CANDIDAT, SCORE, PCT, ELU

    circonscriptions = []
    candidats = []

    current_region = None
    current_circ = None

    for idx, row in df.iterrows():
        # Ignorer la ligne d'en-tête
        if idx < 2:
            continue

        # Ignorer la ligne TOTAL
        if str(row.iloc[0]).upper() == "TOTAL":
            continue

        # Vérifier si c'est une nouvelle région (colonne 0 non vide et pas juste des espaces)
        col0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        col1 = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""

        if col0 and col0.upper() not in ["NAN", "NONE", ""]:
            current_region = clean_region_name(col0)

        # Vérifier si c'est une nouvelle circonscription (code présent en colonne 1)
        if col1 and col1 not in ["", "nan", "None"] and re.match(r'^\d{3}$', col1):
            # Nouvelle circonscription
            current_circ = {
                "region": current_region,
                "code": col1,
                "nom": clean_text(row.iloc[2]),
                "nb_bv": clean_number(row.iloc[3]),
                "inscrits": clean_number(row.iloc[4]),
                "votants": clean_number(row.iloc[5]),
                "taux_participation": clean_percentage(row.iloc[6]),
                "nuls": clean_number(row.iloc[7]),
                "exprimes": clean_number(row.iloc[8]),
                "bulletins_blancs": clean_number(row.iloc[9]),
                "taux_bulletins_blancs": clean_percentage(row.iloc[10]),
            }
            circonscriptions.append(current_circ)

        # Extraire les données du candidat (colonnes 11-15)
        parti = clean_text(row.iloc[11]) if len(row) > 11 else None
        candidat = clean_text(row.iloc[12]) if len(row) > 12 else None
        score = clean_number(row.iloc[13]) if len(row) > 13 else None
        pourcentage = clean_percentage(row.iloc[14]) if len(row) > 14 else None
        elu = 1 if len(row) > 15 and pd.notna(row.iloc[15]) and "ELU" in str(row.iloc[15]).upper() else 0

        if candidat and current_circ:
            candidats.append({
                "code_circ": current_circ["code"],
                "region": current_region,
                "parti": parti,
                "nom": candidat,
                "score": score,
                "pourcentage": pourcentage,
                "elu": elu
            })

    # Créer les DataFrames nettoyés
    df_circ = pd.DataFrame(circonscriptions)
    df_cand = pd.DataFrame(candidats)

    # Sauvegarder
    circ_path = output_path.replace(".csv", "_circonscriptions.csv")
    cand_path = output_path.replace(".csv", "_candidats.csv")

    df_circ.to_csv(circ_path, index=False, encoding="utf-8-sig")
    df_cand.to_csv(cand_path, index=False, encoding="utf-8-sig")

    print(f"Circonscriptions: {len(df_circ)} lignes -> {circ_path}")
    print(f"Candidats: {len(df_cand)} lignes -> {cand_path}")

    return df_circ, df_cand


if __name__ == "__main__":
    input_dir = "C:\\Users\\Goldman_Junior\\Documents\\Ai hiring\\data"
    output_dir = "C:\\Users\\Goldman_Junior\\Documents\\Ai hiring\\data\\cleaned"

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Traiter tous les CSV d'extraction
    fichiers = glob.glob(f"{input_dir}/extraction_tableau_*.csv")

    all_circ = []
    all_cand = []

    for f in fichiers:
        print(f"\n=== Traitement de {f} ===")
        output_path = f"{output_dir}/cleaned_{os.path.basename(f)}"
        df_circ, df_cand = process_csv(f, output_path)
        all_circ.append(df_circ)
        all_cand.append(df_cand)

    # Fusionner tous les fichiers
    if all_circ:
        df_all_circ = pd.concat(all_circ, ignore_index=True).drop_duplicates()
        df_all_cand = pd.concat(all_cand, ignore_index=True)

        df_all_circ.to_csv(f"{output_dir}/all_circonscriptions.csv", index=False, encoding="utf-8-sig")
        df_all_cand.to_csv(f"{output_dir}/all_candidats.csv", index=False, encoding="utf-8-sig")

        print(f"\n=== RÉSUMÉ ===")
        print(f"Total circonscriptions: {len(df_all_circ)}")
        print(f"Total candidats: {len(df_all_cand)}")

        print(f"\n=== Aperçu circonscriptions ===")
        print(df_all_circ.head(10))

        print(f"\n=== Aperçu candidats ===")
        print(df_all_cand.head(10))
