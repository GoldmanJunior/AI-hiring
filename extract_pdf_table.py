"""
Extraction de données tabulaires depuis un PDF avec Camelot.
Camelot est aussi puissant que Tabula mais ne nécessite pas Java.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Installe les dépendances nécessaires."""
    packages = ["camelot-py[base]", "opencv-python", "pandas", "openpyxl", "ghostscript"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


def extract_with_camelot(pdf_path: str, pages: str = "all", flavor: str = "lattice") -> list:
    """
    Extrait les tableaux d'un PDF avec Camelot.

    Args:
        pdf_path: Chemin vers le fichier PDF
        pages: Pages à extraire ("all", "1", "1-3", "1,3,5", etc.)
        flavor: "lattice" pour tableaux avec bordures, "stream" pour sans bordures

    Returns:
        Liste de DataFrames pandas
    """
    import camelot

    tables = camelot.read_pdf(
        pdf_path,
        pages=pages,
        flavor=flavor,
        strip_text='\n',
    )

    print(f"Camelot a détecté {len(tables)} tableau(x)")

    # Convertir en liste de DataFrames
    dataframes = [table.df for table in tables]

    # Afficher les scores de précision
    for i, table in enumerate(tables):
        print(f"  Tableau {i+1}: {len(table.df)} lignes, précision: {table.parsing_report.get('accuracy', 'N/A')}%")

    return dataframes


def extract_with_camelot_stream(pdf_path: str, pages: str = "all") -> list:
    """
    Extraction en mode stream (pour tableaux sans bordures claires).
    """
    import camelot

    tables = camelot.read_pdf(
        pdf_path,
        pages=pages,
        flavor="stream",
        strip_text='\n',
        edge_tol=50,  # Tolérance pour détecter les bords
        row_tol=10,   # Tolérance pour les lignes
    )

    print(f"Camelot (stream) a détecté {len(tables)} tableau(x)")

    dataframes = [table.df for table in tables]

    for i, table in enumerate(tables):
        print(f"  Tableau {i+1}: {len(table.df)} lignes, précision: {table.parsing_report.get('accuracy', 'N/A')}%")

    return dataframes


def clean_dataframe(df):
    """Nettoie un DataFrame extrait."""
    import pandas as pd

    # Supprimer les lignes entièrement vides
    df = df.dropna(how="all")

    # Supprimer les colonnes entièrement vides
    df = df.dropna(axis=1, how="all")

    # Remplacer les NaN par des chaînes vides
    df = df.fillna("")

    # Nettoyer les retours à la ligne et espaces multiples dans les cellules
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (df[col]
                .astype(str)
                .str.replace(r'\r\n|\r|\n', ' ', regex=True)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )

    return df


def save_to_excel(dataframes: list, output_path: str):
    """Sauvegarde les DataFrames dans un fichier Excel."""
    import pandas as pd

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for i, df in enumerate(dataframes):
            sheet_name = f"Tableau_{i + 1}"
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

    print(f"Données sauvegardées dans: {output_path}")


def save_to_csv(dataframes: list, output_dir: str, prefix: str = "extraction"):
    """Sauvegarde chaque DataFrame dans un fichier CSV séparé."""
    os.makedirs(output_dir, exist_ok=True)

    for i, df in enumerate(dataframes):
        output_path = os.path.join(output_dir, f"{prefix}_tableau_{i + 1}.csv")
        df.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")
        print(f"Tableau {i + 1} ({len(df)} lignes) -> {output_path}")


# --- Exemple d'utilisation ---

if __name__ == "__main__":
    # Configuration
    PDF_PATH = "C:\\Users\\Goldman_Junior\\Documents\\Ai hiring\\EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf"
    OUTPUT_DIR = "C:\\Users\\Goldman_Junior\\Documents\\Ai hiring\\data"

    print("=== Extraction avec Camelot ===\n")

    try:
        # Essayer d'abord le mode lattice (tableaux avec bordures)
        print("Tentative d'extraction en mode LATTICE (tableaux avec bordures)...")
        tables = extract_with_camelot(PDF_PATH, pages="all", flavor="lattice")

        if not tables or all(len(t) == 0 for t in tables):
            # Si échec, essayer le mode stream
            print("\nMode lattice sans résultat, essai en mode STREAM...")
            tables = extract_with_camelot_stream(PDF_PATH, pages="all")

        print(f"\nNombre total de tableaux: {len(tables)}")

        if tables:
            # Nettoyer les DataFrames
            cleaned_tables = [clean_dataframe(df) for df in tables]

            # Filtrer les tableaux vides
            cleaned_tables = [df for df in cleaned_tables if len(df) > 0]

            print(f"Tableaux après nettoyage: {len(cleaned_tables)}")

            if cleaned_tables:
                # Afficher un aperçu du premier tableau
                print("\n=== Aperçu du premier tableau ===")
                print(f"Colonnes: {len(cleaned_tables[0].columns)}")
                print(f"Lignes: {len(cleaned_tables[0])}")
                print(cleaned_tables[0].head(15).to_string())

                # Sauvegarder en CSV
                print("\n=== Sauvegarde des fichiers ===")
                save_to_csv(cleaned_tables, OUTPUT_DIR, "extraction")

                # Optionnel: sauvegarder en Excel
                # save_to_excel(cleaned_tables, os.path.join(OUTPUT_DIR, "extraction_complete.xlsx"))

                print(f"\n=== Extraction terminée avec succès ===")
                print(f"Fichiers sauvegardés dans: {OUTPUT_DIR}")
        else:
            print("Aucun tableau trouvé dans le PDF.")

    except ImportError as e:
        print(f"Erreur d'import: {e}")
        print("\nInstallez les dépendances avec:")
        print("  pip install camelot-py[base] opencv-python ghostscript")
    except Exception as e:
        print(f"Erreur lors de l'extraction: {e}")
        import traceback
        traceback.print_exc()
