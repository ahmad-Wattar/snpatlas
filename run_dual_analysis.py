#!/usr/bin/env python3
"""
Dual-Dataset SNP-Analyse.
"""

from snp_analysis_pipeline import run_dual_dataset_analysis

def main():
    """
    Hauptfunktion für die Dual-Dataset-Analyse.
    """
    
    # Konfiguration für Dataset 1
    dataset1_config = {
        'name': 'dataset1',
        'path': 'csv_datein/dataset1/',  # Pfad zu den ersten CSV-Dateien
        'pattern': '*.csv',  # Alle CSV-Dateien im Verzeichnis
        'sep': ';'
    }
    
    # Konfiguration für Dataset 2  
    dataset2_config = {
        'name': 'dataset2',
        'path': 'csv_datein/dataset2/',  #  # Pfad zu den zweiten CSV-Dateien
        'pattern': '*.csv',  # Alle CSV-Dateien im Verzeichnis
        'sep': ';'
    }
    
    # Dual-Dataset-Analyse ausführen
    print("🚀 Starte Dual-Dataset SNP-Analyse...")
    pipeline1, pipeline2 = run_dual_dataset_analysis(
        dataset1_config=dataset1_config,
        dataset2_config=dataset2_config,
        output_base_dir="results"
    )
    
    print("\n Analyse erfolgreich abgeschlossen!")
    print(" Ergebnisse finden Sie in:")
    print(f"   - Dataset 1: results/dataset1/")
    print(f"   - Dataset 2: results/dataset2/")
    
    print(f"\n Dataset 1: {pipeline1.dataset_name}")
    print(f"   - Anzahl Zeilen: {len(pipeline1.df)}")
    print(f"   - Anzahl Accessions: {pipeline1.n_accessions}")
    
    print(f"\n Dataset 2: {pipeline2.dataset_name}")
    print(f"   - Anzahl Zeilen: {len(pipeline2.df)}")
    print(f"   - Anzahl Accessions: {pipeline2.n_accessions}")


if __name__ == "__main__":
    main()
