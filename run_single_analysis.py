#!/usr/bin/env python3
"""
Single-Dataset SNP-Analyse.
"""

from snp_analysis_pipeline import SNPAnalysisPipeline

def main():
    """
    Hauptfunktion für die Single-Dataset-Analyse.
    """
    
    # Konfiguration für das Dataset
    dataset_config = {
        'name': 'Arabidopsis thaliana',
        'path': '.',  # Aktuelles Verzeichnis
        'pattern': '*.csv',  # Alle CSV-Dateien im Verzeichnis
        'sep': ';'
    }
    
    # Pipeline für ein Dataset erstellen
    pipeline = SNPAnalysisPipeline(
        dataset_name=dataset_config['name'],
        output_base_dir="results"
    )
    
    # Komplette Analyse ausführen
    pipeline.run_complete_analysis(
        data_path=dataset_config['path'],
        data_pattern=dataset_config['pattern'],
        data_sep=dataset_config['sep']
    )
    
    print(f" Analyse abgeschlossen! Ergebnisse in: results/{dataset_config['name']}/")
    print(f" Vollständiger Report: results/{dataset_config['name']}/reports/complete_report.txt")


if __name__ == "__main__":
    main()
