#!/usr/bin/env python3


import os
import sys
import time

def main():
    """
    Hauptfunktion für die vereinfachte SNP-Analyse.
    """
    print("=" * 80)
    print("Vereinfachte SNP-Analyse für HPC")
    print("=" * 80)
    
    # System-Info
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Konfiguration für das Dataset
    dataset_config = {
        'name': 'Arabidopsis_thaliana',
        'path': './dataset1/',  # dataset1 Ordner mit allen CSV-Dateien
        'pattern': '*.csv',  # Alle CSV-Dateien im dataset1 Verzeichnis
        'sep': ';'
    }
    
    print(f"\nDataset: {dataset_config['name']}")
    print(f"Data path: {dataset_config['path']}")
    print(f"Pattern: {dataset_config['pattern']}")
    
    # Überprüfe ob dataset1 existiert
    if not os.path.exists(dataset_config['path']):
        print(f"ERROR: Dataset path {dataset_config['path']} does not exist!")
        return
    
    # Zähle CSV-Dateien
    import glob
    csv_files = glob.glob(os.path.join(dataset_config['path'], dataset_config['pattern']))
    print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("ERROR: No CSV files found!")
        return
    
    # Zeige erste paar Dateien
    print("First few files:")
    for i, file in enumerate(csv_files[:5]):
        print(f"  {i+1}: {os.path.basename(file)}")
    
    try:
        # Versuche die Pipeline zu importieren und auszuführen
        print("\nImporting SNP analysis pipeline...")
        from snp_analysis_pipeline import SNPAnalysisPipeline
        
        # Pipeline erstellen
        pipeline = SNPAnalysisPipeline(
            dataset_name=dataset_config['name'],
            output_base_dir="results"
        )
        
        start_time = time.time()
        
        # Komplette Analyse ausführen
        print("\nStarting SNP analysis...")
        pipeline.run_complete_analysis(
            data_path=dataset_config['path'],
            data_pattern=dataset_config['pattern'],
            data_sep=dataset_config['sep']
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"Analyse erfolgreich abgeschlossen!")
        print(f"Laufzeit: {duration/3600:.2f} Stunden ({duration:.0f} Sekunden)")
        print(f"Ergebnisse in: results/{dataset_config['name']}/")
        print(f"{'='*60}")
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Trying to install missing packages...")
        
        # Versuche pip install
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'pandas', 'numpy', 'matplotlib', 'seaborn'])
            print("Packages installed, retrying...")
            # Retry
            from snp_analysis_pipeline import SNPAnalysisPipeline
            # ... rest of the analysis
        except Exception as e2:
            print(f"Failed to install packages: {e2}")
            return
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
