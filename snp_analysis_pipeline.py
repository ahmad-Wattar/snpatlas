import pandas as pd
import glob
import os
import ast
import numpy as np

# HPC-optimierte matplotlib Konfiguration (kein Display)
import matplotlib
matplotlib.use('Agg')  # Agg Backend für HPC ohne Display
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
import sys
from io import StringIO

# Zusätzliche HPC-Optimierungen
plt.rcParams['figure.max_open_warning'] = 0  # Keine Warnungen für viele Plots

class SNPAnalysisPipeline:
    
    def __init__(self, dataset_name, output_base_dir="results"):
        """ 
        Parameters
        ----------
        dataset_name : str
            Name des Datensatzes ( "dataset1", "dataset2")
        output_base_dir : str
            Basis-Verzeichnis für alle Ausgaben
        """
        self.dataset_name = dataset_name
        self.output_base_dir = output_base_dir
        self.results_dir = os.path.join(output_base_dir, dataset_name)
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.reports_dir = os.path.join(self.results_dir, "reports")
        self.csv_dir = os.path.join(self.results_dir, "csv")
        
        # Verzeichnisse erstellen
        for dir_path in [self.results_dir, self.plots_dir, self.reports_dir, self.csv_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Datensatz-spezifische Variablen
        self.df = None
        self.n_accessions = None
        self.isoforms = None
        self.report_file = os.path.join(self.reports_dir, "complete_report.txt")
        self.pdf_report_path = os.path.join(self.plots_dir, "complete_report.pdf")
        self.plots_for_pdf = []  # Liste für alle Plots
        
        # Report-Datei initialisieren
        with open(self.report_file, "w") as f:
            f.write(f"SNP-Analyse Report für {self.dataset_name}\n")
            f.write("=" * 60 + "\n\n")
        
    def load_data(self, path=".", pattern="*.csv", sep=";"):
        """
        Liest eine oder mehrere CSV-Dateien ein und gibt ein kombiniertes DataFrame zurück.
        """
        files = glob.glob(os.path.join(path, pattern))
        if not files:
            raise FileNotFoundError(f"Keine Dateien gefunden mit Muster {pattern} in {path}")

        dfs = []
        with open(self.report_file, "a") as f:
            for file_path in files:
                f.write(f"Lade Datei: {file_path}\n")
                df = pd.read_csv(file_path, sep=sep)
                dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)
        with open(self.report_file, "a") as f:
            f.write(f"Eingelesen: {len(self.df)} Zeilen aus {len(files)} Dateien\n\n")
        return self.df

    def save_dataframe(self, df, filepath):
        """Speichert ein DataFrame als CSV-Datei."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        with open(self.report_file, "a") as f:
            f.write(f"Gespeichert: {filepath}\n")

    def load_dataframe(self, filepath):
        """Lädt ein DataFrame aus einer CSV-Datei."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")
        df = pd.read_csv(filepath)
        with open(self.report_file, "a") as f:
            f.write(f"Geladen: {filepath}\n")
        return df

    def count_effects(self, column='eff', save_path=None):
        """
        Zählt die Vorkommen pro Kategorie in einer Spalte und speichert das Ergebnis (optional).
        """
        counts = self.df[column].value_counts().reset_index()
        counts.columns = ['Effect', 'Raw_Count']
        
        with open(self.report_file, "a") as f:
            f.write(f"Effekt-Zählungen erstellt:\n{counts.to_string()}\n\n")

        if save_path:
            self.save_dataframe(counts, save_path)

        return counts

    def save_effect_plot(self, effect_counts, pdf_name="complete_report.pdf"):
        """Erstellt und speichert Effekt-Plots."""
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.barplot(data=effect_counts, x='Effect', y='Raw_Count', ax=ax)
        ax.set_yscale("log")
        ax.set_title(f'Verteilung der SNP-Effekte - {self.dataset_name} (log-Skala)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # PNG speichern
        png_path = os.path.join(self.plots_dir, "effect_counts.png")
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        with open(self.report_file, "a") as f:
            f.write(f"Plot als PNG gespeichert: {png_path}\n")

        # Zur PDF hinzufügen
        self.add_plot_to_pdf(fig)

        plt.close(fig)
        
        # PDF-Pfad für spätere Verwendung speichern
        self.pdf_report_path = os.path.join(self.plots_dir, pdf_name)

    def add_plot_to_pdf(self, fig):
        """
        Fügt einen Plot zur Liste für die PDF hinzu.
        """
        self.plots_for_pdf.append(fig)

    def create_final_pdf(self):
        """
        Erstellt die finale PDF mit allen gesammelten Plots.
        """
        if not self.plots_for_pdf:
            return
            
        with PdfPages(self.pdf_report_path) as pdf:
            # Titel-Seite
            fig_title = plt.figure(figsize=(12, 8))
            plt.text(0.1, 0.9, f"SNP-Analyse Report", fontsize=20, fontweight='bold')
            plt.text(0.1, 0.8, f"Dataset: {self.dataset_name}", fontsize=16)
            plt.text(0.1, 0.7, f"Erstellt: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12)
            #plt.text(0.1, 0.6, f"Enthält alle Analysen und Visualisierungen", fontsize=12)
            plt.axis('off')
            pdf.savefig(fig_title, bbox_inches='tight')
            plt.close(fig_title)
            
            # Alle Plots hinzufügen
            for fig in self.plots_for_pdf:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
        with open(self.report_file, "a") as f:
            f.write(f"Finale PDF erstellt: {self.pdf_report_path}\n")

    def count_unique_accessions(self, report_file="report.txt"):
        """
        Zählt die Anzahl eindeutiger Accessions aus der 'genotypes'-Spalte.
        """
        df = self.df.copy()
        df['genotypes_list'] = df['genotypes'].apply(lambda x: ast.literal_eval(x))
        all_accessions = set(acc for sublist in df['genotypes_list'] for acc in sublist)
        self.n_accessions = len(all_accessions)

        with open(self.report_file, "a") as f:
            f.write(f"Anzahl eindeutiger Accessions: {self.n_accessions}\n")
            f.write(f"Anzahl eindeutiger Accessions: {self.n_accessions}\n\n")
        return self.n_accessions

    def count_accessions_per_isoform(self, report_file="report.txt"):
        """
        Zählt die Anzahl eindeutiger Accessions pro Isoform.
        """
        df = self.df.copy()
        # Isoformen als Zahlen sortieren, nicht als Strings (sonst ist die Reihnfolge falsch ab 10)
        isoform_numbers = {int(t.split('.')[-1]) for t in df['transcript_id'] if '.' in t}
        self.isoforms = sorted(isoform_numbers)
        
        isoform_accessions = {}
        
        for iso in self.isoforms:
            iso_label = f".{iso}"
            df_iso = df[df['transcript_id'].str.endswith(iso_label)].copy()
            df_iso['genotypes_list'] = df_iso['genotypes'].apply(lambda x: ast.literal_eval(x))
            all_accessions = set(acc for sublist in df_iso['genotypes_list'] for acc in sublist)
            isoform_accessions[iso_label] = len(all_accessions)
        
        with open(self.report_file, "a") as f:
            for iso, count in isoform_accessions.items():
                f.write(f"Anzahl eindeutiger Accessions Isoform {iso}: {count}\n")
            f.write("\n")
        
        return isoform_accessions

    def count_genes_per_isoform(self, report_file="report.txt"):
        """
        Zählt die Anzahl eindeutiger Gene pro Isoform.
        """
        df = self.df.copy()
        # Isoformen als Zahlen sortieren, nicht als Strings
        isoform_numbers = {int(t.split('.')[-1]) for t in df['transcript_id'] if '.' in t}
        isoforms = sorted(isoform_numbers)
        
        isoform_genes = {}
        
        for iso in isoforms:
            iso_label = f".{iso}"
            df_iso = df[df['transcript_id'].str.endswith(iso_label)]
            n_genes_iso = df_iso['gene_id'].nunique()
            isoform_genes[iso_label] = n_genes_iso
        
        with open(self.report_file, "a") as f:
            for iso, count in isoform_genes.items():
                f.write(f"Anzahl eindeutiger Gene Isoform {iso}: {count}\n")
            f.write("\n")
        
        return isoform_genes

    def save_isoform_subsets(self, cols=None, isoforms=None, report_file="report.txt"):
        """
        Speichert für jede Isoform ein CSV mit ausgewählten Spalten. (für mehr Verständnis für die Daten)
        """
        if cols is None:
            cols = ['gene_id','transcript_id','aa_len','cds_len','cdna_len']
        
        if isoforms is None:
            # Isoformen als Zahlen sortieren, nicht als Strings
            isoform_numbers = {int(t.split('.')[-1]) for t in self.df['transcript_id'] if '.' in t}
            isoforms = [f".{iso}" for iso in sorted(isoform_numbers)]
        
        with open(self.report_file, "a") as f:
            for iso in isoforms:
                subset = self.df[self.df['transcript_id'].str.endswith(iso)][cols]
                if subset.empty:
                    msg = f"Isoform {iso}: keine Daten"
                else:
                    # Isoform-spezifischen Ordner erstellen
                    isoform_name = iso.replace('.', '')
                    isoform_csv_dir = os.path.join(self.csv_dir, f"isoform_{isoform_name}")
                    os.makedirs(isoform_csv_dir, exist_ok=True)
                    
                    out_file = os.path.join(isoform_csv_dir, f"isoform_subset.csv")
                    subset.to_csv(out_file, index=False)
                    msg = f"Isoform {iso}: {len(subset)} Zeilen → gespeichert in {out_file}"
                f.write(msg + "\n")

    def snp_effects_per_isoform(self, isoforms=None):
        """
        Berechnet SNP-Effekte pro Isoform, erstellt Barplots und speichert Ergebnisse.
        """
        if isoforms is None:
            # Isoformen als Zahlen sortieren, nicht als Strings
            isoform_numbers = {int(t.split('.')[-1]) for t in self.df['transcript_id'] if '.' in t}
            isoforms = [f".{iso}" for iso in sorted(isoform_numbers)]
        
        for iso in isoforms:
            # Isoform-spezifischen Ordner erstellen
            isoform_name = iso.replace('.', '')
            isoform_plots_dir = os.path.join(self.plots_dir, f"isoform_{isoform_name}")
            isoform_csv_dir = os.path.join(self.csv_dir, f"isoform_{isoform_name}")
            os.makedirs(isoform_plots_dir, exist_ok=True)
            os.makedirs(isoform_csv_dir, exist_ok=True)
            
            df_iso = self.df[self.df['transcript_id'].str.endswith(iso)].copy()
            if df_iso.empty:
                with open(self.report_file, "a") as f:
                    f.write(f"Isoform {iso}: keine Daten\n")
                continue
            
            df_iso['genotypes_list'] = df_iso['genotypes'].apply(lambda x: ast.literal_eval(x))
            df_iso['num_accessions'] = df_iso['genotypes_list'].apply(len)
            
            effect_counts_normalized = df_iso.groupby('eff')['num_accessions'].sum() / self.n_accessions
            effect_counts_sorted = effect_counts_normalized.sort_values(ascending=False)
            raw_counts = df_iso['eff'].value_counts().reindex(effect_counts_sorted.index)
            
            result = pd.DataFrame({
                'Effect': effect_counts_sorted.index,
                'Raw_Count': raw_counts.values,
                'Normalized_per_accession': effect_counts_sorted.values
            })
            
            csv_file = os.path.join(isoform_csv_dir, f"snp_effects.csv")
            result.to_csv(csv_file, index=False)
            
            # Plot
            fig = plt.figure(figsize=(14, 8))
            sns.barplot(data=result, x='Effect', y='Raw_Count', color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.yscale("log")
            plt.title(f'SNP Effekttypen Isoform {iso} - {self.dataset_name} (log-Skala)')
            plt.tight_layout()
            
            # PNG speichern
            plot_file = os.path.join(isoform_plots_dir, f"snp_effects.png")
            plt.savefig(plot_file, bbox_inches='tight', dpi=300)
            
            # Isoform-spezifische Plots NICHT zur PDF hinzufügen
            # self.add_plot_to_pdf(fig)  # Auskommentiert
            
            plt.close()

    def snp_effects_all_transcripts(self):
        """
        Berechnet SNP-Effekte für alle Transkripte zusammen(also alle Isoformen zusammen).
        """
        df = self.df.copy()
        df['genotypes_list'] = df['genotypes'].apply(lambda x: ast.literal_eval(x))
        df['num_accessions'] = df['genotypes_list'].apply(len)

        effect_counts_normalized = df.groupby('eff')['num_accessions'].sum() / self.n_accessions
        effect_counts_sorted = effect_counts_normalized.sort_values(ascending=False)
        raw_counts = df['eff'].value_counts().reindex(effect_counts_sorted.index)

        result = pd.DataFrame({
            'Effect': effect_counts_sorted.index,
            'Raw_Count': raw_counts.values,
            'Normalized_per_accession': effect_counts_sorted.values
        })

        csv_file = os.path.join(self.csv_dir, "snp_effects_all.csv")
        result.to_csv(csv_file, index=False)

        # CSV-Inhalt in Report schreiben
        with open(self.report_file, "a") as f:
            f.write(f"SNP-Effekte alle Transkripte - CSV-Inhalt:\n")
            f.write("=" * 60 + "\n")
            f.write(result.to_string(index=False))
            f.write("\n" + "=" * 60 + "\n\n")

        # Plot
        fig = plt.figure(figsize=(14, 8))
        sns.barplot(data=result, x='Effect', y='Raw_Count', color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.yscale("log") # logairthmiert, weil sonst die Verteilung Schiefe hat
        plt.title(f'SNP Effekttypen - alle Transkripte - {self.dataset_name} (log-Skala)')
        plt.tight_layout()
        
        # PNG speichern
        plot_file = os.path.join(self.plots_dir, "snp_effects_all.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        
        # Zur PDF hinzufügen
        self.add_plot_to_pdf(fig)
        
        plt.close()

        with open(self.report_file, "a") as f:
            f.write(f"Analyse abgeschlossen. Plot → {plot_file}, CSV → {csv_file}\n\n")

    def snp_counts_per_gene_analysis(self, top_n=20):
        """
        Analysiert SNP-Anzahlen pro Gen mit verschiedenen Visualisierungen.
        """
        snp_counts_per_gene = self.df.groupby('gene_id')['eff'].count().reset_index()
        snp_counts_per_gene.columns = ['gene_id', 'num_snps']

        csv_file = os.path.join(self.csv_dir, "snp_counts_per_gene.csv")
        snp_counts_per_gene.to_csv(csv_file, index=False)

        # Statistiken
        with open(self.report_file, "a") as f:
            f.write(f"Gesamtzahl Gene: {len(snp_counts_per_gene)}\n")
            f.write(f"Durchschnitt SNPs pro Gen: {snp_counts_per_gene['num_snps'].mean():.2f}\n")
            f.write(f"Median SNPs pro Gen: {snp_counts_per_gene['num_snps'].median()}\n\n")

        # Plots erstellen
        min_snp = snp_counts_per_gene['num_snps'].min()
        max_snp = snp_counts_per_gene['num_snps'].max()
        bins = np.logspace(np.log10(min_snp), np.log10(max_snp), 25)

        # Histogramm
        fig_hist = plt.figure(figsize=(12, 8))
        plt.hist(snp_counts_per_gene['num_snps'], bins=bins, color='skyblue', edgecolor='black')
        plt.xscale('log')
        plt.xlabel("Anzahl SNPs pro Gen (log-Skala)")
        plt.ylabel("Anzahl Gene")
        plt.title(f"Verteilung der SNP-Anzahlen pro Gen - {self.dataset_name}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # PNG speichern
        hist_file = os.path.join(self.plots_dir, "snp_counts_histogram.png")
        plt.savefig(hist_file, bbox_inches='tight', dpi=300)
        
        # Zur PDF hinzufügen
        self.add_plot_to_pdf(fig_hist)
        plt.close()

        # Top-N Gene
        top_genes = snp_counts_per_gene.nlargest(top_n, 'num_snps')
        fig_top = plt.figure(figsize=(12, 8))
        plt.barh(top_genes['gene_id'], top_genes['num_snps'])
        plt.xscale('log')
        plt.xlabel("Anzahl SNPs (log-Skala)")
        plt.ylabel("Gen")
        plt.title(f"Top {top_n} Gene mit den meisten SNPs - {self.dataset_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # PNG speichern
        top_file = os.path.join(self.plots_dir, f"snp_counts_top{top_n}.png")
        plt.savefig(top_file, bbox_inches='tight', dpi=300)
        
        # Zur PDF hinzufügen
        self.add_plot_to_pdf(fig_top)
        plt.close()

        with open(self.report_file, "a") as f:
            f.write(f"SNP-Analyse pro Gen abgeschlossen. Ergebnisse in {self.plots_dir}\n\n")

    def calculate_allele_frequencies_per_accession(self, genotypes_col='genotypes', snp_effects_col='eff'):
        """
        Berechnet Allelfrequenzen pro Accession. (Vor der Sortierung in nur 6 Kategorien)
        """
        def convert_to_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return [x]
            elif isinstance(x, list):
                return x
            else:
                return [x]
        
        df = self.df.copy()
        df['genotypes_list'] = df[genotypes_col].apply(convert_to_list)
        
        accession_allele_frequencies = {}
        
        for idx, row in df.iterrows():
            accessions = row['genotypes_list']
            effect = row[snp_effects_col]
            
            for accession in accessions:
                if accession not in accession_allele_frequencies:
                    accession_allele_frequencies[accession] = Counter()
                accession_allele_frequencies[accession][effect] += 1
        
        df_result = pd.DataFrame.from_dict(accession_allele_frequencies, orient='index').fillna(0).astype(int)
        df_result.index.name = 'accession'
        df_result['all_effect'] = df_result.sum(axis=1)
        
        csv_file = os.path.join(self.csv_dir, "allele_per_accession.csv")
        df_result.to_csv(csv_file)
        
        with open(self.report_file, "a") as f:
            f.write(f"Allelfrequenzen als DataFrame erstellt und in {csv_file} gespeichert\n\n")
        
        # Visualisierungen erstellen
        self.create_allele_frequency_plots(df_result)
        
        # Neue Kategorie-basierte Allelfrequenzen berechnen und visualisieren
        self.calculate_category_frequencies_per_accession()
        
        return df_result

    def calculate_category_frequencies_per_accession(self, genotypes_col='genotypes', snp_effects_col='eff'):
        """
        Berechnet Allelfrequenzen pro Accession für die 6 Effekt-Kategorien.
        Ähnlich zu calculate_allele_frequencies_per_accession, aber gruppiert nach Kategorien.
        """
        def convert_to_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return [x]
            elif isinstance(x, list):
                return x
            else:
                return [x]
        
        df = self.df.copy()
        df['genotypes_list'] = df[genotypes_col].apply(convert_to_list)
        df['effect_category'] = df[snp_effects_col].apply(self.classify_effect)
        
        accession_category_frequencies = {}
        
        for idx, row in df.iterrows():
            accessions = row['genotypes_list']
            category = row['effect_category']
            
            for accession in accessions:
                if accession not in accession_category_frequencies:
                    accession_category_frequencies[accession] = Counter()
                accession_category_frequencies[accession][category] += 1
        
        df_category_result = pd.DataFrame.from_dict(accession_category_frequencies, orient='index').fillna(0).astype(int)
        df_category_result.index.name = 'accession'
        
        # Die 6 Hauptkategorien sicherstellen
        categories = ['Protein-changing', 'Synonymous', 'Splice-related', 'UTR', 'Intron', 'Non-coding']
        for cat in categories:
            if cat not in df_category_result.columns:
                df_category_result[cat] = 0
        
        df_category_result['all_categories'] = df_category_result[categories].sum(axis=1)
        
        # CSV speichern
        csv_file = os.path.join(self.csv_dir, "category_allele_per_accession.csv")
        df_category_result.to_csv(csv_file)
        
        with open(self.report_file, "a") as f:
            f.write(f"Kategorie-Allelfrequenzen als DataFrame erstellt und in {csv_file} gespeichert\n")
            f.write(f"Verfügbare Kategorien: {list(df_category_result.columns)}\n\n")
        
        # Kategorie-Verteilungsplot erstellen
        self._plot_category_frequency_histograms(df_category_result)
        
        # Kategorie-Heatmap erstellen
        self._plot_top_accessions_category_heatmap(df_category_result)
        
        return df_category_result

    def create_allele_frequency_plots(self, df_allele_freq):
        """
        Erstellt verschiedene Visualisierungen für Allelfrequenzen pro Accession.
        
        Parameters
        ----------
        df_allele_freq : pd.DataFrame
            DataFrame mit Allelfrequenzen (Zeilen=Accessions, Spalten=Effekt-Typen)
        """
        with open(self.report_file, "a") as f:
            f.write(f"Erstelle Allelfrequenz-Visualisierungen...\n")
        
        # 1. Verteilungshistogramme für jeden Effekt-Typ
        self._plot_effect_type_distributions(df_allele_freq)
        
        # 2. Heatmap der Top-Accessions vs Top-Effekte
        self._plot_top_accessions_heatmap(df_allele_freq)
        
        # 3. Zusammenfassungsstatistiken
        self._plot_allele_frequency_summary(df_allele_freq)
        
        with open(self.report_file, "a") as f:
            f.write(f"Allelfrequenz-Visualisierungen erstellt\n\n")

    def _plot_effect_type_distributions(self, df_allele_freq):
        """
        Erstellt Histogramme für die Verteilung jedes Effekt-Typs über alle Accessions.
        """
        # Effekt-Spalten (alle außer 'all_effect')
        effect_cols = [col for col in df_allele_freq.columns if col != 'all_effect']
        
        # Top 12 häufigste Effekte für bessere Darstellung
        effect_sums = df_allele_freq[effect_cols].sum().sort_values(ascending=False)
        top_effects = effect_sums.head(12).index.tolist()
        
        # 4x3 Subplot-Layout
        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, effect in enumerate(top_effects):
            if i < len(axes):
                ax = axes[i]
                data = df_allele_freq[effect]
                
                # NaN-Werte und infinite Werte entfernen
                data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Histogramm mit log-Skala wenn nötig
                if len(data_clean) > 0 and data_clean.max() > 100:
                    bins = np.logspace(0, np.log10(data_clean.max() + 1), 30)
                    ax.hist(data_clean[data_clean > 0], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_xscale('log')
                else:
                    if len(data_clean) > 0:
                        ax.hist(data_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                
                ax.set_title(f'{effect}\n(Σ={effect_sums[effect]:.0f})', fontsize=10)
                ax.set_xlabel('Anzahl SNPs pro Accession')
                ax.set_ylabel('Anzahl Accessions')
                ax.grid(True, alpha=0.3)
        
        # Leere Subplots ausblenden
        for i in range(len(top_effects), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Verteilung der SNP-Effekte über alle Accessions - {self.dataset_name}', 
                     fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Speichern
        plot_path = os.path.join(self.plots_dir, "allele_frequency_distributions.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.add_plot_to_pdf(fig)
        plt.close()

    def _plot_category_frequency_histograms(self, df_category_freq):
        """
        Erstellt Histogramme für die Verteilung der 6 Effekt-Kategorien über alle Accessions.
        Ähnlich zu _plot_effect_type_distributions, aber für die zusammengefassten Kategorien.
        """
        # Die 6 Hauptkategorien
        categories = ['Protein-changing', 'Synonymous', 'Splice-related', 'UTR', 'Intron', 'Non-coding']
        
        # Nur verfügbare Kategorien verwenden
        available_categories = [cat for cat in categories if cat in df_category_freq.columns]
        
        # 2x3 Subplot-Layout für die 6 Kategorien
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Statistiken für jede Kategorie berechnen
        category_sums = df_category_freq[available_categories].sum().sort_values(ascending=False)
        
        for i, category in enumerate(available_categories):
            if i < len(axes):
                ax = axes[i]
                data = df_category_freq[category]
                
                # NaN-Werte und infinite Werte entfernen
                data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Histogramm mit log-Skala wenn nötig
                if len(data_clean) > 0 and data_clean.max() > 100:
                    bins = np.logspace(0, np.log10(data_clean.max() + 1), 30)
                    ax.hist(data_clean[data_clean > 0], bins=bins, alpha=0.7, color=plt.cm.Set3(i), edgecolor='black')
                    ax.set_xscale('log')
                else:
                    if len(data_clean) > 0:
                        ax.hist(data_clean, bins=30, alpha=0.7, color=plt.cm.Set3(i), edgecolor='black')
                
                ax.set_title(f'{category}\n(Σ={category_sums[category]:.0f})', fontsize=12, fontweight='bold')
                ax.set_xlabel('Anzahl SNPs pro Accession')
                ax.set_ylabel('Anzahl Accessions')
                ax.grid(True, alpha=0.3)
        
        # Leere Subplots ausblenden (falls weniger als 6 Kategorien)
        for i in range(len(available_categories), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Verteilung der SNP-Effekt-Kategorien über alle Accessions - {self.dataset_name}', 
                     fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Speichern
        plot_path = os.path.join(self.plots_dir, "category_frequency_distributions.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.add_plot_to_pdf(fig)
        
        with open(self.report_file, "a") as f:
            f.write(f"Kategorie-Verteilungsplot erstellt: {plot_path}\n")
        
        plt.close()

    def _plot_top_accessions_category_heatmap(self, df_category_freq, top_n=50):
        """
        Erstellt eine Heatmap der Top-N Accessions vs 6 Hauptkategorien.
        Ähnlich zu _plot_top_accessions_heatmap, aber für die zusammengefassten Kategorien.
        """
        # Die 6 Hauptkategorien
        categories = ['Protein-changing', 'Synonymous', 'Splice-related', 'UTR', 'Intron', 'Non-coding']
        
        # Nur verfügbare Kategorien verwenden
        available_categories = [cat for cat in categories if cat in df_category_freq.columns]
        
        # Top Accessions (nach Gesamtzahl SNPs in allen Kategorien)
        if 'all_categories' in df_category_freq.columns:
            top_accessions = df_category_freq.nlargest(top_n, 'all_categories').index
        else:
            # Fallback: Summe aller Kategorien berechnen
            category_sum = df_category_freq[available_categories].sum(axis=1)
            top_accessions = category_sum.nlargest(top_n).index
        
        # Daten für Heatmap (Top Accessions vs alle 6 Kategorien)
        heatmap_data = df_category_freq.loc[top_accessions, available_categories]
        
        # Achsen vertauschen: Kategorien auf Y-Achse, Accessions auf X-Achse
        heatmap_data_transposed = heatmap_data.T  # Transponieren
        
        # Heatmap erstellen
        fig, ax = plt.subplots(figsize=(20, 8))  # Breiter statt höher wegen 50 Accessions auf X-Achse
        
        # Log-Transformation für bessere Visualisierung
        heatmap_data_log = np.log1p(heatmap_data_transposed)
        
        # NaN und infinite Werte behandeln
        heatmap_data_log = heatmap_data_log.replace([np.inf, -np.inf], np.nan)
        
        sns.heatmap(heatmap_data_log, 
                   annot=False,  # Keine Zahlen bei 50 Accessions (zu viel)
                   cmap='viridis', 
                   ax=ax,
                   cbar_kws={'label': 'log(SNP count + 1)'})
        
        ax.set_title(f'6 SNP-Effekt-Kategorien vs Top {top_n} Accessions - {self.dataset_name}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Accessions', fontsize=12)
        ax.set_ylabel('SNP-Effekt-Kategorien', fontsize=12)
        
        # X-Achse-Labels rotieren für bessere Lesbarkeit (50 Accessions)
        plt.xticks(rotation=90, ha='center', fontsize=6)  # Kleine Schrift für 50 Accessions
        plt.yticks(rotation=0, fontsize=10)  # Normale Schrift für 6 Kategorien
        
        plt.tight_layout()
        
        # Speichern
        plot_path = os.path.join(self.plots_dir, f"top{top_n}_accessions_category_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.add_plot_to_pdf(fig)
        
        with open(self.report_file, "a") as f:
            f.write(f"Top-{top_n}-Accessions Kategorie-Heatmap erstellt: {plot_path}\n")
            f.write(f"Heatmap zeigt {len(top_accessions)} Accessions vs {len(available_categories)} Kategorien\n\n")
        
        plt.close()

    def _plot_top_accessions_heatmap(self, df_allele_freq, top_n=50):
        """
        Erstellt eine Heatmap der Top-N Accessions vs Top-Effekte.
        """
        # Top Accessions (nach Gesamtzahl SNPs)
        top_accessions = df_allele_freq.nlargest(top_n, 'all_effect').index
        
        # Top Effekte (nach Gesamtsumme)
        effect_cols = [col for col in df_allele_freq.columns if col != 'all_effect']
        effect_sums = df_allele_freq[effect_cols].sum().sort_values(ascending=False)
        top_effects = effect_sums.head(15).index.tolist()
        
        # Daten für Heatmap
        heatmap_data = df_allele_freq.loc[top_accessions, top_effects]
        
        # Heatmap erstellen
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Log-Transformation für bessere Visualisierung
        heatmap_data_log = np.log1p(heatmap_data)
        
        # NaN und infinite Werte behandeln
        heatmap_data_log = heatmap_data_log.replace([np.inf, -np.inf], np.nan)
        
        sns.heatmap(heatmap_data_log, 
                   annot=False, 
                   cmap='viridis', 
                   ax=ax,
                   cbar_kws={'label': 'log(SNP count + 1)'})
        
        ax.set_title(f'Top {top_n} Accessions vs Top 15 SNP-Effekte - {self.dataset_name}', 
                     fontsize=14)
        ax.set_xlabel('SNP-Effekt-Typen')
        ax.set_ylabel('Accessions')
        
        # X-Achse-Labels rotieren
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, fontsize=6)
        
        plt.tight_layout()
        
        # Speichern
        plot_path = os.path.join(self.plots_dir, f"allele_frequency_heatmap_top{top_n}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.add_plot_to_pdf(fig)
        plt.close()

    def _plot_allele_frequency_summary(self, df_allele_freq):
        """
        Erstellt Zusammenfassungsstatistiken als Barplots.
        """
        effect_cols = [col for col in df_allele_freq.columns if col != 'all_effect']
        
        # Statistiken berechnen
        stats = []
        for effect in effect_cols:
            data = df_allele_freq[effect]
            stats.append({
                'effect': effect,
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'max': data.max(),
                'nonzero_count': (data > 0).sum(),
                'total_snps': data.sum()
            })
        
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values('total_snps', ascending=False).head(15)
        
        # 2x2 Subplot für verschiedene Statistiken
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Gesamtzahl SNPs pro Effekt
        axes[0,0].barh(stats_df['effect'], stats_df['total_snps'], color='lightcoral')
        axes[0,0].set_title('Gesamtanzahl SNPs pro Effekt-Typ')
        axes[0,0].set_xlabel('Anzahl SNPs')
        axes[0,0].set_xscale('log')
        
        # 2. Anzahl Accessions mit diesem Effekt
        axes[0,1].barh(stats_df['effect'], stats_df['nonzero_count'], color='lightgreen')
        axes[0,1].set_title('Anzahl Accessions mit mindestens 1 SNP')
        axes[0,1].set_xlabel('Anzahl Accessions')
        
        
        
        # 3. Durchschnittliche SNPs pro Accession
        axes[1,0].barh(stats_df['effect'], stats_df['mean'], color='lightblue')
        axes[1,0].set_title('Durchschnittliche SNPs pro Accession')
        axes[1,0].set_xlabel('Durchschnitt')
       
       
        
        # 4. Maximale SNPs in einer Accession
        axes[1,1].barh(stats_df['effect'], stats_df['max'], color='yellow')
        axes[1,1].set_title('Maximum SNPs in einer Accession')
        axes[1,1].set_xlabel('Maximum')
        axes[1,1].set_xscale('log')
        
        # Alle Y-Achsen invertieren für bessere Lesbarkeit
        for ax in axes.flatten():
            ax.invert_yaxis()
            ax.tick_params(axis='y', labelsize=8)
        
        plt.suptitle(f'Allelfrequenz-Zusammenfassung - {self.dataset_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Speichern
        plot_path = os.path.join(self.plots_dir, "allele_frequency_summary.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.add_plot_to_pdf(fig)
        plt.close()

    def analyze_aa_substitutions(self, top_n=10, plot=True):
        """
        Analysiert Aminosäure-Substitutionen pro Isoform.
        Unterstützt Multi-Allel-SNPs (mehrere alt_aa pro ref_aa).
        """
        # Isoformen als Zahlen sortieren, nicht als Strings
        isoform_numbers = {int(t.split('.')[-1]) for t in self.df['transcript_id'] if pd.notna(t) and '.' in t}
        isoforms = [f".{iso}" for iso in sorted(isoform_numbers)]
        isoforms.append("all")

        results = {}

        for iso in isoforms:
            # Isoform-spezifische Ordner erstellen (außer für "all")
            if iso != "all":
                isoform_name = iso.replace('.', '')
                isoform_plots_dir = os.path.join(self.plots_dir, f"isoform_{isoform_name}")
                isoform_csv_dir = os.path.join(self.csv_dir, f"isoform_{isoform_name}")
                os.makedirs(isoform_plots_dir, exist_ok=True)
                os.makedirs(isoform_csv_dir, exist_ok=True)
            else:
                # Für "all" verwende die Standard-Ordner
                isoform_plots_dir = self.plots_dir
                isoform_csv_dir = self.csv_dir
            
            if iso == "all":
                df_iso = self.df.copy()
            else:
                df_iso = self.df[self.df["transcript_id"].str.endswith(iso)].copy()

            if df_iso.empty:
                with open(self.report_file, "a") as f:
                    f.write(f"Isoform {iso}: keine Daten\n")
                continue

            aa_df = df_iso[df_iso['snp_aa_seq'].notna()].copy()
            aa_df = aa_df[aa_df['snp_aa_seq'].str.contains("->", regex=False)]

            if aa_df.empty:
                with open(self.report_file, "a") as f:
                    f.write(f"Isoform {iso}: keine gültigen Aminosäure-SNPs\n")
                continue

            # Einfache Substitutions-Verarbeitung (keine Multi-Allel-Expansion nötig)
            substitutions = []
            for idx, row in aa_df.iterrows():
                ref_aa = row['snp_aa_seq'].split(" -> ")[0]
                alt_aa = row['snp_aa_seq'].split(" -> ")[1].strip()
                
                # Alle delins als eine Kategorie behandeln
                if alt_aa.startswith('delins'):
                    alt_aa = 'delins'
                    
                substitutions.append({
                    'ref_aa': ref_aa,
                    'alt_aa': alt_aa
                })

            # DataFrame erstellen
            substitutions_df = pd.DataFrame(substitutions)
            
            # Debug-Informationen 
            # with open(self.report_file, "a") as f:
            #     f.write(f"Debug - Multi-Allel-Verarbeitung für {iso}:\n")
            #     f.write(f"  - Originale SNPs: {len(aa_df)}\n")
            #     f.write(f"  - Erweiterte Substitutionen: {len(expanded_df)}\n")
            #     f.write(f"  - Eindeutige ref_aa: {expanded_df['ref_aa'].nunique()}\n")
            #     f.write(f"  - Eindeutige alt_aa: {expanded_df['alt_aa'].nunique()}\n")
            #     f.write(f"  - Multi-Allel-Substitutionen: {expanded_df['is_multi_allele'].sum()}\n")
            #     
            #     # Beispiel für Multi-Allel-Erweiterung zeigen
            #     multi_allele_examples = expanded_df[expanded_df['is_multi_allele'] == True].head(3)
            #     if not multi_allele_examples.empty:
            #         f.write(f"  - Beispiel Multi-Allel-Erweiterung:\n")
            #         for _, row in multi_allele_examples.iterrows():
            #             f.write(f"    {row['ref_aa']} -> {row['alt_aa']}\n")
            #     f.write("\n")
            
            if substitutions_df.empty:
                with open(self.report_file, "a") as f:
                    f.write(f"Isoform {iso}: keine gültigen Substitutionen\n")
                continue

            # Substitutionsmatrix
            matrix = pd.crosstab(substitutions_df['ref_aa'], substitutions_df['alt_aa'])
            out_matrix_csv = os.path.join(isoform_csv_dir, f"aa_substitution_matrix.csv")
            matrix.to_csv(out_matrix_csv)
            
            # Matrix-Informationen
            with open(self.report_file, "a") as f:
                f.write(f"Aminosäure-Substitutionen für {iso}:\n")
                f.write(f"  - Matrix-Größe: {matrix.shape[0]}x{matrix.shape[1]}\n")
                f.write(f"  - Gesamte Substitutionen: {matrix.sum().sum()}\n\n")

            if plot:
                fig_matrix = plt.figure(figsize=(14, 12))
                plt.imshow(matrix, cmap="viridis")
                plt.colorbar(label="Häufigkeit")
                plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha='right')
                plt.yticks(range(len(matrix.index)), matrix.index)
                plt.title(f"Aminosäure-Substitutionen ({iso}) - {self.dataset_name}")
                plt.xlabel("Alt AA")
                plt.ylabel("Ref AA")
                plt.tight_layout()

                # PNG speichern
                out_matrix_png = os.path.join(isoform_plots_dir, f"aa_substitution_matrix.png")
                plt.savefig(out_matrix_png, bbox_inches='tight', dpi=300)
                
                # Isoform-spezifische Plots NICHT zur PDF hinzufügen
                if iso == "all":
                    # Nur "all" Plots zur PDF hinzufügen
                    self.add_plot_to_pdf(fig_matrix)
                plt.close()

            # Top-N Substitutionen
            counts = (
                substitutions_df.groupby(["ref_aa", "alt_aa"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )

            out_counts_csv = os.path.join(isoform_csv_dir, f"aa_substitution_counts.csv")
            counts.to_csv(out_counts_csv, index=False)

            if plot:
                top_counts = counts.head(top_n)
                fig_counts = plt.figure(figsize=(12, 8))
                plt.barh(
                    [f"{r}→{a}" for r, a in zip(top_counts["ref_aa"], top_counts["alt_aa"])],
                    top_counts["count"],
                    color="steelblue"
                )
                plt.xlabel("Häufigkeit")
                plt.ylabel("Substitution")
                plt.title(f"Top {top_n} Aminosäure-Substitutionen ({iso}) - {self.dataset_name}")
                plt.gca().invert_yaxis()
                plt.tight_layout()

                # PNG speichern
                out_counts_png = os.path.join(isoform_plots_dir, f"top{top_n}_aa_substitutions.png")
                plt.savefig(out_counts_png, bbox_inches='tight', dpi=300)
                
                # Isoform-spezifische Plots NICHT zur PDF hinzufügen
                if iso == "all":
                    # Nur "all" Plots zur PDF hinzufügen
                    self.add_plot_to_pdf(fig_counts)
                plt.close()

            results[iso] = {
                "matrix": matrix, 
                "counts": counts
            }

        return results



    def run_complete_analysis(self, data_path=".", data_pattern="*.csv", data_sep=";"):
        """
        Führt die komplette SNP-Analyse-Pipeline aus.
        """
        with open(self.report_file, "a") as f:
            f.write(f"Starte komplette SNP-Analyse für {self.dataset_name}\n")
            f.write("=" * 60 + "\n\n")
        
        # 1. Daten laden
        with open(self.report_file, "a") as f:
            f.write("Lade Daten...\n")
        self.load_data(path=data_path, pattern=data_pattern, sep=data_sep)
        
        # 2. Grundlegende Statistiken
        with open(self.report_file, "a") as f:
            f.write("Erstelle Grundstatistiken...\n")
        effect_counts = self.count_effects(save_path=os.path.join(self.csv_dir, "effect_counts.csv"))
        self.save_effect_plot(effect_counts)
        
        # 3. Accession-Analyse
        with open(self.report_file, "a") as f:
            f.write("Analysiere Accessions...\n")
        self.count_unique_accessions()
        self.count_accessions_per_isoform()
        self.count_genes_per_isoform()
        
        # 4. Isoform-spezifische Analysen
        with open(self.report_file, "a") as f:
            f.write("Erstelle Isoform-subsets...\n")
        self.save_isoform_subsets()
        
        # 5. SNP-Effekt-Analysen
        with open(self.report_file, "a") as f:
            f.write("Analysiere SNP-Effekte...\n")
        self.snp_effects_per_isoform()
        self.snp_effects_all_transcripts()
        
        # 6. Gen-spezifische Analysen
        with open(self.report_file, "a") as f:
            f.write("Analysiere SNPs pro Gen...\n")
        self.snp_counts_per_gene_analysis()
        
        # 7. Allelfrequenz-Analyse
        with open(self.report_file, "a") as f:
            f.write("Berechne Allelfrequenzen...\n")
        self.calculate_allele_frequencies_per_accession()
        
        # 8. Aminosäure-Substitutions-Analyse
        with open(self.report_file, "a") as f:
            f.write("Analysiere Aminosäure-Substitutionen...\n")
        self.analyze_aa_substitutions()
        
        # 9.  SNP-Dichte-Analyse
        with open(self.report_file, "a") as f:
            f.write("Analysiere SNP-Dichten...\n")
        snp_density_results = self.analyse_snp_densities_optimized(
            out_dir=os.path.join(self.results_dir, "snp_densities")
        )
        
        # 10. Finale PDF erstellen
        with open(self.report_file, "a") as f:
            f.write("Erstelle finale PDF mit allen Plots...\n")
        self.create_final_pdf()
        
        with open(self.report_file, "a") as f:
            f.write(f"Komplette Analyse für {self.dataset_name} abgeschlossen!\n")
            f.write(f"Ergebnisse gespeichert in: {self.results_dir}\n")
            f.write("=" * 60 + "\n")

    def classify_effect(self, effect: str) -> str:
        """
        Klassifiziert eine Mutation in eine Hauptkategorie:
        - Protein-changing
        - Synonymous
        - Splice-related
        - UTR
        - Intron
        - Non-coding
        """
        # Lowercase für Sicherheit
        e = effect.lower()

        # Synonymous zuerst (weil "synonymous" + "splice" auch vorkommen kann)
        if "synonymous" in e or "retained" in e:
            return "Synonymous"

        # Splice-related
        if "splice" in e:
            return "Splice-related"

        # Protein-changing
        if any(keyword in e for keyword in [
            "missense", "frameshift", "stop", "start", "inframe", "initiator_codon"
        ]):
            return "Protein-changing"

        # UTR
        if "utr" in e:
            return "UTR"

        # Intron (nur wenn wirklich reines intron oder Kombination ohne splice)
        if "intron" in e:
            return "Intron"

        # Non-coding
        if "non_coding" in e:
            return "Non-coding"

        # Fallback
        return "Other"

    def get_normalization_lengths_dict(self) -> dict:
        """
        Erstellt ein Dictionary mit Normalisierungslängen für jede Effekt-Kategorie.
        """
        normalization_lengths = {
            "Protein-changing": "cds_len",                   # CDS-Länge für Protein-Effekte
            "Synonymous": "cds_len",                         # CDS-Länge für synonyme Mutationen
            "Splice-related": "cdna_len",                    # cDNA-Länge für Splice-Effekte
            "UTR": "cdna_len - cds_len",                     # UTR-Länge (gesamte cDNA minus CDS)
            "Intron": "cdna_len",                            # cDNA-Länge für Intron-Effekte
            "Non-coding": "cdna_len"                         # cDNA-Länge für Non-coding Effekte
        }
        
        return normalization_lengths

    def analyse_snp_densities_optimized(self, out_dir: str = "snp_densities"):
        """
        Analysiert SNP-Dichten mit Normalisierung basierend auf Effekt-Kategorien.
        Verwendet classify_effect und get_normalization_lengths_dict für die Berechnung.
        
        Parameter
        ---------
        out_dir : str
            Ordner, in dem alle Dateien gespeichert werden.
        """
        if self.df is None:
            raise ValueError("Daten müssen zuerst geladen werden. Rufen Sie load_data() auf.")
        
        os.makedirs(out_dir, exist_ok=True)
        
        # Effekt-Kategorien für alle SNPs zuweisen
        self.df['effect_category'] = self.df['eff'].apply(self.classify_effect)
        
        # Normalisierungslängen-Dictionary erhalten
        norm_lengths_dict = self.get_normalization_lengths_dict()
        
        # Längen-Spalten als numerisch konvertieren
        for col in ['aa_len', 'cds_len', 'cdna_len']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df.loc[self.df[col] <= 0, col] = np.nan
        
        # Non-coding Länge berechnen (cdna_len - cds_len)
        self.df['noncoding_len'] = np.nan
        mask_cdna_ok = self.df['cdna_len'].notna() & self.df['cds_len'].notna()
        self.df.loc[mask_cdna_ok, 'noncoding_len'] = (
            self.df.loc[mask_cdna_ok, 'cdna_len'] - self.df.loc[mask_cdna_ok, 'cds_len']
        )
        self.df.loc[self.df['noncoding_len'] <= 0, 'noncoding_len'] = np.nan
        
        # Isoformen extrahieren - numerisch sortieren
        isoform_strings = set(self.df['transcript_id'].str.extract(r'(\.\d+)$')[0].dropna())
        isoform_numbers = [int(iso.replace('.', '')) for iso in isoform_strings]
        isoforms = [f".{num}" for num in sorted(isoform_numbers)]
        isoforms.append("all")
        
        results = {}
        all_summaries = []
        all_effects = []
        report_lines = []
        
        for isoform in isoforms:
            report_lines.append('='*80)
            report_lines.append(f"*** Analyse für Isoform {isoform} ***")
            
            # Isoform-spezifischen Ordner erstellen
            isoform_dir = os.path.join(out_dir, f"isoform_{isoform.replace('.', '')}" if isoform != "all" else "all_isoforms")
            os.makedirs(isoform_dir, exist_ok=True)
            
            df_iso = self.df.copy() if isoform == "all" else self.df[self.df['transcript_id'].str.endswith(isoform)].copy()
            report_lines.append(f"Isoform {isoform}: {len(df_iso)} Varianten insgesamt")
            
            if df_iso.empty:
                report_lines.append(f"Keine Daten für Isoform {isoform}")
                continue
            
            # Unique Varianten pro Gen
            unique_vars = df_iso.dropna(subset=['gene_id', 'snp_id_c']).drop_duplicates(subset=['snp_id_c', 'gene_id']).copy()
            report_lines.append(f"Unique Varianten: {unique_vars.shape[0]}")
            
            # Effekt-Zählungen pro Gen
            total_unique_per_gene = unique_vars.groupby('gene_id')['snp_id_c'].nunique().reset_index()
            total_unique_per_gene.columns = ['gene_id', 'unique_snps']
            
            unique_effect_counts = unique_vars.groupby(['gene_id', 'eff'])['snp_id_c'].nunique().unstack(fill_value=0).reset_index()
            eff_cols = [c for c in unique_effect_counts.columns if c != 'gene_id']
            counts = total_unique_per_gene.merge(unique_effect_counts, on='gene_id', how='left').fillna(0)
            
            # Längen pro Gen (Maximum pro Gen)
            lengths = df_iso.groupby('gene_id').agg({
                'aa_len': 'max', 
                'cds_len': 'max', 
                'cdna_len': 'max',
                'noncoding_len': 'max'
            }).reset_index()
            
            counts = counts.merge(lengths, on='gene_id', how='left')
            report_lines.append(f"Anzahl Gene: {len(counts)}")
            
            #  SNP-Dichte berechnen: für jedes SNP nur die passende Länge verwenden
            counts['optimized_snp_density'] = counts.apply(
                lambda row: self._calculate_optimized_density(row, unique_vars, norm_lengths_dict), 
                axis=1
            )
            
            # Effekt-spezifische Dichten berechnen
            effect_densities = []
            for effect in eff_cols:
                if effect in unique_vars['eff'].values:
                    effect_density = self._calculate_effect_density(
                        effect, counts, unique_vars, norm_lengths_dict
                    )
                    effect_densities.append(effect_density)
            
            effect_densities_df = pd.DataFrame(effect_densities)
            if not effect_densities_df.empty:
                effect_densities_df = effect_densities_df.sort_values('density', ascending=False)
                effect_densities_df['isoform'] = isoform
                all_effects.append(effect_densities_df)
                
                report_lines.append("Top Effekte normalisiert nach der Dichte:")
                report_lines.append(effect_densities_df.head(10).to_string(index=False))
            
            # Summary erstellen
            summary_df = pd.DataFrame({
                'metric': ['optimized_density'],
                'avg_density': [counts['optimized_snp_density'].mean()],
                'isoform': [isoform]
            })
            all_summaries.append(summary_df)
            report_lines.append("Durchschnittliche SNP-Dichte pro Gen:")
            report_lines.append(f"{counts['optimized_snp_density'].mean():.6f}")
            
            # Dateien in isoform-spezifischen Ordner speichern
            counts.to_csv(os.path.join(isoform_dir, f'gene_optimized_snp_densities.csv'), index=False)
            summary_df.to_csv(os.path.join(isoform_dir, f'optimized_density_summary.csv'), index=False)
            if not effect_densities_df.empty:
                effect_densities_df.to_csv(os.path.join(isoform_dir, f'effect_densities.csv'), index=False)
            
            results[isoform] = {
                'counts': counts, 
                'summary': summary_df, 
                'effect_densities': effect_densities_df if not effect_densities_df.empty else pd.DataFrame()
            }
        
        # Plots erstellen
        self._create_snp_density_plots(all_summaries, all_effects, out_dir)
        
        # Report speichern
        with open(os.path.join(self.reports_dir, 'snp_density_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))
        
        with open(self.report_file, "a") as f:
            f.write(f" SNP-Dichte-Analyse abgeschlossen. Ergebnisse in: {out_dir}\n\n")
        
        return results

    def _calculate_optimized_density(self, row, unique_vars, norm_lengths_dict):
        """
        Berechnet  SNP-Dichte für ein Gen basierend auf Effekt-Kategorien.
        """
        gene_id = row['gene_id']
        gene_vars = unique_vars[unique_vars['gene_id'] == gene_id]
        
        if gene_vars.empty:
            return np.nan
        
        total_density = 0
        total_snps = 0
        
        for _, var in gene_vars.iterrows():
            effect_category = self.classify_effect(var['eff'])
            length_type = norm_lengths_dict.get(effect_category, 'cds_len')
            
            # Passende Länge für diesen Effekt-Typ
            if length_type == 'aa_len' and pd.notna(row['aa_len']) and row['aa_len'] > 0:
                density = 1 / row['aa_len']
            elif length_type == 'cds_len' and pd.notna(row['cds_len']) and row['cds_len'] > 0:
                density = 1 / row['cds_len']
            elif length_type == 'cdna_len' and pd.notna(row['cdna_len']) and row['cdna_len'] > 0:
                density = 1 / row['cdna_len']
            elif length_type == 'cdna_len - cds_len' and pd.notna(row['noncoding_len']) and row['noncoding_len'] > 0:
                density = 1 / row['noncoding_len']
            else:
                # Fallback auf CDS-Länge
                density = 1 / row['cds_len'] if pd.notna(row['cds_len']) and row['cds_len'] > 0 else np.nan
            
            if not np.isnan(density):
                total_density += density
                total_snps += 1
        
        return total_density / total_snps if total_snps > 0 else np.nan

    def _calculate_effect_density(self, effect, counts, unique_vars, norm_lengths_dict):
        """
        Berechnet Dichte für einen spezifischen Effekt.
        """
        effect_category = self.classify_effect(effect)
        length_type = norm_lengths_dict.get(effect_category, 'cds_len')
        
        # SNPs dieses Effekts zählen
        effect_snps = counts[effect].sum()
        
        # Passende Gesamtlänge für diesen Effekt-Typ
        if length_type == 'aa_len':
            total_length = counts['aa_len'].sum()
        elif length_type == 'cds_len':
            total_length = counts['cds_len'].sum()
        elif length_type == 'cdna_len':
            total_length = counts['cdna_len'].sum()
        elif length_type == 'cdna_len - cds_len':
            total_length = counts['noncoding_len'].sum()
        else:
            total_length = counts['cds_len'].sum()
        
        density = effect_snps / total_length if total_length > 0 else np.nan
        
        return {
            'effect': effect,
            'effect_category': effect_category,
            'length_type': length_type,
            'density': density
        }

    def _create_snp_density_plots(self, all_summaries, all_effects, out_dir):
        """
        Erstellt Plots für die SNP-Dichte-Analyse.
        """
        if all_summaries:
            all_summaries_df = pd.concat(all_summaries, ignore_index=True)
            
            # NaN und infinite Werte entfernen
            all_summaries_clean = all_summaries_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=all_summaries_clean, x='isoform', y='avg_density')
            plt.ylabel('Durchschnittliche SNP-Dichte')
            plt.title('Vergleich der SNP-Dichten pro Isoform')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(out_dir, 'optimized_snp_density_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Zur PDF hinzufügen
            self.add_plot_to_pdf(plt.gcf())
            plt.close()
        
        if all_effects:
            all_effects_df = pd.concat(all_effects, ignore_index=True)
            
            # NaN und infinite Werte entfernen
            all_effects_clean = all_effects_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Plot 1: Top 5 Effekte (vor Gruppierung)
            top_eff = all_effects_clean.groupby('effect')['density'].mean().sort_values(ascending=False).head(5).index
            plt.figure(figsize=(12, 6))
            sns.barplot(data=all_effects_clean[all_effects_clean['effect'].isin(top_eff)], 
                       x='effect', y='density', hue='isoform')
            plt.ylabel('SNP-Dichte')
            plt.title('Top-5 Effekte nach Dichte')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plot_path = os.path.join(out_dir, 'top5_effects_optimized.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Zur PDF hinzufügen
            self.add_plot_to_pdf(plt.gcf())
            plt.close()
            
            # Plot 2: Alle 6 Effekt-Kategorien (nach Gruppierung)
            category_densities = []
            for isoform in all_effects_clean['isoform'].unique():
                isoform_data = all_effects_clean[all_effects_clean['isoform'] == isoform]
                
                # Effekte nach Kategorien gruppieren
                for category in ['Protein-changing', 'Synonymous', 'Splice-related', 'UTR', 'Intron', 'Non-coding']:
                    category_effects = isoform_data[isoform_data['effect_category'] == category]
                    if not category_effects.empty:
                        avg_density = category_effects['density'].mean()
                        category_densities.append({
                            'isoform': isoform,
                            'effect_category': category,
                            'avg_density': avg_density
                        })
            
            if category_densities:
                category_df = pd.DataFrame(category_densities)
                
                plt.figure(figsize=(12, 6))
                sns.barplot(data=category_df, x='effect_category', y='avg_density', hue='isoform')
                plt.ylabel('Durchschnittliche SNP-Dichte')
                plt.title('SNP-Dichte nach Effekt-Kategorien (alle 6 Kategorien)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                plot_path = os.path.join(out_dir, 'effect_categories_density.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                # Zur PDF hinzufügen
                self.add_plot_to_pdf(plt.gcf())
                plt.close()
                
                # Kategorie-Daten auch als CSV speichern
                category_df.to_csv(os.path.join(out_dir, 'effect_categories_density.csv'), index=False)
                
                # Zusätzliche detaillierte Plots für die 6 Kategorien erstellen
                self._create_detailed_category_plots(category_df, all_effects_df, out_dir)

    def _create_detailed_category_plots(self, category_df, all_effects_df, out_dir):
        """
        Erstellt detaillierte Visualisierungen für die 6 Effekt-Kategorien.
        """
        # 1. Heatmap: Kategorien vs Isoformen
        self._plot_category_isoform_heatmap(category_df, out_dir)
        
        # 2. Einzelne Plots für jede Kategorie
        self._plot_individual_categories(category_df, out_dir)
        
        # 3. Verteilungsanalyse pro Kategorie
        self._plot_category_distributions(all_effects_df, out_dir)

    def _plot_category_isoform_heatmap(self, category_df, out_dir):
        """
        Erstellt eine Heatmap der 6 Kategorien vs Isoformen.
        """
        # Pivot für Heatmap
        heatmap_data = category_df.pivot(index='effect_category', columns='isoform', values='avg_density')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # NaN und infinite Werte behandeln
        heatmap_data_clean = heatmap_data.replace([np.inf, -np.inf], np.nan)
        
        # Direkte Darstellung ohne Log-Transformation
        sns.heatmap(heatmap_data_clean, 
                   annot=False, 
                   cmap='viridis', 
                   ax=ax,
                   cbar_kws={'label': 'SNP-Dichte'})
        
        ax.set_title(f'SNP-Dichte Heatmap: 6 Kategorien × Isoformen - {self.dataset_name}')
        ax.set_xlabel('Isoformen')
        ax.set_ylabel('Effekt-Kategorien')
        
        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'category_isoform_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Zur PDF hinzufügen
        self.add_plot_to_pdf(fig)
        plt.close()

    def _plot_individual_categories(self, category_df, out_dir):
        """
        Erstellt individuelle Barplots für jede der 6 Kategorien.
        """
        categories = ['Protein-changing', 'Synonymous', 'Splice-related', 'UTR', 'Intron', 'Non-coding']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, category in enumerate(categories):
            ax = axes[i]
            cat_data = category_df[category_df['effect_category'] == category]
            
            if not cat_data.empty:
                # Sortiere Isoformen numerisch für konsistente Darstellung
                cat_data['isoform_num'] = cat_data['isoform'].str.replace('.', '').str.replace('all', '999').astype(int)
                cat_data_sorted = cat_data.sort_values('isoform_num')
                
                bars = ax.bar(cat_data_sorted['isoform'], cat_data_sorted['avg_density'], 
                             color=plt.cm.Set3(i), alpha=0.8)
                
                
                ax.set_title(f'{category}', fontweight='bold')
                ax.set_ylabel('SNP-Dichte')
                ax.set_xlabel('Isoform')
                ax.tick_params(axis='x', rotation=45)
                
                # Y-Achse formatieren - normale Dezimaldarstellung
                ax.ticklabel_format(style='plain', axis='y')
            else:
                ax.text(0.5, 0.5, f'Keine Daten für\n{category}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{category}', fontweight='bold')
        
        plt.suptitle(f'SNP-Dichten pro Kategorie und Isoform - {self.dataset_name}', 
                     fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join(out_dir, 'individual_categories_density.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Zur PDF hinzufügen
        self.add_plot_to_pdf(fig)
        plt.close()

    def _plot_category_distributions(self, all_effects_df, out_dir):
        """
        Erstellt Verteilungsanalyse für jede Kategorie (Box-Plots).
        """
        fig, (ax, ax_stats) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3, 1]})
        
        # Daten für Box-Plot vorbereiten
        categories = ['Protein-changing', 'Synonymous', 'Splice-related', 'UTR', 'Intron', 'Non-coding']
        
        # Nur Kategorien mit Daten verwenden
        available_categories = [cat for cat in categories if cat in all_effects_df['effect_category'].values]
        
        if available_categories:
            # Box-Plot im linken Panel
            sns.boxplot(data=all_effects_df, x='effect_category', y='density', ax=ax)
            
            # Log-Skala für Y-Achse
            #ax.set_yscale('log')
            ax.set_title(f'Verteilung der SNP-Dichten pro Kategorie - {self.dataset_name}')
            ax.set_xlabel('Effekt-Kategorien')
            ax.set_ylabel('SNP-Dichte (log-Skala)')
            
            # X-Achse Labels rotieren
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            
            # Statistiken im rechten Panel als Tabelle
            stats_data = []
            for category in available_categories:
                cat_data = all_effects_df[all_effects_df['effect_category'] == category]['density']
                if not cat_data.empty:
                    median_val = cat_data.median()
                    mean_val = cat_data.mean()
                    stats_data.append([category, f'{median_val:.2e}', f'{mean_val:.2e}'])
            
            # Statistiken-Tabelle erstellen
            ax_stats.axis('off')
            ax_stats.set_title('Statistiken', fontweight='bold', pad=20)
            
            if stats_data:
                table = ax_stats.table(cellText=stats_data,
                                     colLabels=['Kategorie', 'Median', 'Mean'],
                                     cellLoc='left',
                                     loc='center',
                                     colWidths=[0.5, 0.25, 0.25])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                
                # Tabellen-Styling
                for i in range(len(stats_data) + 1):
                    for j in range(3):
                        cell = table[(i, j)]
                        if i == 0:  # Header
                            cell.set_facecolor('#40466e')
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'category_density_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Zur PDF hinzufügen
        self.add_plot_to_pdf(fig)
        plt.close()


def run_dual_dataset_analysis(dataset1_config, dataset2_config, output_base_dir="results"):
    """
    Führt die SNP-Analyse auf zwei verschiedenen Datensätzen durch.
    
    Parameters
    ----------
    dataset1_config : dict
        Konfiguration für ersten Datensatz:
        {
            'name': 'dataset1',
            'path': 'path/to/dataset1',
            'pattern': '*.csv',
            'sep': ';'
        }
    dataset2_config : dict
        Konfiguration für zweiten Datensatz (gleiche Struktur)
    output_base_dir : str
        Basis-Verzeichnis für alle Ausgaben
    """
    print("Starte Dual-Dataset SNP-Analyse")
    print("=" * 80)
    
    # Dataset 1 analysieren
    print(f"\nAnalysiere Dataset 1: {dataset1_config['name']}")
    pipeline1 = SNPAnalysisPipeline(dataset1_config['name'], output_base_dir)
    pipeline1.run_complete_analysis(
        data_path=dataset1_config['path'],
        data_pattern=dataset1_config.get('pattern', '*.csv'),
        data_sep=dataset1_config.get('sep', ';')
    )
    
    # Dataset 2 analysieren
    print(f"\nAnalysiere Dataset 2: {dataset2_config['name']}")
    pipeline2 = SNPAnalysisPipeline(dataset2_config['name'], output_base_dir)
    pipeline2.run_complete_analysis(
        data_path=dataset2_config['path'],
        data_pattern=dataset2_config.get('pattern', '*.csv'),
        data_sep=dataset2_config.get('sep', ';')
    )
    
    print("\nDual-Dataset-Analyse erfolgreich abgeschlossen!")
    print(f"Alle Ergebnisse gespeichert in: {output_base_dir}")
    print("=" * 80)
    
    return pipeline1, pipeline2



if __name__ == "__main__":
    # Konfiguration für zwei Datensätze
    dataset1_config = {
        'name': 'dataset1',
        'path': 'csv_datein/dataset1/',
        'pattern': '*.csv',
        'sep': ';'
    }
    
    dataset2_config = {
        'name': 'dataset2', 
        'path': 'csv_datein/dataset2/',
        'pattern': '*.csv',
        'sep': ';'
    }
    
    # Dual-Dataset-Analyse ausführen
    pipeline1, pipeline2 = run_dual_dataset_analysis(dataset1_config, dataset2_config)
