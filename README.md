# SNP Analysis Pipeline

A comprehensive bioinformatics tool for processing, analyzing, visualizing, and reporting Single Nucleotide Polymorphism (SNP) data from genomic datasets.

## Overview

This pipeline provides a complete analysis workflow for SNP data, including:
- SNP effect classification and categorization
- Allele frequency analysis
- Amino acid substitution analysis
- SNP density calculations
- Multiple SNP analysis per codon
- Comprehensive visualization and reporting

## Features

### Core Analysis Functions
- **SNP Effect Classification**: Categorizes SNPs into 10 disjoint categories
- **Allele Frequency Analysis**: Calculates frequencies per accession and category
- **Amino Acid Substitution Analysis**: Analyzes protein-changing mutations
- **SNP Density Analysis**: Normalizes SNP counts by gene/transcript lengths
- **Multiple SNP Analysis**: Identifies and analyzes codons with multiple SNPs
- **Unique Position Analysis**: Counts distinct SNP positions per gene

### Visualization
- **Bar plots**: Effect distributions, top genes, category comparisons
- **Histograms**: SNP count distributions, position distributions
- **Heatmaps**: Accession vs. effect matrices, category vs. isoform matrices
- **Box plots**: SNP density distributions across categories
- **Scatter plots**: Density comparisons and correlations

### Output Formats
- **PNG plots**: High-resolution images for presentations
- **PDF reports**: Comprehensive multi-page reports
- **CSV files**: Structured data for further analysis
- **Text reports**: Detailed statistical summaries

## Installation

### Prerequisites
- Python 3.7+
- Required packages:
  ```bash
  pip install pandas numpy matplotlib seaborn
  ```

### Setup
1. Clone or download the repository
2. Ensure your data files are in the correct format (see Data Format section)
3. Run the analysis scripts

## Usage

### Single Dataset Analysis
```python
from snp_analysis_pipeline import SNPAnalysisPipeline

# Initialize pipeline
pipeline = SNPAnalysisPipeline(
    dataset_name="Arabidopsis thaliana",
    output_base_dir="results"
)

# Run complete analysis
pipeline.run_complete_analysis(
    data_path=".",
    data_pattern="*.csv",
    data_sep=";"
)
```

### Command Line Usage
```bash
# Single dataset analysis
python run_single_analysis.py

# Dual dataset comparison
python run_dual_analysis.py
```

## Data Format

The pipeline expects CSV files with the following columns:
- `snp_id_c`: SNP identifier
- `transcript_ids`: Transcript identifiers
- `eff`: SNP effect annotation
- `alt`: Alternative allele
- `snp_id_e`: Extended SNP identifier
- `transcript_id`: Single transcript identifier
- `ref`: Reference allele
- `genotypes`: List of accessions with this SNP
- `gene_name`: Gene name
- `gene_id`: Gene identifier
- `feature_type`: Feature type annotation
- `transcript_biotype`: Transcript biotype
- `rank`: Rank information
- `hgvs_c`: HGVS coding sequence notation
- `snp_aa_label`: Amino acid change label
- `errors`: Error information
- `dp`: Depth information
- `aa_pos`: Amino acid position
- `aa_len`: Amino acid length
- `cdna_pos`: cDNA position
- `cdna_len`: cDNA length
- `cds_pos`: Coding sequence position
- `cds_len`: Coding sequence length
- `distance`: Distance information
- `annotation_impact`: Impact annotation
- `snp_aa_seq`: Amino acid sequence

## SNP Effect Categories

The pipeline classifies SNPs into 10 disjoint categories:

1. **Splice-related-coding-synonymous**: Splice region variants that are synonymous
2. **Splice-related-coding-non-synonymous**: Splice region variants that change amino acids
3. **Splice-related-non-coding**: Splice variants in non-coding regions
4. **Missense**: Amino acid changing variants
5. **Protein-changing-non-missense**: Other protein-changing variants (frameshift, stop, etc.)
6. **Synonymous**: Silent mutations
7. **UTR**: Untranslated region variants
8. **Intron**: Intronic variants
9. **Non-coding**: Non-coding transcript variants
10. **Other**: All other variants

## Output Structure

```
results/
├── [dataset_name]/
│   ├── plots/
│   │   ├── effect_counts.png
│   │   ├── effect_categories.png
│   │   ├── snp_effects_all.png
│   │   ├── multiple_snps_per_codon.png
│   │   ├── top10_genes_unique_positions.png
│   │   ├── unique_positions_histogram.png
│   │   ├── complete_report.pdf
│   │   └── [isoform_plots]/
│   ├── csv/
│   │   ├── effect_counts.csv
│   │   ├── snp_counts_per_gene.csv
│   │   ├── unique_positions_per_gene.csv
│   │   ├── allele_per_accession.csv
│   │   ├── category_allele_per_accession.csv
│   │   └── [isoform_data]/
│   └── reports/
│       ├── complete_report.txt
│       ├── multiple_snps_per_codon_report.txt
│       └── [other_reports]/
```

## Key Analysis Functions

### `classify_effect(effect: str) -> str`
Classifies a single SNP effect into one of 10 disjoint categories using priority-based logic.

### `analyze_multiple_snps_per_codon()`
Identifies codons with multiple SNPs and analyzes their effect combinations.

### `analyze_unique_positions_per_gene()`
Counts the number of distinct SNP positions per gene.

### `calculate_allele_frequencies_per_accession()`
Calculates allele frequencies for each accession across all effect categories.

### `analyze_aa_substitutions()`
Analyzes amino acid substitutions and creates substitution matrices.


### Customizing Analysis Parameters
```python
# Modify top N genes to display
pipeline.snp_counts_per_gene_analysis(top_n=20)

# Customize heatmap accessions
pipeline._plot_top_accessions_category_heatmap(top_n=50)
```

## Performance Notes

- The pipeline is optimized for datasets with thousands to millions of SNPs
- Memory usage scales with dataset size and number of accessions
- Processing time depends on dataset complexity and number of isoforms
- Large datasets may require several minutes to hours for complete analysis




## License

This project is licensed under the MIT License.

## Citation

If you use this pipeline in your research, please cite:

```
SNP Analysis Pipeline
[Name/Institution]
[Year]
```

## Contact

For questions, issues, or contributions, please contact [www@www.com] or open an issue on the project repository.



### Version 1.0.0
- Initial release
- Complete SNP analysis pipeline
- 10 disjoint effect categories
- Comprehensive visualization suite
- PDF report generation
- Multiple SNP analysis per codon
- Unique position analysis per gene
