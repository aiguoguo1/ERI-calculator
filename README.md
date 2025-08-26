# ERI-calculator
A Python-based tool for calculating Ecological Risk Index (ERI) by integrating microbial community structure, metabolic function, detoxification capacity, and soil health parameters.
This tool calculates a comprehensive Ecological Risk Index (ERI) that integrates four key components:

CSI (Community Shift Index): Measures changes in microbial community structure using Bray-Curtis distance

MSI (Metabolic Shift Index): Quantifies alterations in metabolic function using Euclidean distance of carbon utilization patterns

DSI (Detoxification Shift Index): Assesses changes in enzymatic detoxification capacity

SHSI (Soil Health Shift Index): Evaluates alterations in soil physicochemical properties

The final ERI is calculated as:
ERI = 0.25 × (CSI + MSI + DSI + SHSI)

Features
Flexible Input: Automatically detects and handles various file encodings and formats

Adaptive Processing: Works with any number of parameters and samples

Robust Calculations: Includes error handling and normalization procedures

Comprehensive Output: Generates detailed results with intermediate calculations
Installation
Clone this repository:

bash
git clone https://github.com/yourusername/eri-calculator.git
cd eri-calculator
Install required dependencies:

bash
pip install pandas numpy scipy chardet

Usage
Basic Command
python eri_calculator.py \
    -enzyme enzyme_data.tsv \
    -otu otu_table.tsv \
    -spc soil_properties.tsv \
    -eco metabolic_data.tsv \
    -group sample_groups.tsv \
    -o results.tsv
-enzyme: Enzyme activity data file (TSV format)

-otu: OTU abundance table (TSV format)

-spc: Soil physicochemical properties (TSV format)

-eco: Metabolic function data (TSV format)

-group: Sample grouping information (TSV format)

-o: Output file path

--debug: Enable debug mode (optional)

An example dataset is provided in the example/ directory. To run the tool with the example data:
python eri_calculator.py \
    -enzyme example/enzyme_data.tsv \
    -otu example/otu_table.tsv \
    -spc example/soil_properties.tsv \
    -eco example/metabolic_data.tsv \
    -group example/sample_groups.tsv \
    -o example_results.tsv
