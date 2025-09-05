# Uncensored AI in the Wild: Tracking Publicly Available and Locally Deployable LLMs

This repository contains the code and data for the research paper "Uncensored AI in the Wild: Tracking Publicly Available and Locally Deployable LLMs" which presents the first large-scale empirical analysis of safety-modified open-weight language models.

## Overview

This study analyzes 7,977 model repositories from Hugging Face to identify 2,238 distinct models explicitly adapted to bypass alignment safeguards. The research demonstrates systematic patterns in how these "uncensored" models are created, distributed, and optimized for local deployment.

## Repository Structure

### Data Collection Scripts
- `hf_incremental_data_scraping.py` - Main scraper for collecting model metadata from Hugging Face repositories
- `hf_compliance_selection.py` - Script for selecting representative models for safety evaluation

### Analysis Scripts  
- `model_trends_analysis.py` - Processes raw scrape data and generates normalized datasets with family attribution
- `unaligned_models_analysis.py` - Generates the paper figures and tables from the output of `model_trends_analyis.py`
- `hf_model_benchmarker_.py` - Evaluates selected models using Hugging Face API
- `hf_model_benchmarker_gguf.py` - Evaluates GGUF-format models using llama.cpp

### Generated Data Files

#### Model Catalogs
- `repo_catalog.csv` - Complete catalog of scraped repositories with metadata after filtering with `model_trends_analysis.py`
- `evaluated_models_metadata.csv` - Metadata for the subset of models evaluated for safety
- `model_canonical_summary.csv` - Deduplicated canonical model summaries

#### Evaluation Results
- `modified_model_evaluation_revised.csv` - Safety evaluation results for tested models
- `prompt_list.csv` - Catalog of unsafe prompts used for evaluation with regional classifications

#### Analysis Summaries
- `family_summary.csv` - Statistics by model family (LLaMA, Qwen, Mistral, etc.)
- `family_timeseries_monthly.csv` - Monthly release trends by family
- `provider_summary.csv` - Analysis by hosting provider
- `packaging_by_family.csv` / `packaging_by_model.csv` - Distribution of packaging formats (GGUF, safetensors, etc.)
- `quantization_by_family.csv` / `quantization_by_model.csv` - Quantization level analysis

## Key Findings

- **Scale**: Over 2,200 distinct safety-modified models identified
- **Growth**: Exponential increase from <10 models/month in early 2023 to >200/month by mid-2025
- **Effectiveness**: Modified models show ~80% compliance with unsafe prompts vs ~20% for unmodified models
- **Accessibility**: Heavy optimization for consumer hardware via GGUF packaging (73.5% of downloads) and aggressive quantization
- **Concentration**: Top 5% of providers account for 79% of downloads despite ecosystem's decentralized nature

## Installation

```bash
# Install required dependencies
pip install transformers torch huggingface_hub requests pandas numpy

# For GGUF evaluation
# Follow llama.cpp installation instructions at: https://github.com/ggerganov/llama.cpp
```

## Usage

### Data Collection
```bash
# Scrape Hugging Face for safety-modified models
python hf_incremental_data_scraping.py

# Process and normalize the scraped data
python model_trends_analysis.py
```

### Model Evaluation
```bash
# Select models for evaluation
python hf_compliance_selection.py

# Evaluate models for safety compliance
python hf_model_benchmarker_.py
python hf_model_benchmarker_gguf.py  # For GGUF models
```

### Analysis
```bash
# Generate summaries and visualizations
python unaligned_models_analysis.py
```

## Data Format

The CSV files use consistent schemas:
- Model identifiers follow Hugging Face's `owner/repository` format
- Download counts reflect Hugging Face package downloads
- Compliance scores range 0-6 (0=refusal, 6=full compliance)
- Geographic prompt categories include General, EU, China, and Russia-specific regulations

## Ethics and Safety

This research examines publicly available models to understand AI safety challenges. The model responses for safety evaluation are not included in this public repository due to their sensitive nature, but are available from the author by request for legitimate research purposes. The raw scraping results will be made available on another public repository due to size constraints on GitHub.

## Citation

```bibtex
[PLACEHOLDER - Citation to be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the research or data access requests, please contact the author.
