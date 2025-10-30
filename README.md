# HuggingFace Data Processing Pipelines (Modular)

This repository contains modular data processing pipelines that fetch, process, and analyze data from HuggingFace Hub.

## 📁 Project Structure

```
├── config.py              # Configuration for models pipeline
├── config_datasets.py     # Configuration for datasets pipeline
├── config_papers.py       # Configuration for papers pipeline
├── utils.py               # Shared utility functions and logging
├── data_fetcher.py        # Data fetching for models
├── data_fetcher_datasets.py # Data fetching for datasets
├── data_fetcher_papers.py # Data fetching for papers
├── tag_processor.py       # Tag processing for models
├── tag_processor_datasets.py # Tag processing for datasets
├── data_processor.py      # Main processing logic for models
├── data_processor_datasets.py # Main processing logic for datasets
├── data_processor_papers.py # Semantic taxonomy mapping for papers
├── main.py               # Models pipeline orchestrator
├── main_datasets.py      # Datasets pipeline orchestrator
├── main_papers.py        # Papers pipeline orchestrator
├── test_pipeline.py      # Integration test for models
├── test_pipeline_datasets.py # Integration test for datasets
├── test_pipeline_papers.py # Integration test for papers
├── hub_download.py       # Weekly snapshot downloader
├── integrated_ml_taxonomy.json # ML taxonomy for papers
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

## 🚀 Available Pipelines

### 1. Models Pipeline (`main.py`)
Processes HuggingFace model data with feature extraction and categorization.

### 2. Datasets Pipeline (`main_datasets.py`)
Processes HuggingFace datasets data.

### 3. Papers Pipeline (`main_papers.py`) ⭐ NEW
Processes academic papers with **semantic taxonomy mapping** using spaCy NLP.

## 📄 Papers Pipeline Details

The papers pipeline includes advanced semantic analysis and citation tracking:

- **Data Source**: Loads papers from `cfahlgren1/hub-stats` dataset (daily_papers.parquet)
- **Semantic Taxonomy**: Uses spaCy's `en_core_web_lg` model for semantic similarity
- **Hierarchical Classification**: Maps paper keywords to ML taxonomy:
  - **Categories** (e.g., Computer Vision, NLP, Deep Learning)
  - **Subcategories** (e.g., Object Detection, Text Classification, GANs)
  - **Topics** (e.g., YOLO, BERT, Transformers)
- **Multi-Label Classification**: Papers can have multiple categories if they have close similarity scores (within 90% of top score)
- **Citation Tracking**: Fetches citation counts using `paperscraper` (via DOI and title)
- **Rich Metadata**: Preserves all 33+ original columns (authors, GitHub repos, upvotes, etc.)
- **Reports & Analytics**: Generates detailed matching reports and statistics
- **Auto-Upload**: Uploads to HuggingFace `evijit/paperverse_daily_data`

### Papers Pipeline Output

The pipeline generates:
1. `papers_with_semantic_taxonomy.parquet` - Full dataset with taxonomy
2. `papers_with_semantic_taxonomy.csv` - CSV version
3. `taxonomy_report.txt` - Detailed text report
4. `taxonomy_distribution.json` - Statistics in JSON format

## 🔧 Configuration

Key settings in respective config files:

**Models (`config.py`)**:
- `MODEL_ID_TO_DEBUG`: Specific model ID for debugging
- `TAG_MAP`: Feature flags and keywords
- `MODEL_SIZE_RANGES`: Size categorization thresholds

**Papers (`config_papers.py`)**:
- `TAXONOMY_FILE_PATH`: Path to ML taxonomy JSON
- `SIMILARITY_THRESHOLD`: Minimum cosine similarity (default: 0.55)
- `SPACY_MODEL`: NLP model to use (default: `en_core_web_lg`)
- `HF_REPO_ID`: Target HuggingFace repository
- `ENABLE_CITATION_FETCHING`: Enable/disable citation fetching (default: True)
- `CITATION_BATCH_SIZE`: Batch size for progress updates (default: 100)
- `MULTI_CLASS_ENABLED`: Allow multiple classifications per paper (default: True)
- `MULTI_CLASS_SCORE_THRESHOLD`: Include classes within 90% of top score (default: 0.90)
- `MAX_CLASSIFICATIONS`: Maximum classifications per level (default: 5)

## 🧪 Testing Individual Modules

You can test each pipeline independently:

```bash
# Test models pipeline (small subset)
export TEST_DATA_LIMIT=100
python test_pipeline.py

# Test datasets pipeline (small subset)
export TEST_DATA_LIMIT=100
python test_pipeline_datasets.py

# Test papers pipeline (small subset)
export TEST_DATA_LIMIT=50
python test_pipeline_papers.py

# Run full pipelines
python main.py           # Models
python main_datasets.py  # Datasets
python main_papers.py    # Papers
```

## 📦 Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Papers Pipeline - Additional Setup

The papers pipeline requires the spaCy language model and citation scraper:

```bash
# Download the spaCy model (will auto-download if missing)
python -m spacy download en_core_web_lg

# Install paperscraper for citation tracking
pip install paperscraper
```

**Notes**: 
- The `en_core_web_lg` model is ~500MB and will auto-download if not found
- `paperscraper` fetches citation counts from Semantic Scholar and Google Scholar
- Citation fetching can be disabled by setting `ENABLE_CITATION_FETCHING = False` in config

## ☁️ HuggingFace Upload

To enable automatic upload to HuggingFace:

```bash
# Set your HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

# Run the papers pipeline
python main_papers.py
```

The papers pipeline will upload results to: `evijit/paperverse_daily_data`

### Getting a HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **write** permissions
3. Copy the token and set it as an environment variable

## 🔄 GitHub Actions / CI/CD

For automated runs, add `HF_TOKEN` to your repository secrets:

1. Go to repository Settings → Secrets and variables → Actions
2. Add new secret: `HF_TOKEN` with your token value
3. The workflow will automatically upload results

