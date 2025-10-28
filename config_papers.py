# --- config_papers.py ---
"""Configuration for papers data processing pipeline."""

# Data source
HF_PARQUET_URL = "https://huggingface.co/datasets/cfahlgren1/hub-stats/resolve/main/papers.parquet"

# Columns to fetch from source
RAW_DATA_COLUMNS_TO_FETCH = [
    'id',
    'paper_ai_keywords',
    'downloads',
    'downloadsAllTime',
    'likes',
    'tags'
]

# Expected columns after initial setup
EXPECTED_COLUMNS_SETUP = {
    'id': str,
    'paper_ai_keywords': object,  # List of keywords
    'downloads': float,
    'downloadsAllTime': float,
    'likes': float,
    'tags': object
}

# Final expected columns in output
FINAL_EXPECTED_COLUMNS = [
    'id',
    'paper_ai_keywords',
    'downloads',
    'downloadsAllTime',
    'likes',
    'tags',
    'organization',
    'data_download_timestamp',
    'taxonomy_categories',
    'taxonomy_subcategories',
    'taxonomy_topics',
    'primary_category',
    'primary_subcategory',
    'primary_topic',
    'matched_keywords_details',
    'category_scores',
    'subcategory_scores',
    'topic_scores'
]

# Output file
PROCESSED_PARQUET_FILE_PATH = 'papers_with_semantic_taxonomy.parquet'

# Taxonomy settings
TAXONOMY_FILE_PATH = 'integrated_ml_taxonomy.json'
SIMILARITY_THRESHOLD = 0.55
SPACY_MODEL = 'en_core_web_lg'

# HuggingFace upload settings
HF_REPO_ID = 'evijit/paperverse_daily_data'
HF_REPO_TYPE = 'dataset'
