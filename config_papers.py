# --- config_papers.py ---
"""Configuration for papers data processing pipeline."""

# Data source
HF_PARQUET_URL = "https://huggingface.co/datasets/cfahlgren1/hub-stats/resolve/main/daily_papers.parquet"

# Columns to fetch from source - fetch all available columns
RAW_DATA_COLUMNS_TO_FETCH = [
    'publishedAt',
    'title',
    'summary',
    'thumbnail',
    'numComments',
    'submittedBy',
    'organization',
    'isAuthorParticipating',
    'mediaUrls',
    'paper_id',
    'paper_authors',
    'paper_publishedAt',
    'paper_submittedOnDailyAt',
    'paper_title',
    'paper_summary',
    'paper_upvotes',
    'paper_discussionId',
    'paper_ai_summary',
    'paper_ai_keywords',
    'paper_submittedOnDailyBy._id',
    'paper_submittedOnDailyBy.avatarUrl',
    'paper_submittedOnDailyBy.isPro',
    'paper_submittedOnDailyBy.fullname',
    'paper_submittedOnDailyBy.user',
    'paper_submittedOnDailyBy.type',
    'paper_organization._id',
    'paper_organization.name',
    'paper_organization.fullname',
    'paper_organization.avatar',
    'paper_githubRepo',
    'paper_githubStars',
    'paper_mediaUrls',
    'paper_projectPage',
    'paper_withdrawnAt'
]

# Expected columns after initial setup (keep all as object for flexibility)
EXPECTED_COLUMNS_SETUP = {col: object for col in RAW_DATA_COLUMNS_TO_FETCH}
# Override specific columns that need numeric types
EXPECTED_COLUMNS_SETUP.update({
    'paper_upvotes': float,
    'numComments': float,
    'paper_githubStars': float,
})

# Final expected columns in output (all original columns plus taxonomy columns)
FINAL_EXPECTED_COLUMNS = RAW_DATA_COLUMNS_TO_FETCH + [
    'data_download_timestamp',
    'organization_name',  # Clean string version of organization
    'citation_count',  # Number of citations from Semantic Scholar
    'semantic_scholar_id',  # Semantic Scholar paper ID
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

# Citation fetching settings
ENABLE_CITATION_FETCHING = True  # Set to False to skip citation fetching
CITATION_BATCH_SIZE = 100  # Process citations in batches to show progress
CITATION_RATE_LIMIT_DELAY = 3  # Seconds between requests (100 req/5min = 3s delay)
MAX_PAPERS_FOR_CITATIONS = None  # None = all papers (parallel jobs handle time limits automatically)

# Output file
PROCESSED_PARQUET_FILE_PATH = 'papers_with_semantic_taxonomy.parquet'

# Taxonomy settings
TAXONOMY_FILE_PATH = 'integrated_ml_taxonomy.json'
SIMILARITY_THRESHOLD = 0.55
SPACY_MODEL = 'en_core_web_lg'

# Multi-classification settings
MULTI_CLASS_ENABLED = True  # Allow multiple classifications per paper
MULTI_CLASS_SCORE_THRESHOLD = 0.90  # Include additional classes within 90% of top score
# E.g., if top score is 0.8, include all scores >= 0.72 (0.8 * 0.9)
MAX_CLASSIFICATIONS = 5  # Maximum number of classifications per level

# HuggingFace upload settings
HF_REPO_ID = 'evijit/paperverse_daily_data'
HF_REPO_TYPE = 'dataset'
