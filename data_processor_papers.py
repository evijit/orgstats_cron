# --- data_processor_papers.py ---
"""Main data processing module for Papers with semantic taxonomy mapping."""

import pandas as pd
import numpy as np
import json
import time
import os
from collections import defaultdict
from utils import log_progress, log_memory_usage, extract_org_from_id, validate_dataframe_structure
from config_papers import (
    EXPECTED_COLUMNS_SETUP,
    TAXONOMY_FILE_PATH,
    SIMILARITY_THRESHOLD,
    SPACY_MODEL,
    ENABLE_CITATION_FETCHING,
    CITATION_BATCH_SIZE,
    CITATION_RATE_LIMIT_DELAY,
    MAX_PAPERS_FOR_CITATIONS,
    MULTI_CLASS_ENABLED,
    MULTI_CLASS_SCORE_THRESHOLD,
    MAX_CLASSIFICATIONS
)

# Global variables for loaded resources
_nlp = None
_taxonomy_embeddings = None
_comprehensive_taxonomy = None

def load_spacy_model():
    """Load spaCy model with automatic download if needed."""
    global _nlp
    
    if _nlp is not None:
        return _nlp
    
    log_progress(f"🔤 Loading spaCy model '{SPACY_MODEL}'...")
    
    try:
        import spacy
        try:
            _nlp = spacy.load(SPACY_MODEL)
            log_progress(f"✅ spaCy model '{SPACY_MODEL}' loaded successfully")
            return _nlp
        except OSError:
            # Model not found, try to download it
            log_progress(f"⚠️  spaCy model '{SPACY_MODEL}' not found. Downloading now...")
            log_progress(f"   This is a ~500MB download and may take a few minutes...")
            
            import subprocess
            import sys
            
            # Use subprocess to download the model
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                log_progress(f"❌ Download failed: {result.stderr}")
                raise RuntimeError(f"Failed to download spaCy model. Please run: python -m spacy download {SPACY_MODEL}")
            
            log_progress(f"✅ Download complete. Loading model...")
            _nlp = spacy.load(SPACY_MODEL)
            log_progress(f"✅ spaCy model '{SPACY_MODEL}' loaded successfully")
            return _nlp
            
    except ImportError:
        log_progress(f"❌ spaCy not installed. Please run: pip install spacy")
        raise
    except Exception as e:
        log_progress(f"❌ Failed to load spaCy model: {e}")
        log_progress(f"Please manually install with: python -m spacy download {SPACY_MODEL}")
        raise

def load_taxonomy():
    """Load the comprehensive ML taxonomy from JSON file."""
    global _comprehensive_taxonomy
    
    if _comprehensive_taxonomy is not None:
        return _comprehensive_taxonomy
    
    log_progress(f"📚 Loading taxonomy from '{TAXONOMY_FILE_PATH}'...")
    
    if not os.path.exists(TAXONOMY_FILE_PATH):
        raise FileNotFoundError(f"Taxonomy file not found: {TAXONOMY_FILE_PATH}")
    
    with open(TAXONOMY_FILE_PATH, 'r') as f:
        _comprehensive_taxonomy = json.load(f)
    
    log_progress(f"✅ Loaded taxonomy with {len(_comprehensive_taxonomy)} categories:")
    for category in _comprehensive_taxonomy.keys():
        log_progress(f"     - {category}")
    
    return _comprehensive_taxonomy

def build_taxonomy_embeddings():
    """Build embeddings for all taxonomy terms."""
    global _taxonomy_embeddings
    
    if _taxonomy_embeddings is not None:
        return _taxonomy_embeddings
    
    log_progress("🧠 Building taxonomy embeddings from JSON...")
    
    nlp = load_spacy_model()
    taxonomy = load_taxonomy()
    
    _taxonomy_embeddings = {}
    
    for category, subcategories in taxonomy.items():
        # Add category-level embedding
        cat_doc = nlp(category)
        if cat_doc.has_vector:
            _taxonomy_embeddings[category] = {
                'vector': cat_doc.vector,
                'path': [category],
                'level': 'category'
            }
        
        for subcategory, topics in subcategories.items():
            # Add subcategory-level embedding
            subcat_doc = nlp(subcategory)
            if subcat_doc.has_vector:
                key = f"{category}|{subcategory}"
                _taxonomy_embeddings[key] = {
                    'vector': subcat_doc.vector,
                    'path': [category, subcategory],
                    'level': 'subcategory'
                }
            
            # Add topic-level embeddings
            if isinstance(topics, list):
                for topic in topics:
                    topic_doc = nlp(topic)
                    if topic_doc.has_vector:
                        key = f"{category}|{subcategory}|{topic}"
                        _taxonomy_embeddings[key] = {
                            'vector': topic_doc.vector,
                            'path': [category, subcategory, topic],
                            'level': 'topic'
                        }
    
    log_progress(f"✅ Built {len(_taxonomy_embeddings)} taxonomy embeddings:")
    log_progress(f"     - Categories: {sum(1 for v in _taxonomy_embeddings.values() if v['level'] == 'category')}")
    log_progress(f"     - Subcategories: {sum(1 for v in _taxonomy_embeddings.values() if v['level'] == 'subcategory')}")
    log_progress(f"     - Topics: {sum(1 for v in _taxonomy_embeddings.values() if v['level'] == 'topic')}")
    
    return _taxonomy_embeddings

def semantic_map_keywords(keywords, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Map keywords to taxonomy using semantic similarity.
    
    Args:
        keywords: List of keywords
        similarity_threshold: Minimum cosine similarity (0-1)
        
    Returns:
        Dictionary with categories, subcategories, topics, and matched keywords with scores
    """
    nlp = load_spacy_model()
    taxonomy_embeddings = build_taxonomy_embeddings()
    
    # Default empty result
    empty_result = {
        'categories': [],
        'subcategories': [],
        'topics': [],
        'matched_keywords': [],
        'category_scores': {},
        'subcategory_scores': {},
        'topic_scores': {}
    }
    
    # Handle None
    if keywords is None:
        return empty_result
    
    # Handle scalar NA/NaN values (not arrays)
    try:
        if not isinstance(keywords, (list, tuple, np.ndarray)) and pd.isna(keywords):
            return empty_result
    except (ValueError, TypeError):
        # pd.isna might fail on some types, continue
        pass
    
    # Handle empty lists or non-iterable values
    try:
        if len(keywords) == 0:
            return empty_result
    except (TypeError, AttributeError):
        # Not iterable or has no len
        return empty_result
    
    # Track best matches for each level
    category_scores = defaultdict(float)
    subcategory_scores = defaultdict(float)
    topic_scores = defaultdict(float)
    matched_keywords = []
    
    for keyword in keywords:
        keyword_str = str(keyword).strip()
        keyword_doc = nlp(keyword_str)
        
        if not keyword_doc.has_vector:
            continue
        
        keyword_vector = keyword_doc.vector
        best_match = None
        best_score = similarity_threshold
        
        # Find best matching taxonomy term
        for tax_key, tax_info in taxonomy_embeddings.items():
            tax_vector = tax_info['vector']
            
            # Compute cosine similarity
            similarity = np.dot(keyword_vector, tax_vector) / (
                np.linalg.norm(keyword_vector) * np.linalg.norm(tax_vector)
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match = (tax_key, tax_info, similarity)
        
        # If we found a good match
        if best_match:
            tax_key, tax_info, score = best_match
            path = tax_info['path']
            
            # Extract category, subcategory, topic from path
            category = path[0] if len(path) >= 1 else None
            subcategory = path[1] if len(path) >= 2 else None
            topic = path[2] if len(path) >= 3 else None
            
            # Keep highest score for each level
            if category:
                category_scores[category] = max(category_scores[category], score)
            if subcategory:
                subcategory_scores[subcategory] = max(subcategory_scores[subcategory], score)
            if topic:
                topic_scores[topic] = max(topic_scores[topic], score)
            
            matched_keywords.append({
                'keyword': keyword_str,
                'matched_to': tax_key.split('|')[-1],  # Last element of path
                'score': float(score),
                'category': category,
                'subcategory': subcategory,
                'topic': topic,
                'match_level': tax_info['level']
            })
    
    # Sort by score
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_subcategories = sorted(subcategory_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Apply multi-classification logic if enabled
    if MULTI_CLASS_ENABLED and len(sorted_categories) > 0:
        top_cat_score = sorted_categories[0][1]
        threshold_cat_score = top_cat_score * MULTI_CLASS_SCORE_THRESHOLD
        filtered_categories = [(cat, score) for cat, score in sorted_categories 
                              if score >= threshold_cat_score][:MAX_CLASSIFICATIONS]
    else:
        filtered_categories = sorted_categories
    
    if MULTI_CLASS_ENABLED and len(sorted_subcategories) > 0:
        top_subcat_score = sorted_subcategories[0][1]
        threshold_subcat_score = top_subcat_score * MULTI_CLASS_SCORE_THRESHOLD
        filtered_subcategories = [(subcat, score) for subcat, score in sorted_subcategories 
                                 if score >= threshold_subcat_score][:MAX_CLASSIFICATIONS]
    else:
        filtered_subcategories = sorted_subcategories
    
    if MULTI_CLASS_ENABLED and len(sorted_topics) > 0:
        top_topic_score = sorted_topics[0][1]
        threshold_topic_score = top_topic_score * MULTI_CLASS_SCORE_THRESHOLD
        filtered_topics = [(topic, score) for topic, score in sorted_topics 
                          if score >= threshold_topic_score][:MAX_CLASSIFICATIONS]
    else:
        filtered_topics = sorted_topics
    
    return {
        'categories': [cat for cat, score in filtered_categories],
        'subcategories': [subcat for subcat, score in filtered_subcategories],
        'topics': [topic for topic, score in filtered_topics],
        'matched_keywords': matched_keywords,
        'category_scores': {k: float(v) for k, v in filtered_categories},
        'subcategory_scores': {k: float(v) for k, v in filtered_subcategories},
        'topic_scores': {k: float(v) for k, v in filtered_topics}
    }

def setup_initial_dataframe(df_raw, data_download_timestamp):
    """Set up initial DataFrame for papers."""
    log_progress("🔧 Setting up initial DataFrame structure for Papers...")
    df = pd.DataFrame()
    
    for col_name, target_dtype in EXPECTED_COLUMNS_SETUP.items():
        if col_name in df_raw.columns:
            df[col_name] = df_raw[col_name]
            # Only convert numeric types, keep everything else as-is
            if target_dtype == float:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0.0)
        else:
            log_progress(f"     Column {col_name} missing, creating default values")
            if target_dtype == float:
                df[col_name] = 0.0
            else:
                df[col_name] = None

    df['data_download_timestamp'] = data_download_timestamp
    
    # Validate critical columns
    if 'paper_id' not in df.columns or df['paper_id'].isna().all():
        raise ValueError("'paper_id' column is required but missing or empty")
    
    log_progress(f"✅ DataFrame setup completed with {len(df.columns)} columns")
    log_memory_usage()
    return df

def get_paper_citations(paper_id, paper_title=None, paper_authors=None, log_details=False):
    """
    Fetch citation count and Semantic Scholar ID for a paper.
    
    Uses semanticscholar Python package with timeout handling.
    More reliable than REST API for batch processing despite being slower.
    
    Args:
        paper_id: ArXiv paper ID (e.g., '2510.22236') - not used but kept for compatibility
        paper_title: Paper title
        paper_authors: Not used (kept for compatibility)
        log_details: If True, log each paper's citation fetch result
        
    Returns:
        tuple: (citation_count, semantic_scholar_id) or (None, None) if unavailable
    """
    try:
        from semanticscholar import SemanticScholar
        from semanticscholar.SemanticScholarException import ObjectNotFoundException
        import signal
        
        # Timeout handler
        class TimeoutException(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException()
        
        # Search by title only
        if paper_title and isinstance(paper_title, str) and paper_title.strip():
            try:
                query = paper_title.strip()
                
                if log_details:
                    # Truncate title for display
                    display_title = query[:80] + "..." if len(query) > 80 else query
                    log_progress(f"🔍 Searching: {display_title}")
                
                # Set timeout to 90 seconds (same as before)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(90)
                
                try:
                    sch = SemanticScholar()
                    results = sch.search_paper(query, limit=1)
                    
                    # Cancel the alarm
                    signal.alarm(0)
                    
                    if results and len(results) > 0:
                        paper = results[0]
                        citation_count = paper.citationCount if hasattr(paper, 'citationCount') else None
                        ss_paper_id = paper.paperId if hasattr(paper, 'paperId') else None
                        
                        if log_details:
                            if citation_count is not None:
                                log_progress(f"   ✅ Found: {citation_count:,} citations (ID: {ss_paper_id})")
                            else:
                                log_progress(f"   ⚠️  Found paper but no citation data (ID: {ss_paper_id})")
                        
                        return (citation_count, ss_paper_id)
                    else:
                        if log_details:
                            log_progress(f"   ❌ Not found in Semantic Scholar")
                            
                except TimeoutException:
                    signal.alarm(0)
                    if log_details:
                        log_progress(f"   ⚠️  Timeout (90s)")
                except ObjectNotFoundException:
                    signal.alarm(0)
                    if log_details:
                        log_progress(f"   ❌ Not found in Semantic Scholar")
                        
            except Exception as e:
                signal.alarm(0)
                if log_details:
                    log_progress(f"   ❌ Error: {str(e)[:100]}")
        
        return (None, None)
        
    except ImportError:
        if log_details:
            log_progress("   ❌ semanticscholar package not installed")
        return (None, None)
    except Exception as e:
        if log_details:
            log_progress(f"   ❌ Unexpected error: {str(e)[:100]}")
        return (None, None)

def fetch_citations(df):
    """
    Fetch citation counts and Semantic Scholar IDs for all papers in the dataframe.
    
    Args:
        df: DataFrame with paper_id and paper_title columns
        
    Returns:
        DataFrame with citation_count and semantic_scholar_id columns added
    """
    if not ENABLE_CITATION_FETCHING:
        log_progress("⚠️  Citation fetching is disabled. Skipping...")
        df['citation_count'] = None
        df['semantic_scholar_id'] = None
        return df
    
    total_papers = len(df)
    
    # Determine how many papers to process
    if MAX_PAPERS_FOR_CITATIONS is not None and total_papers > MAX_PAPERS_FOR_CITATIONS:
        papers_to_process = MAX_PAPERS_FOR_CITATIONS
        log_progress(f"📚 Fetching citation counts for top {papers_to_process:,} papers (limited from {total_papers:,})...")
        log_progress(f"   Sorting by paper_upvotes to prioritize popular papers...")
        # Sort by upvotes descending and take top N
        df_sorted = df.sort_values('paper_upvotes', ascending=False, na_position='last')
        df = df_sorted.copy()
    else:
        papers_to_process = total_papers
        log_progress(f"📚 Fetching citation counts for all {total_papers:,} papers...")
    
    log_progress("   Using Semantic Scholar API (title search, first result)")
    log_progress(f"   Rate limit: {CITATION_RATE_LIMIT_DELAY}s delay between requests")
    estimated_time = papers_to_process * CITATION_RATE_LIMIT_DELAY / 60
    log_progress(f"   Estimated time: ~{estimated_time:.1f} minutes")
    
    start_time = time.time()
    citation_counts = []
    semantic_scholar_ids = []
    successful_fetches = 0
    
    for i, row in df.iterrows():
        # Stop after processing the limit
        if i >= papers_to_process:
            # Fill remaining papers with None
            remaining = total_papers - papers_to_process
            citation_counts.extend([None] * remaining)
            semantic_scholar_ids.extend([None] * remaining)
            break
        
        paper_id = row.get('paper_id', '')
        paper_title = row.get('paper_title', '')
        
        citations, ss_id = get_paper_citations(paper_id, paper_title)
        citation_counts.append(citations)
        semantic_scholar_ids.append(ss_id)
        
        if citations is not None:
            successful_fetches += 1
        
        # Add delay to respect rate limits
        if i < papers_to_process - 1:  # Don't delay after last paper
            time.sleep(CITATION_RATE_LIMIT_DELAY)
        
        # Show progress every batch
        if (i + 1) % CITATION_BATCH_SIZE == 0 or (i + 1) == papers_to_process:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (papers_to_process - i - 1) / rate if rate > 0 else 0
            log_progress(f"   Processed {i + 1:,}/{papers_to_process:,} papers " +
                        f"({successful_fetches} citations found, {rate:.2f} papers/sec, ETA: {eta/60:.1f}min)...")
    
    df['citation_count'] = citation_counts
    df['semantic_scholar_id'] = semantic_scholar_ids
    
    elapsed_time = time.time() - start_time
    log_progress(f"✅ Citation fetching completed in {elapsed_time/60:.1f} minutes")
    log_progress(f"   Found citations for {successful_fetches:,}/{papers_to_process:,} papers " +
                f"({successful_fetches/papers_to_process*100:.1f}%)")
    if papers_to_process < total_papers:
        log_progress(f"   Note: {total_papers - papers_to_process:,} papers not processed (limit: {MAX_PAPERS_FOR_CITATIONS:,})")
    
    # Show statistics
    valid_citations = df['citation_count'].dropna()
    if len(valid_citations) > 0:
        log_progress(f"   Citation statistics:")
        log_progress(f"      - Mean: {valid_citations.mean():.1f}")
        log_progress(f"      - Median: {valid_citations.median():.1f}")
        log_progress(f"      - Max: {valid_citations.max():.0f}")
        log_progress(f"      - Total citations: {valid_citations.sum():.0f}")
    
    log_memory_usage()
    return df

def enrich_data(df):
    """Extract organization name from organization dict/string."""
    log_progress("✨ Processing organization data...")
    
    # The organization column might be a dict, string, or None
    # Extract the name if it's a dict
    def extract_org_name(org):
        if org is None or pd.isna(org):
            return "unaffiliated"
        if isinstance(org, dict):
            # Try to get name from dict
            return org.get('name') or org.get('fullname') or org.get('_id') or "unaffiliated"
        if isinstance(org, str):
            return org if org else "unaffiliated"
        return "unaffiliated"
    
    # Create a clean organization_name column
    df['organization_name'] = df['organization'].apply(extract_org_name)
    
    org_count = df['organization_name'].nunique()
    log_progress(f"   Found {org_count:,} unique organizations.")
    log_memory_usage()
    return df

def apply_semantic_taxonomy(df):
    """Apply semantic taxonomy mapping to papers."""
    log_progress("🏷️  Applying semantic taxonomy matching to papers...")
    log_progress("   This may take a few minutes depending on dataset size...")
    
    start_time = time.time()
    
    # Pre-load models and embeddings
    build_taxonomy_embeddings()
    
    # Process in batches to show progress
    batch_size = 1000
    results = []
    
    total_rows = len(df)
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_results = batch['paper_ai_keywords'].apply(
            lambda x: semantic_map_keywords(x, similarity_threshold=SIMILARITY_THRESHOLD)
        )
        results.extend(batch_results)
        log_progress(f"   Processed {min(i+batch_size, total_rows):,}/{total_rows:,} papers...")
    
    log_progress("   Creating taxonomy columns...")
    
    # Create new columns
    df['taxonomy_info'] = results
    df['taxonomy_categories'] = df['taxonomy_info'].apply(lambda x: x['categories'])
    df['taxonomy_subcategories'] = df['taxonomy_info'].apply(lambda x: x['subcategories'])
    df['taxonomy_topics'] = df['taxonomy_info'].apply(lambda x: x['topics'])
    df['primary_category'] = df['taxonomy_categories'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['primary_subcategory'] = df['taxonomy_subcategories'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['primary_topic'] = df['taxonomy_topics'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['matched_keywords_details'] = df['taxonomy_info'].apply(lambda x: x['matched_keywords'])
    df['category_scores'] = df['taxonomy_info'].apply(lambda x: x['category_scores'])
    df['subcategory_scores'] = df['taxonomy_info'].apply(lambda x: x['subcategory_scores'])
    df['topic_scores'] = df['taxonomy_info'].apply(lambda x: x['topic_scores'])
    
    # Drop intermediate column
    df = df.drop('taxonomy_info', axis=1)
    
    elapsed_time = time.time() - start_time
    log_progress(f"✅ Semantic taxonomy matching completed in {elapsed_time:.2f}s")
    
    # Show coverage
    total_papers = len(df)
    classified_papers = df['primary_category'].notna().sum()
    log_progress(f"   Coverage: {classified_papers:,}/{total_papers:,} papers ({classified_papers/total_papers*100:.1f}%)")
    
    # Show multi-classification statistics
    if MULTI_CLASS_ENABLED:
        multi_cat_papers = df['taxonomy_categories'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False).sum()
        log_progress(f"   Multi-category papers: {multi_cat_papers:,} ({multi_cat_papers/total_papers*100:.1f}%)")
        avg_cats_per_paper = df['taxonomy_categories'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        log_progress(f"   Average categories per paper: {avg_cats_per_paper:.2f}")
    
    # Show category distribution
    category_counts = df['primary_category'].value_counts()
    log_progress(f"   Top categories:")
    for cat, count in category_counts.head(10).items():
        log_progress(f"      - {cat}: {count:,}")
    
    log_memory_usage()
    return df

if __name__ == "__main__":
    log_progress("🧪 Testing data_processor_papers module...")
    
    # Create sample data with organization as dict (like real data)
    raw_data = {
        'paper_id': ['org1/paper1', 'org2/paper2', 'unaffiliated_paper3'],
        'paper_title': ['Deep Learning Paper', 'Computer Vision Study', 'NLP Research'],
        'paper_ai_keywords': [
            ['neural networks', 'deep learning'],
            ['computer vision', 'object detection'],
            ['natural language processing']
        ],
        'paper_upvotes': [100, 200, 300],
        'paper_publishedAt': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'organization': [
            {'name': 'org1', 'fullname': 'Organization 1'},
            {'name': 'org2', 'fullname': 'Organization 2'},
            None
        ]
    }
    df_raw_test = pd.DataFrame(raw_data)
    timestamp_test = pd.Timestamp.now(tz='UTC')

    try:
        df_test = setup_initial_dataframe(df_raw_test, timestamp_test)
        df_test = enrich_data(df_test)
        df_test = apply_semantic_taxonomy(df_test)
        
        log_progress("✅ Data processor test successful")
        print("\n--- Final Test DataFrame ---")
        print(df_test[['paper_id', 'organization_name', 'primary_category', 'primary_subcategory', 'primary_topic']].to_string())
        print("--------------------------\n")

    except Exception as e:
        log_progress(f"❌ Data processor test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
