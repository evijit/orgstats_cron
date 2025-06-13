# HuggingFace Models Data Processing Pipeline (Modular)

This repository contains a modular data processing pipeline that fetches, processes, and analyzes model data from HuggingFace Hub.

## 📁 Project Structure

```
├── config.py              # Configuration and constants
├── utils.py               # Utility functions and logging
├── data_fetcher.py        # Data fetching from HuggingFace
├── tag_processor.py       # Tag processing and feature extraction
├── data_processor.py      # Main data processing logic
├── main.py               # Pipeline orchestrator
├── requirements.txt      # Python dependencies
├── daily_update_modular.yml # GitHub Actions workflow
└── README_MODULAR.md     # This documentation
```

## 🚀 Pipeline Overview

The pipeline consists of 6 main steps:

1. **Data Fetching** (`data_fetcher.py`)
   - Downloads raw model data from HuggingFace
   - Validates data integrity
   - Reports data statistics

2. **Initial Processing** (`data_processor.py`)
   - Sets up DataFrame structure
   - Calculates model file sizes
   - Categorizes models by size
   - Extracts organization information

3. **Tag Processing** (`tag_processor.py`)
   - Standardizes tag formats
   - Creates feature flags (robotics, audio, vision, etc.)
   - Analyzes tag distribution

4. **Final Processing**
   - Cleans up DataFrame
   - Ensures all expected columns exist
   - Validates final structure

5. **Data Saving**
   - Saves processed data to Parquet format
   - Performs file verification

6. **Upload**
   - Uploads to HuggingFace Space
   - Creates workflow summary

## 🔧 Configuration

Key settings in `config.py`:

- `MODEL_ID_TO_DEBUG`: Set to a specific model ID for detailed debugging
- `TAG_MAP`: Defines feature flags and their associated keywords
- `MODEL_SIZE_RANGES`: Size categorization thresholds
- `FINAL_EXPECTED_COLUMNS`: Expected output columns

## 🧪 Testing Individual Modules

You can test each module independently:

```bash
# Test data fetcher
python data_fetcher.py

# Test tag processor
python tag_processor.py

# Test data processor
python data_processor.py

# Run full pipeline
python main.py
```

