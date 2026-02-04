"""
Data ingestion pipeline for the Canadian Labour Market Analytics project.

Responsibilities:
- Discover raw monthly CSV files
- Normalize schema
- Append into a unified dataset
- Persist cleaned dataset as parquet

Usage:
    python src/ingest.py
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

# -----------------------------
# Configuration
# -----------------------------

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
OUTPUT_FILE = PROCESSED_DATA_PATH / "job_postings.parquet"

# Define columns you consider critical
REQUIRED_COLUMNS = [
    "job_title",
    "noc",
    "naics",
    "province",
    "city",
    "vacancies",
    "salary"
]

# -----------------------------
# Logging Setup
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------
# Helper Functions
# -----------------------------


def extract_month_from_filename(file_path: Path) -> str:
    """
    Extract YYYY-MM from filename.

    Example:
        job_postings_2023-01.csv -> 2023-01
    """
    for part in file_path.stem.split("_"):
        if "-" in part and len(part) == 7:
            return part

    logger.warning(f"Could not extract month from {file_path.name}")
    return "unknown"


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist.
    Add missing columns as NA.
    """
    df.columns = df.columns.str.lower().str.strip()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            logger.warning(f"Missing column '{col}' — filling with NA")
            df[col] = pd.NA

    return df


def read_csv_file(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-16", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            logger.info(f"Loaded {file_path.name} using {encoding}")
            
            df = normalize_schema(df)
            df["source_file"] = file_path.name
            df["source_month"] = extract_month_from_filename(file_path)

            return df

        except Exception:
            continue

    logger.error(f"Could not decode {file_path.name} with any encoding.")
    return pd.DataFrame()

def ingest_all_files() -> pd.DataFrame:
    """
    Discover and ingest all CSV files.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError("data/raw directory does not exist.")

    files = list(RAW_DATA_PATH.rglob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found in data/raw.")

    logger.info(f"Discovered {len(files)} files.")

    dfs = []

    for file in tqdm(files, desc="Ingesting files"):
        df = read_csv_file(file)

        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError(
            "No datasets were loaded. Check file encodings or file integrity."
        )

    combined = pd.concat(dfs, ignore_index=True)

    logger.info(f"Combined dataset shape: {combined.shape}")

    return combined


def save_dataset(df: pd.DataFrame):
    """
    Persist dataset as parquet.
    """
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)

    logger.info(f"Saved dataset -> {OUTPUT_FILE}")


def main():
    logger.info("Starting ingestion pipeline...")

    df = ingest_all_files()

    save_dataset(df)

    logger.info("Ingestion complete ✅")


if __name__ == "__main__":
    main()
