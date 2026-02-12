"""
DATA PREPARATION: Raw Text to JSONL Consolidation

PURPOSE:
Consolidates 1,000+ categorical text files into a single structured .jsonl file.
This serves as the primary data ingestion step before model training/evaluation.

WHY JSONL?
- Memory Efficiency: Enables line-by-line streaming, preventing OOM (Out of Memory) issues.
- Schema Rigidity: Pairs each text chunk with its category label and metadata in one row.
- Reproducibility: Ensures ease of replication of the exact dataset used in this case study.

SETUP:
- Ensure the 'trellis_assessment_ds' folder is extracted and located in the /data directory.
- If the dataset is moved, update the DATA_PATH variable below.
"""
import os
import json
from pathlib import Path

def consolidate_to_jsonl(root_dir, output_file):
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Walk through the data directory
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            
            # Ensure we are looking at folders (business, food, etc.)
            if os.path.isdir(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith(".txt"):
                        file_path = os.path.join(category_path, filename)
                        
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                            content = f_in.read().strip()
                            
                            # Create a structured record
                            record = {
                                "category": category,
                                "file_name": filename,
                                "text": content
                            }
                            
                            # Write as a single line in JSONL
                            f_out.write(json.dumps(record) + '\n')
                            count += 1
                            
    print(f"Success! Processed {count} files into {output_file}")

from pathlib import Path

if __name__ == "__main__":
    # Get the project root based on where this script is located
    BASE_DIR = Path(__file__).resolve().parent.parent 
    
    # Define your paths clearly
    DATA_PATH = BASE_DIR / "data" / "trellis_assessment_ds"
    OUTPUT_PATH = BASE_DIR / "data" / "dataset.jsonl"
    
    print(f"Reading from: {DATA_PATH}")
    print(f"Saving to: {OUTPUT_PATH}")

    # check if the dataset directory exists
    if not DATA_PATH.exists():
        print(f"Error: Dataset directory not found at {DATA_PATH}")
        exit(1)
    
    consolidate_to_jsonl(str(DATA_PATH), str(OUTPUT_PATH))
