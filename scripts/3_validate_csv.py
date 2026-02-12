import csv
from pathlib import Path

# Configuration - Matches your previous script's output
DATA_DIR = Path('data/training')
FILES = ['train.csv', 'val.csv', 'test.csv']

def validate_split(file_path):
    print(f"--- Validating: {file_path.name} ---")
    
    categories = set()
    row_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Using DictReader to simulate how most libraries parse CSVs
            reader = csv.DictReader(f)
            
            # Check for header presence
            if not reader.fieldnames:
                print(f"❌ Error: {file_path.name} has no headers!")
                return
            
            print(f"✅ Headers found: {reader.fieldnames}")
            
            for row in reader:
                row_count += 1
                categories.add(row['category'])
        
        print(f"✅ Row count: {row_count}")
        print(f"✅ Unique categories: {len(categories)}")
        
        # Specific Logic Checks
        if file_path.name == 'train.csv' or file_path.name == 'val.csv':
            if 'other' in categories:
                print(f"🚨 CRITICAL: 'other' class detected in {file_path.name}!")
            else:
                print(f"✅ No OOD (other) leakage detected.")
        
        if file_path.name == 'test.csv':
            if 'other' in categories:
                print(f"✅ 'other' class present for OOD testing.")
                
    except Exception as e:
        print(f"❌ Failed to read {file_path.name}: {e}")
    print("\n")

def run_validation():
    if not DATA_DIR.exists():
        print(f"Error: Directory {DATA_DIR} not found. Did you run the split script?")
        return

    for filename in FILES:
        file_path = DATA_DIR / filename
        if file_path.exists():
            validate_split(file_path)
        else:
            print(f"Skipping {filename}: File not found.")

if __name__ == "__main__":
    run_validation()