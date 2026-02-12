import uvicorn
import os
import sys

# Add project root to sys.path so we can import 'app'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

if __name__ == "__main__":
    print(f"Starting API server from {PROJECT_ROOT}...")
    # reload=True for dev convenience
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
