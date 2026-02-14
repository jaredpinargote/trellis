"""
Development Entry Point.

Run this script directly for local debugging (e.g. in IDEs).
For production, use the Dockerfile which runs uvicorn directly.
"""
from app.api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
