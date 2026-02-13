from app.inference import ModelWrapper
try:
    mw = ModelWrapper()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model load failed: {e}")
