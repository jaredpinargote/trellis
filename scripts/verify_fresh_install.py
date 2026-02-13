import os
import sys
import subprocess
import shutil
import time

def print_step(step):
    print(f"\n{'='*50}")
    print(f"STEP: {step}")
    print(f"{'='*50}")

def check_file(path):
    if os.path.exists(path):
        print(f"[OK] Found {path}")
        return True
    else:
        print(f"[FAIL] Missing {path}")
        return False

def main():
    print_step("Checking Environment Readiness")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check critical files
    required_files = [
        "requirements.txt",
        "requirements-prod.txt",
        "Dockerfile",
        "docker-compose.yml",
        "app/main.py",
        "models/baseline.joblib"
    ]
    
    missing = [f for f in required_files if not check_file(f)]
    if missing:
        print(f"Critical files missing: {missing}")
        sys.exit(1)

    print_step("Verifying Model Artifact")
    # Check model size
    model_size = os.path.getsize("models/baseline.joblib") / (1024 * 1024)
    print(f"Model Size: {model_size:.2f} MB")
    if model_size > 50:
        print("[WARN] Model is unexpectedly large!")
    else:
        print("[OK] Model is lightweight.")

    print_step("Running Unit Tests (Fast)")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/test_api.py", "-q"], check=True)
        print("[OK] Unit tests passed.")
    except subprocess.CalledProcessError:
        print("[FAIL] Unit tests failed.")
        sys.exit(1)

    print_step("Verifying Docker Build (Dry Run)")
    # Just check if docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("[OK] Docker is installed and accessible.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARN] Docker not found. Skipping Docker verification.")

    print("\n✅ FRESH INSTALL VERIFICATION PASSED!")
    print("Next: Run 'docker-compose up --build' to start the server.")

if __name__ == "__main__":
    main()
