import subprocess
import time
import requests
import sys

def run_integration_test():
    print("Starting integration test...")
    # Start container
    try:
        subprocess.run(["docker-compose", "up", "-d", "--build"], check=True)
    except subprocess.CalledProcessError:
        print("Failed to start docker-compose")
        sys.exit(1)

    print("Waiting for service to be healthy...")
    url = "http://localhost:8000/health"
    retries = 10
    for i in range(retries):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("Service is UP!")
                break
        except requests.ConnectionError:
            pass
        print(f"Waiting... ({i+1}/{retries})")
        time.sleep(3)
    else:
        print("Service failed to become healthy.")
        subprocess.run(["docker-compose", "logs", "api"], check=False)
        subprocess.run(["docker-compose", "down"], check=False)
        sys.exit(1)

    # Test Prediction
    print("Testing prediction endpoint...")
    pred_url = "http://localhost:8000/classify_document"
    payload = {"document_text": "The court ruled in favor of the defendant."}
    try:
        r = requests.post(pred_url, json=payload)
        data = r.json()
        print(f"Response: {data}")
        assert r.status_code == 200
        assert "label" in data
        assert "model_version" in data
        print("Integration Test PASSED!")
    except Exception as e:
        print(f"Test Failed: {e}")
        subprocess.run(["docker-compose", "logs", "api"], check=False)
        sys.exit(1)
    finally:
        print("Cleaning up...")
        subprocess.run(["docker-compose", "down"], check=False)

if __name__ == "__main__":
    run_integration_test()
