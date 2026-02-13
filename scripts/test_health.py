import requests
import time
import sys

URL = "http://127.0.0.1:8000/health"

if __name__ == "__main__":
    print(f"Pinging {URL}...")
    try:
        # Retry loop
        for i in range(5):
            try:
                resp = requests.get(URL)
                if resp.status_code == 200:
                    print("✅ API is UP!")
                    print(resp.json())
                    sys.exit(0)
            except requests.exceptions.ConnectionError:
                pass
            
            print(f"Waiting for server... ({i+1}/5)")
            time.sleep(1)
            
        print("❌ Failed to connect to API.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
