#!/usr/bin/env python3
import argparse
import json
import logging
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Configure logging
logger = logging.getLogger("auth_verifier")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_API_KEY = "dev-secret-key"
BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/health"
CLASSIFY_URL = f"{BASE_URL}/classify_document"

def run_command(command, check=True):
    """Runs a shell command and logs output if in debug mode."""
    logger.debug(f"Executing: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            logger.debug(f"Command STDOUT:\n{result.stdout.strip()}")
        if result.stderr:
            logger.debug(f"Command STDERR:\n{result.stderr.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

def wait_for_service(timeout=60):
    """Polls the health endpoint until the service is ready."""
    logger.info("Waiting for API to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(HEALTH_URL) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    logger.info(f"API is ready! Status: {data.get('status')}")
                    logger.debug(f"Health Check Response: {data}")
                    return
        except urllib.error.URLError:
            pass
        except Exception as e:
            logger.debug(f"Health check failed: {e}")

        time.sleep(2)
        print(".", end="", flush=True)

    print()
    logger.error("Timeout waiting for API to become ready.")
    sys.exit(1)

def make_request(url, api_key=None, data=None):
    """Helper to make HTTP requests with optional API key."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data else None,
        headers=headers,
        method="POST" if data else "GET"
    )

    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode()
            return response.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return e.code, body
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return 0, str(e)

def run_tests():
    """Runs the authentication test suite."""
    logger.info("Starting Authentication Tests...")

    payload = {"document_text": "The company announced record profits for the third quarter."}

    # Test 1: No API Key
    logger.info("\n--- Test 1: Request without API Key ---")
    status, body = make_request(CLASSIFY_URL, data=payload)
    logger.debug(f"Response: {status} - {body}")

    if status == 401:
        logger.info("✅ SUCCESS: Received 401 Unauthorized as expected.")
    else:
        logger.error(f"❌ FAILURE: Expected 401, got {status}")
        sys.exit(1)

    # Test 2: Invalid API Key
    logger.info("\n--- Test 2: Request with Invalid API Key ---")
    status, body = make_request(CLASSIFY_URL, api_key="wrong-key-123", data=payload)
    logger.debug(f"Response: {status} - {body}")

    if status == 403:
        logger.info("✅ SUCCESS: Received 403 Forbidden as expected.")
    else:
        logger.error(f"❌ FAILURE: Expected 403, got {status}")
        sys.exit(1)

    # Test 3: Valid API Key
    logger.info("\n--- Test 3: Request with Valid API Key ---")
    status, body = make_request(CLASSIFY_URL, api_key=DEFAULT_API_KEY, data=payload)
    logger.debug(f"Response: {status} - {body}")

    if status == 200:
        logger.info("✅ SUCCESS: Received 200 OK as expected.")
        try:
            resp_json = json.loads(body)
            logger.info(f"   Label: {resp_json.get('label')}, Confidence: {resp_json.get('confidence')}")
        except:
            pass
    else:
        logger.error(f"❌ FAILURE: Expected 200, got {status}")
        logger.error(f"Response Body: {body}")
        sys.exit(1)

    logger.info("\n🎉 All authentication tests passed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Build, Run, and Test API Authentication.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    logger.info("🚀 Initializing environment...")

    try:
        # Build and start services
        logger.info("Building and starting Docker services...")
        run_command("docker-compose up -d --build")

        # Wait for API readiness
        wait_for_service()

        # Run Tests
        run_tests()

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        logger.info("\n🛑 Cleaning up...")
        try:
            run_command("docker-compose down")
            logger.info("Services stopped and removed.")
        except Exception as e:
            logger.error(f"Failed to clean up Docker services: {e}")

if __name__ == "__main__":
    main()
