# validation_smoke_requests.py (The Final, Resilient Version)

import sys
import requests
import time

TEST_URL_GET = "https://httpbin.org/get"
TEST_URL_POST = "https://httpbin.org/post"
MAX_RETRIES = 3 # We will try up to 3 times
RETRY_DELAY = 10 # Wait 10 seconds between retries

def run_requests_smoke_test():
    """
    Performs a simple but representative workflow with the requests library,
    now with a robust retry mechanism to handle network flakiness.
    """
    print("--- Starting requests Smoke Test ---")
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"\nAttempt {attempt + 1} of {MAX_RETRIES}...")

            # --- Test 1: The "Simple" Test (GET request) ---
            print("  Running Basic Test: Make a simple GET request...")
            response_get = requests.get(TEST_URL_GET, timeout=20) # Increased timeout
            response_get.raise_for_status() # This will raise an exception for 4xx or 5xx errors

            assert response_get.json()["url"] == TEST_URL_GET
            print("  Basic Test PASSED.")

            # --- Test 2: The "Complex" Test (POST request) ---
            print("  Running Complex Test: Make a POST request with data and headers...")
            payload = {"agent": "AURA", "status": "testing"}
            headers = {"X-AURA-Test": "success"}
            response_post = requests.post(TEST_URL_POST, json=payload, headers=headers, timeout=20)
            response_post.raise_for_status()

            response_data = response_post.json()
            assert response_data["json"] == payload
            assert response_data["headers"]["X-Aura-Test"] == "success"
            print("  Complex Test PASSED.")
            
            # If we get here, both tests passed.
            print("\n--- requests Smoke Test: ALL TESTS PASSED ---")
            return 0

        except requests.exceptions.RequestException as e:
            print(f"  WARNING: Attempt {attempt + 1} failed. Network-related error: {type(e).__name__}", file=sys.stderr)
            if attempt < MAX_RETRIES - 1:
                print(f"  Will retry in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("\n--- requests Smoke Test: FAILED ---", file=sys.stderr)
                print(f"Final attempt failed. Aborting. Error: {type(e).__name__} - {e}", file=sys.stderr)
                return 1
        
        except Exception as e:
            print(f"\n--- requests Smoke Test: FAILED ---", file=sys.stderr)
            print(f"An unexpected error occurred: {type(e).__name__} - {e}", file=sys.stderr)
            return 1

if __name__ == "__main__":
    sys.exit(run_requests_smoke_test())