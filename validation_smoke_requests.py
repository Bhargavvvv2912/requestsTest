# validation_smoke_requests.py

import sys
import requests

# Use httpbin.org - a fantastic, reliable service for testing HTTP requests.
# It is guaranteed to be stable and is designed for this exact purpose.
TEST_URL_GET = "https://httpbin.org/get"
TEST_URL_POST = "https://httpbin.org/post"

def run_requests_smoke_test():
    """
    Performs a simple but representative workflow with the requests library
    to validate its core functionality.
    """
    print("--- Starting requests Smoke Test ---")
    
    try:
        # --- Test 1: The "Simple" Test (GET request) ---
        # Goal: Can we successfully make a GET request and get a 200 OK response?
        # This tests the entire critical path: DNS, TCP/IP, TLS, HTTP parsing.
        print("Running Basic Test: Make a simple GET request...")
        
        response_get = requests.get(TEST_URL_GET, timeout=10) # 10 second timeout
        
        # Verify the most important outcome
        assert response_get.status_code == 200, \
            f"Basic Test Failed: Expected status code 200, got {response_get.status_code}"
        
        # Verify the content seems correct (httpbin echoes the request URL)
        assert response_get.json()["url"] == TEST_URL_GET, \
            "Basic Test Failed: Response JSON does not contain the correct URL."

        print("Basic Test PASSED.")


        # --- Test 2: The "Complex" Test (POST request with JSON and headers) ---
        # Goal: Can we send data, custom headers, and correctly receive the echo?
        # This tests more advanced features of the library.
        print("\nRunning Complex Test: Make a POST request with data and headers...")
        
        payload = {"agent": "AURA", "status": "testing"}
        headers = {"X-AURA-Test": "success"}

        response_post = requests.post(TEST_URL_POST, json=payload, headers=headers, timeout=10)

        assert response_post.status_code == 200, \
            f"Complex Test Failed: Expected status code 200, got {response_post.status_code}"

        # Verify that httpbin correctly received and echoed our data and headers
        response_data = response_post.json()
        assert response_data["json"] == payload, \
            f"Complex Test Failed: POST data mismatch. Sent {payload}, got {response_data['json']}"
        assert response_data["headers"]["X-Aura-Test"] == "success", \
            "Complex Test Failed: Custom header was not received correctly."
        
        print("Complex Test PASSED.")
        
        
        print("\n--- requests Smoke Test: ALL TESTS PASSED ---")
        return 0 # Return success code

    except requests.exceptions.RequestException as e:
        print("\n--- requests Smoke Test: FAILED ---", file=sys.stderr)
        print(f"A network-related error occurred: {type(e).__name__} - {e}", file=sys.stderr)
        return 1 # Return failure code
        
    except Exception as e:
        print(f"\n--- requests Smoke Test: FAILED ---", file=sys.stderr)
        print(f"An unexpected error occurred: {type(e).__name__} - {e}", file=sys.stderr)
        return 1
        

if __name__ == "__main__":
    sys.exit(run_requests_smoke_test())