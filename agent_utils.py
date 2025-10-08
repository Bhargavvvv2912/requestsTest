# agent_utils.py

import subprocess
import re
import sys

def start_group(title):
    print(f"\n::group::{title}")

def end_group():
    print("::endgroup::")

def run_command(command, cwd=None):
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr, result.returncode

def validate_changes(python_executable, group_title="Running Validation Script"):
    """
    Runs the 'requests' library's own test suite as the validation oracle.
    """
    start_group(group_title)
    
    # The test suite for 'requests' is in the 'tests' directory.
    # We run pytest from within the cloned repo.
    validation_command = [python_executable, "-m", "pytest", "tests"]
    
    print("\n--- Running Requests Test Suite (pytest) ---")
    
    # We MUST run the command from inside the 'requests' directory
    # so pytest can find all the files.
    stdout, stderr, returncode = run_command(validation_command, cwd="requests")

    if stdout:
        print(f"STDOUT:\n---\n{stdout}\n---")
    if stderr:
        print(f"STDERR:\n---\n{stderr}\n---")

    if returncode != 0:
        print("Validation Failed: Pytest returned a non-zero exit code.", file=sys.stderr)
        end_group()
        return False, None, stdout + stderr
    
    print("Validation script (pytest) completed successfully.")
    end_group()

    try:
        # A good metric for pytest is the number of tests passed.
        tests_passed = re.search(r"(\d+) passed", stdout).group(1)
        metrics_body = f"Performance Metrics:\n- Tests Passed: {tests_passed}"
        return True, metrics_body, stdout + stderr
    except (AttributeError, IndexError):
        return True, "Metrics not available, but validation passed.", stdout + stderr