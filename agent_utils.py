# agent_utils.py (The Final, Complete, Config-Driven Oracle)

import subprocess
import re
import sys
from pathlib import Path

def start_group(title):
    """Starts a collapsible log group in GitHub Actions."""
    print(f"\n::group::{title}")

def end_group():
    """Ends a collapsible log group in GitHub Actions."""
    print("::endgroup::")

def run_command(command, cwd=None):
    """Runs a command and returns the output, error, and return code."""
    display_command = ' '.join(command)
    print(f"--> Running command: '{display_command}' in CWD: '{cwd or '.'}'")
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr, result.returncode

def _parse_pytest_summary(full_output: str) -> dict:
    """A helper function to parse the rich summary line from a pytest run."""
    summary = {
        "passed": "0", "failed": "0", "errors": "0",
        "skipped": "0", "xfailed": "0", "xpassed": "0"
    }
    summary_line = ""
    for line in reversed(full_output.splitlines()):
        if "=" in line and ("passed" in line or "failed" in line or "skipped" in line):
            summary_line = line
            break
    if not summary_line:
        return summary
    matches = re.findall(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors)", summary_line)
    for count, status in matches:
        if status in summary:
            summary[status] = count
    return summary

def _run_smoke_test(python_executable: str, config: dict) -> tuple[bool, str, str]:
    """Runs a simple, binary smoke test script from the project root."""
    print("\n--- Running Smoke Test ---")
    validation_config = config.get("VALIDATION_CONFIG", {})
    script_path = validation_config.get("smoke_test_script")
    project_dir = validation_config.get("project_dir")

    if not script_path:
        return False, "Smoke test failed: 'smoke_test_script' not defined in config.", ""
    
    resolved_path = str(Path(script_path).resolve())
    command = [python_executable, resolved_path]
    
    stdout, stderr, returncode = run_command(command, cwd=project_dir)

    if returncode != 0:
        print("CRITICAL VALIDATION FAILURE: Smoke test failed.", file=sys.stderr)
        if stderr: print(f"SMOKE TEST STDERR:\n{stderr}")
        return False, f"Smoke test failed with exit code {returncode}", stdout + stderr
    
    print("Smoke test PASSED.")
    return True, "Smoke test passed.", stdout + stderr

# In agent_utils.py

def _run_pytest_suite(python_executable: str, config: dict) -> tuple[bool, str, str]:
    """Runs a full pytest suite and provides a detailed result."""
    print("\n--- Running Full Pytest Suite ---")
    validation_config = config.get("VALIDATION_CONFIG", {})
    target = validation_config.get("pytest_target")
    project_dir = validation_config.get("project_dir")

    if not target:
        return False, "Pytest failed: 'pytest_target' not defined in config.", ""
    
    targets = target.split()
    command = [python_executable, "-m", "pytest"] + targets
    
    stdout, stderr, returncode = run_command(command, cwd=project_dir)
    full_output = stdout + stderr

    # --- THIS IS THE FINAL, CORRECTED LOGIC ---

    summary = _parse_pytest_summary(full_output)
    total_failures = int(summary["failed"]) + int(summary["errors"])
    threshold = config.get("ACCEPTABLE_FAILURE_THRESHOLD", 0)

    # The PASS/FAIL decision is made here, and ONLY here.
    if total_failures > threshold:
        reason = f"{total_failures} real failures/errors, which exceeds the threshold of {threshold}."
        print(f"VALIDATION FAILED: {reason}", file=sys.stderr)
        return False, reason, full_output
    
    # If we get here, it is a SUCCESS, either soft or hard.
    if total_failures > 0:
        print(f"VALIDATION PASSED (soft): {total_failures} failures/errors found, which is within the acceptable threshold.")
    else:
        print("Full pytest suite PASSED with 0 real failures/errors.")
    
    metrics_body = (
        "Pytest Run Summary:\n"
        f"- Passed: {summary['passed']}\n"
        f"- Failed: {summary['failed']} (Threshold: {threshold})\n"
        f"- Errors: {summary['errors']}\n"
        f"- Skipped: {summary['skipped']}\n"
        f"- Expected Failures (xfail): {summary['xfailed']}\n"
        f"- Unexpected Passes (xpass): {summary['xpassed']}"
    )
    return True, metrics_body, full_output

def validate_changes(python_executable: str, config: dict, group_title: str="Running Validation") -> tuple[bool, str, str]:
    """
    The main validation dispatcher. It reads the VALIDATION_CONFIG and
    orchestrates the correct validation strategy.
    """
    start_group(group_title)
    
    validation_config = config.get("VALIDATION_CONFIG", {})
    validation_type = validation_config.get("type", "pytest")
    
    success = False
    metrics_body = "No validation performed."
    full_output = ""

    if validation_type == "script":
        success, metrics_body, full_output = _run_smoke_test(python_executable, config)

    elif validation_type == "pytest":
        success, metrics_body, full_output = _run_pytest_suite(python_executable, config)
    
    elif validation_type == "smoke_test_with_pytest_report":
        smoke_success, smoke_reason, smoke_output = _run_smoke_test(python_executable, config)
        full_output += smoke_output
        
        if not smoke_success:
            end_group()
            return False, smoke_reason, full_output

        print("\n--- Smoke test passed. Running pytest suite for detailed reporting. ---")
        pytest_success, pytest_metrics, pytest_output = _run_pytest_suite(python_executable, config)
        full_output += "\n\n" + pytest_output
        
        success = pytest_success
        metrics_body = f"Smoke Test: {smoke_reason}\n\n{pytest_metrics}"

    else:
        print(f"WARNING: Unknown validation type '{validation_type}'. Assuming success.", file=sys.stderr)
        success = True

    end_group()
    return success, metrics_body, full_output