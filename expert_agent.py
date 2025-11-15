# expert_agent.py

import re
import json
import ast
from google.api_core.exceptions import ResourceExhausted

class ExpertAgent:
    """
    The "Expert" Agent (CORE). A pure reasoning engine that handles all
    interactions with the Large Language Model.
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        self.llm_available = True

    def _extract_key_constraints(self, error_log: str) -> list:
        """A simple helper to find the most important lines in a pip error log."""
        key_lines = []
        # This regex looks for lines that explicitly state a requirement.
        # e.g., "pandas 2.2.0 requires numpy<2,>=1.22.4"
        pattern = re.compile(r"^\s*([a-zA-Z0-9\-_]+.* requires .*)$", re.MULTILINE)
        for match in pattern.finditer(error_log):
            key_lines.append(match.group(1).strip())
        
        # Limit to the 5 most relevant lines to keep the prompt clean.
        return key_lines[:5]

    def summarize_error(self, error_message: str) -> str:
        """Generates a concise, one-sentence summary of a pip error log."""
        if not self.llm_available: return "(LLM summary unavailable)"
        
        # --- IMPROVEMENT: Use the key constraints for a better summary ---
        key_constraints = self._extract_key_constraints(error_message)
        if key_constraints:
            context = "Key constraints found:\n" + "\n".join(key_constraints)
        else:
            context = f"Full Error Log: --- {error_message} ---"

        prompt = (
            "Summarize the root cause of the following Python dependency conflict "
            "in a single, concise sentence. Focus on the package names and versions. "
            f"Context: {context}"
        )
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip().replace('\n', ' ')
        except Exception as e:
            print(f"  -> LLM_ERROR: Could not get summary: {e}")
            return "Failed to get summary from LLM."

    def propose_co_resolution(self, target_package: str, error_log: str, available_updates: dict) -> dict | None:
        """
        Analyzes a complex conflict and proposes a multi-package "moonshot" update.
        Responds with a structured plan if a plausible solution is found.
        """
        if not self.llm_available: return None

        # --- IMPROVEMENT #1: Extract key constraints for a focused prompt ---
        key_constraints = self._extract_key_constraints(error_log)
        if not key_constraints:
             # If we can't find specific constraints, the error is too generic to solve.
             print("  -> Expert Agent: The error log is too generic. Cannot propose a co-resolution.")
             return None

        # --- IMPROVEMENT #2: Provide "Oracle Metadata" of available versions ---
        # available_updates is now a dict: {'package_name': 'latest_version'}
        update_options_str = json.dumps(available_updates, indent=2)

        prompt = f"""
        You are an expert Python dependency conflict resolver named CORE.
        Your task is to propose a multi-package update to fix a conflict.
        Respond ONLY with a valid JSON object with the keys "plausible" (boolean) and "proposed_plan" (a list of 'package==version' strings).

        ANALYSIS CONTEXT:
        1. The primary goal was to update the package: '{target_package}'.
        2. The attempt failed due to the following specific constraints:
           --- KEY CONSTRAINTS ---
           {key_constraints}
           --- END KEY CONSTRAINTS ---
        3. The following packages have updates available. YOU MUST CHOOSE FROM THIS LIST:
           --- AVAILABLE UPDATES (ORACLE METADATA) ---
           {update_options_str}
           --- END AVAILABLE UPDATES ---

        YOUR TASK:
        Based on the key constraints, is it plausible that updating one or more of the related packages
        *along with* '{target_package}' would solve the conflict? If so, construct the
        most likely plan using the versions provided in the AVAILABLE UPDATES.

        EXAMPLE RESPONSE:
        {{
            "plausible": true,
            "proposed_plan": ["{target_package}==<version_from_list>", "related_package_A==<version_from_list>"]
        }}
        """
        try:
            response = self.llm.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not match:
                print(f"  -> LLM_WARNING: Co-resolution response was not valid JSON: {response.text}")
                return None
            
            plan = json.loads(match.group(0))
            if isinstance(plan.get("plausible"), bool) and isinstance(plan.get("proposed_plan"), list):
                # We can add a final check here to ensure the LLM didn't hallucinate.
                for requirement in plan.get("proposed_plan", []):
                    pkg, ver = requirement.split('==')
                    if pkg not in available_updates or ver != available_updates[pkg]:
                        print(f"  -> LLM_WARNING: Plan contains a hallucinated or incorrect version: {requirement}")
                        return None
                return plan
            return None
        except (json.JSONDecodeError, AttributeError, Exception) as e:
            print(f"  -> LLM_ERROR: Could not get/parse co-resolution plan: {e}")
            return None
        
        # In expert_agent.py

    def diagnose_conflict_from_log(self, error_log: str) -> list[str]:
        """
        Analyzes a pip error log and returns a structured list of conflicting package names.
        """
        if not self.llm_available:
            return []

        prompt = f"""
        You are an expert Python dependency conflict analyst. Your task is to read a pip
        error log and identify the specific, root-cause package names that are in conflict.

        Respond ONLY with a valid, clean JSON list of strings.

        EXAMPLE 1:
        Log: "ERROR: Cannot install -r file.txt (line 16) and google-api-core==1.34.0 because..."
        Response: ["google-api-core", "google-generativeai"]

        EXAMPLE 2:
        Log: "ERROR: a==1.0 requires b<2.0, but you have b==2.5"
        Response: ["a", "b"]

        Here is the error log to analyze:
        --- ERROR LOG ---
        {error_log}
        --- END ERROR LOG ---
        """
        try:
            response = self.llm.generate_content(prompt)
            match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if not match:
                print(f"  -> LLM_WARNING: Diagnose conflict response was not valid JSON list: {response.text}")
                return []
            
            package_list = json.loads(match.group(0))
            if isinstance(package_list, list) and all(isinstance(p, str) for p in package_list):
                return package_list
            return []
        except (json.JSONDecodeError, AttributeError, Exception) as e:
            print(f"  -> LLM_ERROR: Could not get/parse conflict diagnosis: {e}")
            return []
        
    