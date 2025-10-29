# agent_logic.py

import os
import sys
import venv
from pathlib import Path
import ast
import shutil
import re
import json
from google.api_core.exceptions import ResourceExhausted
from pypi_simple import PyPISimple
from packaging.version import parse as parse_version
from agent_utils import start_group, end_group, run_command, validate_changes

class DependencyAgent:
    def __init__(self, config, llm_client):
        self.config = config
        self.llm = llm_client
        self.pypi = PyPISimple()
        self.requirements_path = Path(config["REQUIREMENTS_FILE"])
        self.primary_packages = self._load_primary_packages()
        self.llm_available = True
        self.usage_scores = self._calculate_risk_scores()

        # *** THE FIX IS HERE: Initialize the missing attribute. ***
        self.exclusions_from_this_run = set()

    def _calculate_risk_scores(self):
        start_group("Analyzing Codebase for Update Risk")
        scores = {}
        repo_root = Path('.')
        for py_file in repo_root.rglob('*.py'):
            if any(part in str(py_file) for part in ['temp_venv', 'final_venv', 'bootstrap_venv', 'agent_logic.py', 'agent_utils.py', 'dependency_agent.py']):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                module_name = self._get_package_name_from_spec(alias.name)
                                scores[module_name] = scores.get(module_name, 0) + 1
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            module_name = self._get_package_name_from_spec(node.module)
                            scores[module_name] = scores.get(module_name, 0) + 1
            except Exception: continue
        
        normalized_scores = {name.replace('_', '-'): score for name, score in scores.items()}
        print("Usage scores calculated.")
        end_group()
        return normalized_scores

    def _get_package_name_from_spec(self, spec_line):
        match = re.match(r'([a-zA-Z0-9\-_]+)', spec_line)
        return match.group(1) if match else None

    def _load_primary_packages(self):
        primary_path = Path(self.config["PRIMARY_REQUIREMENTS_FILE"])
        if not primary_path.exists(): return set()
        with open(primary_path, "r") as f:
            return {self._get_package_name_from_spec(line.strip()) for line in f if line.strip() and not line.startswith('#')}

    def _get_requirements_state(self):
        if not self.requirements_path.exists(): sys.exit(f"Error: {self.config['REQUIREMENTS_FILE']} not found.")
        with open(self.requirements_path, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return all('==' in line for line in lines), lines

    def _bootstrap_unpinned_requirements(self):
        start_group("BOOTSTRAP: Establishing a Stable Baseline")
        print("Unpinned requirements detected. Creating and validating a stable baseline...")
        venv_dir = Path("./bootstrap_venv")
        if venv_dir.exists(): shutil.rmtree(venv_dir)
        venv.create(venv_dir, with_pip=True)
        
        # This function now uses the new, robust helper function
        success, result, error_log = self._run_bootstrap_and_validate(venv_dir, self.requirements_path)
        
        if success:
            print("\nInitial baseline is valid and stable!")
            with open(self.requirements_path, "w") as f: f.write(result["packages"])
            start_group("View new requirements.txt content"); print(result["packages"]); end_group()
            if result["metrics"] and "not available" not in result["metrics"]:
                print(f"\n{'='*70}\n=== BOOTSTRAP SUCCESSFUL: METRICS FOR THE NEW BASELINE ===\n" + "\n".join([f"  {line}" for line in result['metrics'].split('\n')]) + f"\n{'='*70}\n")
                with open(self.config["METRICS_OUTPUT_FILE"], "w") as f: f.write(result["metrics"])
            end_group()
            return

        # --- The rest of the new, resilient bootstrap logic follows ---
        print("\nCRITICAL: Initial baseline failed validation. Initiating Bootstrap Healing Protocol.", file=sys.stderr)
        start_group("View Initial Baseline Failure Log"); print(error_log); end_group()
        
        python_executable = str((venv_dir / "bin" / "python").resolve())
        initial_failing_packages_list, _, _ = run_command([python_executable, "-m", "pip", "freeze"])
        initial_failing_packages = self._prune_pip_freeze(initial_failing_packages_list).split('\n')

        healed_packages_str = self._attempt_llm_bootstrap_heal(initial_failing_packages, error_log)
        
        if not healed_packages_str:
            print("\nINFO: LLM healing failed. Falling back to Deterministic Downgrade Protocol.")
            healed_packages_str = self._attempt_deterministic_bootstrap_heal(initial_failing_packages)

        if healed_packages_str:
            print("\nSUCCESS: Bootstrap Healing Protocol found a stable baseline.")
            with open(self.requirements_path, "w") as f: f.write(healed_packages_str)
            start_group("View Healed and Pinned requirements.txt"); print(healed_packages_str); end_group()
        else:
            sys.exit("CRITICAL ERROR: All bootstrap healing attempts failed. Cannot establish a stable baseline.")
        end_group()

    
    def _run_bootstrap_and_validate(self, venv_dir, requirements_source):
        """
        Installs a set of requirements into a venv and runs the validation script.
        This is a core helper used by both the initial bootstrap and the healing protocols.
        """
        # THE FIX IS HERE: Use .resolve() to get an absolute path for robustness.
        python_executable = str((venv_dir / "bin" / "python").resolve())
        
        # This function is smart: it can take a file path OR a list of packages.
        if isinstance(requirements_source, (Path, str)):
            pip_command = [python_executable, "-m", "pip", "install", "-r", str(requirements_source)]
        else: # It's a list of packages, so we write a temporary file.
            temp_reqs_path = venv_dir / "temp_reqs.txt"
            with open(temp_reqs_path, "w") as f:
                f.write("\n".join(requirements_source))
            pip_command = [python_executable, "-m", "pip", "install", "-r", str(temp_reqs_path)]
            
        _, stderr_install, returncode = run_command(pip_command)
        if returncode != 0:
            return False, None, f"Failed to install dependencies. Error: {stderr_install}"

        # Now, the absolute path is passed to validate_changes.
        success, metrics, validation_output = validate_changes(python_executable, self.config, group_title="Running Validation on New Baseline")
        if not success:
            return False, None, validation_output
            
        installed_packages, _, _ = run_command([python_executable, "-m", "pip", "freeze"])
        return True, {"metrics": metrics, "packages": self._prune_pip_freeze(installed_packages)}, None

   # In agent_logic.py

def run(self):
    if os.path.exists(self.config["METRICS_OUTPUT_FILE"]):
        os.remove(self.config["METRICS_OUTPUT_FILE"])
    
    is_pinned, _ = self._get_requirements_state()
    if not is_pinned:
        self._bootstrap_unpinned_requirements()
        is_pinned, _ = self._get_requirements_state()
        if not is_pinned:
             sys.exit("CRITICAL: Bootstrap process failed to produce a fully pinned requirements file.")

    final_successful_updates = {}
    final_failed_updates = {}
    pass_num = 0
    
    while pass_num < self.config["MAX_RUN_PASSES"]:
        pass_num += 1
        start_group(f"UPDATE PASS {pass_num}/{self.config['MAX_RUN_PASSES']}")
        
        progressive_baseline_path = Path(f"./pass_{pass_num}_progressive_reqs.txt")
        shutil.copy(self.requirements_path, progressive_baseline_path)
        
        changed_packages_this_pass = set()

        # --- Build the initial update plan for this pass ---
        packages_to_update = []
        with open(progressive_baseline_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        for line in lines:
            package_part = line.split(';')[0].strip()
            if '==' not in package_part or line.strip().startswith('-e'): continue
            
            parts = package_part.split('==')
            if len(parts) != 2: continue

            package, current_version = self._get_package_name_from_spec(parts[0]), parts[1]
            latest_version = self.get_latest_version(package)
            if latest_version and parse_version(latest_version) > parse_version(current_version):
                packages_to_update.append((package, current_version, latest_version))

        # --- Handle the two "nothing to do" scenarios correctly ---
        if not packages_to_update:
            if pass_num == 1:
                print("\nInitial baseline is already fully up-to-date.")
                print("Running a final health check on the baseline for confirmation.")
                self._run_final_health_check()
            else:
                print("\nNo further updates are available. The system has successfully converged.")
            
            end_group()
            if progressive_baseline_path.exists(): progressive_baseline_path.unlink()
            break 

        packages_to_update.sort(key=lambda p: self._calculate_update_risk(p[0], p[1], p[2]), reverse=True)
        print("\nPrioritized Update Plan for this Pass:")
        total_updates_in_plan = len(packages_to_update)
        for i, (pkg, cur_ver, target_ver) in enumerate(packages_to_update):
            score = self._calculate_update_risk(pkg, cur_ver, target_ver)
            print(f"  {i+1}/{total_updates_in_plan}: {pkg} (Risk: {score:.2f}) -> {target_ver}")
        
        # --- The Main Healing and Update Loop ---
        for i, (package, current_ver, target_ver) in enumerate(packages_to_update):
            print(f"\n" + "-"*80); print(f"PULSE: [PASS {pass_num} | ATTEMPT {i+1}/{total_updates_in_plan}] Processing '{package}'"); print(f"PULSE: Changed packages this pass so far: {changed_packages_this_pass}"); print("-"*80)
            
            # The call no longer passes `changed_packages_this_pass`
            success, reason_or_new_version, _ = self.attempt_update_with_healing(
                package, current_ver, target_ver, [], 
                progressive_baseline_path
            )
            
            if success:
                # --- START OF CORRECTED LOGIC ---
                
                # Determine the actual version that was reached and if a real change occurred.
                reached_version = ""
                is_a_real_change = False

                if "skipped" in str(reason_or_new_version):
                    # This is the optimization case: validation was skipped because there was no version change.
                    reached_version = current_ver
                    is_a_real_change = False
                else:
                    # An update attempt was made. The new version is in the return variable.
                    reached_version = reason_or_new_version
                    # A real change only happens if the version number is actually different.
                    if current_ver != reached_version:
                        is_a_real_change = True
                
                # Use our clean variables for accurate reporting and state management.
                final_successful_updates[package] = (target_ver, reached_version)
                
                if is_a_real_change:
                    changed_packages_this_pass.add(package)
                    
                    # Update the "living document" for the next attempt in this pass.
                    print(f"  -> SUCCESS. Locking in {package}=={reached_version} into the progressive baseline for this pass.")
                    with open(progressive_baseline_path, "r") as f:
                        temp_lines = f.readlines()
                    with open(progressive_baseline_path, "w") as f:
                        for line in temp_lines:
                            if self._get_package_name_from_spec(line.split(';')[0]) == package:
                                marker_part = ""
                                if ";" in line:
                                    marker_part = " ;" + line.split(";", 1)[1]
                                f.write(f"{package}=={reached_version}{marker_part}\n")
                            else:
                                f.write(line)
                
                # --- END OF CORRECTED LOGIC ---

            else: # if success is False
                final_failed_updates[package] = (target_ver, reason_or_new_version)
        
        end_group()

        # --- Promote the "living document" to the new Golden Record ---
        if changed_packages_this_pass:
            print("\nPass complete with changes")
            shutil.copy(progressive_baseline_path, self.requirements_path)
        
        if progressive_baseline_path.exists():
            progressive_baseline_path.unlink()

        # The final convergence check.
        if not changed_packages_this_pass:
            print("\nNo effective version changes were possible in this pass. The system has converged.")
            break
    
    if final_successful_updates:
        self._print_final_summary(final_successful_updates, final_failed_updates)
        self._run_final_health_check()

    def _apply_pass_updates(self, successful_updates, baseline_reqs_path):
        """
        Takes the successful updates from a pass and creates a new, frozen requirements.txt.
        This is the commit part of the transactional update logic.
        """
        print("\nApplying successful changes from this pass...")
        venv_dir = Path("./temp_venv")
        if venv_dir.exists(): shutil.rmtree(venv_dir)
        venv.create(venv_dir, with_pip=True)
        python_executable = str((venv_dir / "bin" / "python").resolve())
        
        with open(baseline_reqs_path, "r") as f_read:
            lines = [line.strip() for line in f_read if line.strip()]

        for package, new_version in successful_updates.items():
             lines = [f"{package}=={new_version}" if self._get_package_name_from_spec(l) == package else l for l in lines]
        
        temp_reqs_path = venv_dir / "final_pass_reqs.txt"
        with open(temp_reqs_path, "w") as f_write:
            f_write.write("\n".join(lines))
        
        # Install the full updated set and freeze it to capture any new transitive dependencies correctly
        _, stderr, returncode = run_command([python_executable, "-m", "pip", "install", "-r", str(temp_reqs_path)])
        if returncode != 0:
            print(f"CRITICAL: Failed to install combined updates at end of pass. Error: {stderr}", file=sys.stderr)
            # If the final combination fails, we revert to the baseline from the start of the pass for safety.
            shutil.copy(baseline_reqs_path, self.requirements_path)
            return

        final_packages, _, _ = run_command([python_executable, "-m", "pip", "freeze"])
        with open(self.requirements_path, "w") as f:
            f.write(self._prune_pip_freeze(final_packages))
        print("Successfully applied and froze all successful updates for this pass.")

    def _calculate_update_risk(self, package, current_ver, target_ver):
        usage = self.usage_scores.get(package, 0)
        is_primary = 1 if package in self.primary_packages else 0
        try:
            old_v, new_v = parse_version(current_ver), parse_version(target_ver)
            if new_v.major > old_v.major: semver_severity = 3
            elif new_v.minor > old_v.minor: semver_severity = 2
            else: semver_severity = 1
        except: semver_severity = 1
        return (usage * 5.0) + (is_primary * 3.0) + (semver_severity * 2.0)

    def _print_final_summary(self, successful, failed):
        print("\n" + "#"*70); print("### OVERALL UPDATE RUN SUMMARY ###")
        if successful:
            print("\n[SUCCESS] The following packages were successfully updated:")
            print(f"{'Package':<30} | {'Target Version':<20} | {'Reached Version':<20}")
            print(f"{'-'*30} | {'-'*20} | {'-'*20}")
            for pkg, (target_ver, version) in successful.items(): print(f"{pkg:<30} | {target_ver:<20} | {version:<20}")
        if failed:
            print("\n[FAILURE] Updates were attempted but FAILED for:")
            print(f"{'Package':<30} | {'Target Version':<20} | {'Reason for Failure'}")
            print(f"{'-'*30} | {'-'*20} | {'-'*40}")
            for pkg, (target_ver, reason) in failed.items(): print(f"{pkg:<30} | {target_ver:<20} | {reason}")
        print("#"*70 + "\n")

    def _run_final_health_check(self):
        print("\n" + "#"*70); print("### FINAL SYSTEM HEALTH CHECK ###"); print("#"*70 + "\n")
        venv_dir = Path("./final_venv")
        if venv_dir.exists(): shutil.rmtree(venv_dir)
        venv.create(venv_dir, with_pip=True)
        
        # *** THE FIX IS HERE: Use .resolve() to get the absolute path. ***
        python_executable = str((venv_dir / "bin" / "python").resolve())

        # First, install the dependencies using the final requirements file.
        _, stderr, returncode = run_command([python_executable, "-m", "pip", "install", "-r", str(self.requirements_path)])
        if returncode != 0:
            print("CRITICAL ERROR: Final installation of combined dependencies failed!", file=sys.stderr); return

        # Then, run the validation using the newly created environment.
        success, metrics, _ = validate_changes(python_executable, self.config, group_title="Final System Health Check")
        
        if success and metrics and "not available" not in metrics:
            print("\n" + "="*70); print("=== FINAL METRICS FOR THE FULLY UPDATED ENVIRONMENT ===")
            indented_metrics = "\n".join([f"  {line}" for line in metrics.split('\n')])
            print(indented_metrics); print("="*70)
        elif success:
            print("\n" + "="*70); print("=== Final validation passed, but metrics were not available in output. ==="); print("="*70)
        else:
            print("\n" + "!"*70); print("!!! CRITICAL ERROR: Final validation of combined dependencies failed! !!!"); print("!"*70)

    def get_latest_version(self, package_name):
        try:
            package_info = self.pypi.get_project_page(package_name)
            if not (package_info and package_info.packages): return None
            stable_versions = [p.version for p in package_info.packages if p.version and not parse_version(p.version).is_prerelease]
            return max(stable_versions, key=parse_version) if stable_versions else max([p.version for p in package_info.packages if p.version], key=parse_version)
        except Exception: return None

    # In agent_logic.py

def _try_install_and_validate(self, package_to_update, new_version, dynamic_constraints, baseline_reqs_path, is_probe):
    """
    Creates a temporary environment to test a single package update.
    It runs a robust installation and then proceeds to validation.
    This version includes an optimization to skip redundant validation runs.
    """
    venv_dir = Path("./temp_venv")
    if venv_dir.exists(): shutil.rmtree(venv_dir)
    venv.create(venv_dir, with_pip=True)
    python_executable = str((venv_dir / "bin" / "python").resolve())
    
    # Determine the 'old_version' from the current progressive baseline for comparison.
    old_version = "N/A"
    with open(baseline_reqs_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            # Ensure we're checking the correct package, ignoring markers
            if self._get_package_name_from_spec(line.split(';')[0]) == package_to_update:
                if '==' in line:
                    old_version = line.split(';')[0].strip().split('==')[1]
                    break
    
    # --- START OF OPTIMIZATION LOGIC ---
    # If the new version is the same as the old version in the baseline,
    # it means no change is being made. The environment state is identical to the
    # last known-good state, so we can skip the expensive validation step.
    if new_version == old_version:
        # We only print the verbose analysis if it's not a quick probe from the backtrack algorithm
        if not is_probe:
             print(f"--> Change analysis: '{package_to_update}' version remains at {old_version}.")
             print("--> Validation skipped (no change).")
        # Return True because staying at the current version is a successful and valid outcome.
        # The special string in the 'reason' field signals to the caller that no update was made.
        return True, "Validation skipped (no change)", ""

    if not is_probe:
        print(f"\nChange analysis: Updating '{package_to_update}' from {old_version} -> {new_version}. Validation is required.")
    # --- END OF OPTIMIZATION LOGIC ---

    temp_reqs_path = venv_dir / "temp_requirements.txt"
    
    # Create the temporary, modified requirements file for this specific test run.
    with open(baseline_reqs_path, "r") as f_read, open(temp_reqs_path, "w") as f_write:
        lines_for_file = []
        for line in f_read:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            package_part = line.split(';')[0].strip()
            
            if self._get_package_name_from_spec(package_part) == package_to_update:
                marker_part = ""
                if ";" in line:
                    marker_part = " ;" + line.split(";", 1)[1]
                lines_for_file.append(f"{package_to_update}=={new_version}{marker_part}")
            else:
                lines_for_file.append(line)

        for constraint in dynamic_constraints:
             if self._get_package_name_from_spec(constraint) != package_to_update:
                lines_for_file.append(constraint)

        f_write.write("\n".join(lines_for_file))

    # --- STEP 1: The Reliable, Robust Install Attempt ---
    _, stderr_install, returncode = run_command([python_executable, "-m", "pip", "install", "-r", str(temp_reqs_path)])
    
    # --- STEP 2: The Diagnostic Phase (if Step 1 failed) ---
    if returncode != 0:
        print("INFO: Main installation failed. Retrying with verbose logging to identify conflicting packages...")
        
        with open(temp_reqs_path, 'r') as f:
            requirements_list_for_log = [line.strip() for line in f if line.strip()]
        pip_command_for_logs = [python_executable, "-m", "pip", "install"] + requirements_list_for_log
        _, stderr_for_logs, _ = run_command(pip_command_for_logs)
        
        conflict_match = re.search(r"Cannot install(?P<packages>[\s\S]+?)because", stderr_for_logs)
        
        reason = ""
        if conflict_match:
            conflicting_packages = ' '.join(conflict_match.group('packages').split())
            conflicting_packages = conflicting_packages.replace(' and ', ', ').replace(',', ', ')
            reason = f"Conflict between packages: {conflicting_packages}"
            print(f"DIAGNOSIS: {reason}")
        else:
            print("DIAGNOSIS: Could not parse specific conflicts. Falling back to LLM summary.")
            llm_summary = self._ask_llm_to_summarize_error(stderr_install)
            reason = f"Installation conflict. Summary: {llm_summary}"
        
        return False, reason, stderr_install
    
    # --- STEP 3: Validation of the new environment ---
    group_title = f"Validation for {package_to_update}=={new_version}"
    success, metrics, validation_output = validate_changes(python_executable, self.config, group_title=group_title)

    if not success:
        return False, "Validation script failed", validation_output
    
    return True, metrics, ""
    

    def attempt_update_with_healing(self, package, current_version, target_version, dynamic_constraints, baseline_reqs_path, changed_packages_this_pass):
        """
        Attempts to update a package and intelligently chooses a healing strategy
        based on the type of failure (Installation vs. Validation).
        This is the main "Triage" function.
        """
        print(f"\nAttempting to validate {package}=={target_version}...")
        
        success, result_data, stderr = self._try_install_and_validate(
            package, target_version, dynamic_constraints, baseline_reqs_path, 
            is_probe=False, changed_packages=changed_packages_this_pass
        )
        
        if success:
            print(f"Direct update to {package}=={target_version} succeeded.")
            return True, result_data if "skipped" in str(result_data) else target_version, None

        # --- THIS IS THE CORRECTED TRIAGE LOGIC ---
        # We determine the failure type based on the content of the "reason" string.
        failure_type = "ValidationConflict" if "Validation script failed" in str(result_data) else "InstallationConflict"
        
        print(f"\nINFO: Initial update for '{package}' failed. Reason: '{result_data}'")
        start_group("View Full Error Log for Initial Failure"); print(stderr); end_group()
        print(f"DIAGNOSIS: Detected a {failure_type}.")
        print("INFO: Entering specialized healing mode.")
        
        healed_version = None
        if failure_type == "InstallationConflict":
            print("  -> Strategy: Using LLM-first approach with binary search fallback.")
            healed_version = self._heal_with_llm_first(
                package, current_version, target_version, dynamic_constraints, 
                baseline_reqs_path, changed_packages_this_pass, stderr  # Correctly passing stderr
            )
        else: # ValidationConflict
            print("  -> Strategy: Using binary search backtrack for runtime/validation failure.")
            healed_version = self._intelligent_backtrack(
                package, current_version, target_version, dynamic_constraints, 
                baseline_reqs_path, changed_packages_this_pass
            )

        if healed_version:
            return True, healed_version, None
        
        return False, f"All healing attempts for {package} failed.", None
    

    def _intelligent_backtrack(self, package, last_good_version, failed_version, dynamic_constraints, baseline_reqs_path, changed_packages):
        start_group(f"Healing Backtrack for {package} (Intelligent Bisection Search)")
        
        versions = self.get_all_versions_between(package, last_good_version, failed_version)
        
        # Use a class member to store the best version found across all recursive calls.
        self.best_working_version = last_good_version
        
        if not versions:
            print("Intelligent Search: No intermediate versions to test.")
            return self.best_working_version

        print(f"Intelligent Search: Searching {len(versions)} versions using Backtracking Bisection...")
        
        # --- YOUR BRILLIANT RECURSIVE ALGORITHM ---
        def search(sub_list):
            if not sub_list:
                return

            mid_index = len(sub_list) // 2
            test_version = sub_list[mid_index]
            
            if "tested_versions" not in self.__dict__: self.tested_versions = set()
            if test_version in self.tested_versions: return
            self.tested_versions.add(test_version)

            print(f"  -> Probing version {test_version}...")
            success, reason, _ = self._try_install_and_validate(
                package, test_version, dynamic_constraints, baseline_reqs_path, True, changed_packages
            )
            
            if success:
                print(f"  -- SUCCESS: Version {test_version} is a new candidate.")
                if parse_version(test_version) > parse_version(self.best_working_version):
                    self.best_working_version = test_version
                
                # YOUR INSIGHT: Aggressively search the newer half, discard the left.
                search(sub_list[mid_index + 1:])
            else:
                print(f"  -- FAILURE: Version {test_version} failed. Reason: {reason}")
                # First, aggressively check the newer half for islands.
                search(sub_list[mid_index + 1:])
                # Then, backtrack to the older half.
                search(sub_list[:mid_index])

        self.tested_versions = set()
        search(versions)

        end_group()

        print(f"Intelligent Search COMPLETE: The definitive latest stable version is {self.best_working_version}")
        return self.best_working_version
    

    def get_all_versions_between(self, package_name, start_ver_str, end_ver_str):
        try:
            package_info = self.pypi.get_project_page(package_name)
            if not (package_info and package_info.packages): return []
            start_v, end_v = parse_version(start_ver_str), parse_version(end_ver_str)
            candidate_versions = [v for p in package_info.packages if p.version and start_v <= (v := parse_version(p.version)) < end_v]
            return sorted([str(v) for v in set(candidate_versions)], key=parse_version)
        except Exception: return []

    def _ask_llm_to_summarize_error(self, error_message):
        if not self.llm_available: return "(LLM unavailable due to quota)"
        prompt = f"The following is a Python pip install error log. Please summarize the root cause of the conflict in a single, concise sentence. Error Log: --- {error_message} ---"
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip().replace('\n', ' ')
        except Exception: return "Failed to get summary from LLM."
            
    def _prune_pip_freeze(self, freeze_output):
        lines = freeze_output.strip().split('\n')
        return "\n".join([line for line in lines if '==' in line and not line.startswith('-e')])

    def _heal_with_llm_first(self, package, current_version, target_version, dynamic_constraints, baseline_reqs_path, changed_packages, error_log):
        """
        Implements your "Conversational Correction" loop with a "Reality Check".
        It tries to get a valid suggestion from the LLM, verifies it against PyPI,
        and then falls back to the definitive intelligent backtrack.
        """
        print("  -> Phase 1: Collaborating with LLM for intelligent version candidates.")
        
        all_real_versions = self._get_all_available_versions(package)
        if not all_real_versions:
            print("  -> LLM SKIPPED: Could not retrieve any available versions from PyPI.")
            return self._intelligent_backtrack(package, current_version, target_version, dynamic_constraints, baseline_reqs_path, changed_packages)

        verified_candidates = []
        conversation_history = []
        for attempt in range(3):
            suggestions = self._ask_llm_for_install_candidates(
                package, current_version, target_version, error_log, baseline_reqs_path, conversation_history
            )
            
            verified_candidates = [v for v in suggestions if v in all_real_versions]
            
            if verified_candidates:
                break
            else:
                print("  -> LLM WARNING: All suggestions were invalid (hallucinated). Retrying with corrective feedback.")
                feedback = "Your previous suggestions were not found on PyPI. Please try again, suggesting only real, existing, stable versions."
                conversation_history.append({"role": "model", "parts": [str(suggestions)]})
                conversation_history.append({"role": "user", "parts": [feedback]})
        
        if verified_candidates:
            print(f"  -> INFO: LLM provided VERIFIED suggestions: {verified_candidates}")
            for candidate in sorted(verified_candidates, key=parse_version, reverse=True):
                if not (parse_version(current_version) < parse_version(candidate) < parse_version(target_version)):
                    continue
                
                print(f"  -> Attempting VERIFIED LLM suggestion: {package}=={candidate}")
                success, _, _ = self._try_install_and_validate(
                    package, candidate, dynamic_constraints, baseline_reqs_path, False, changed_packages
                )
                if success:
                    print(f"  -> SUCCESS: LLM-suggested version {candidate} was correct.")
                    return candidate
        
        print("  -> Phase 2: LLM collaboration failed. Falling back to the definitive Intelligent Backtrack search.")
        return self._intelligent_backtrack(
            package, current_version, target_version, dynamic_constraints, baseline_reqs_path, changed_packages
        )

    def _ask_llm_for_install_candidates(self, package, current_version, failed_version, error_message, baseline_reqs_path, conversation_history):
        """
        Asks the LLM to act as an expert resolver and suggest version candidates.
        It now supports a multi-turn conversation to correct hallucinations.
        """
        if not self.llm_available: return []
        
        try:
            with open(baseline_reqs_path, "r") as f:
                current_requirements = f.read()
        except FileNotFoundError:
            current_requirements = "Not available."

        # The initial system prompt is now separated from the user query
        initial_prompt = f"""You are AURA, an expert Autonomous Universal Refinement Agent specializing in Python dependency conflict resolution. Your performance is being benchmarked against a simple Reverse Linear Scan. You must be smarter and more accurate. Your suggestion should be the latest possible working version.
        
        A user has a stable environment defined by the following `requirements.txt`:
        ---
        {current_requirements}
        ---
        
        They attempted to upgrade '{package}' from '{current_version}' to '{failed_version}', but it failed with this error:
        ---
        {error_message}
        ---

        Your task is to suggest a list of up to 3 specific, stable versions of '{package}' that are most likely to resolve this specific conflict. The versions MUST be newer than '{current_version}'. Respond ONLY with a valid Python list of strings in descending order.
        Example response: ["2.12.0", "2.11.9", "2.11.8"]
        """
        
        # Build the full conversation for the API call
        full_conversation = [{"role": "user", "parts": [initial_prompt]}] + conversation_history

        try:
            # Use the more advanced `start_chat` for multi-turn conversations
            chat = self.llm.start_chat(history=full_conversation[:-1])
            response = chat.send_message(full_conversation[-1], request_options={"timeout": 180})
            
            match = re.search(r'(\[.*\])', response.text, re.DOTALL)
            if not match: 
                print(f"  -> LLM WARNING: Response was not a valid list. Response: {response.text}")
                return []
            
            candidates = ast.literal_eval(match.group(0))
            if not all(isinstance(c, str) for c in candidates): return []
            
            print(f"  -> LLM provided {len(candidates)} raw suggestions: {candidates}")
            return candidates
        except Exception as e:
            print(f"  -> LLM ERROR: Failed to get/parse version candidates: {type(e).__name__}")
            return []
        
    def _get_all_available_versions(self, package_name: str) -> set[str]:
        """Gets a set of all available, non-prerelease versions for a package from PyPI."""
        try:
            page = self.pypi.get_project_page(package_name)
            if not (page and page.packages):
                return set()
            
            versions = {
                p.version for p in page.packages
                if p.version and not parse_version(p.version).is_prerelease
            }
            return versions
        except Exception:
            return set()