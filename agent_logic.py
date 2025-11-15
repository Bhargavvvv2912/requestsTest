# agent_logic.py

import os
import sys
import venv
from pathlib import Path
import ast
import shutil
import re
import json
import subprocess
from google.api_core.exceptions import ResourceExhausted
from pypi_simple import PyPISimple
from packaging.version import parse as parse_version
from agent_utils import start_group, end_group, run_command, validate_changes
from expert_agent import ExpertAgent

class DependencyAgent:
    def __init__(self, config, llm_client):
        self.config = config
        self.expert = ExpertAgent(llm_client) 
        self.pypi = PyPISimple()
        self.requirements_path = Path(config["REQUIREMENTS_FILE"])
        self.primary_packages = self._load_primary_packages()
        self.llm_available = True
        self.usage_scores = self._calculate_risk_scores()
        self.exclusions_from_this_run = set()
    
    # In agent_logic.py

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

    def _calculate_update_risk_components(self, package, current_ver_str, target_ver_str):
        """Calculates the raw, unweighted components of the HURM 4.0 risk score."""
        try:
            old_v, new_v = parse_version(current_ver_str), parse_version(target_ver_str)
            if new_v.major > old_v.major: severity_score = 10
            elif new_v.minor > old_v.minor: severity_score = 5
            else: severity_score = 1
        except Exception:
            severity_score = 5

        usage_score = self.usage_scores.get(package, 0)
        criticality_score = 1 if package in self.primary_packages else 0
        
        # This part requires you to have a pre-computed dependency graph
        if hasattr(self, 'dependency_graph_metrics') and package in self.dependency_graph_metrics:
            ecosystem_score = self.dependency_graph_metrics[package].get('dependents', 0)
            depth_score = self.dependency_graph_metrics[package].get('depth', 0)
        else:
            ecosystem_score, depth_score = 0, 0
            
        return {
            "severity": severity_score, "usage": usage_score, "criticality": criticality_score,
            "ecosystem": ecosystem_score, "depth": depth_score
        }

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

        print("\nCRITICAL: Initial baseline failed validation. Cannot proceed.", file=sys.stderr)
        start_group("View Initial Baseline Failure Log"); print(error_log); end_group()
        sys.exit("CRITICAL ERROR: Bootstrap failed. Please provide a working set of requirements.")
    
    def _run_bootstrap_and_validate(self, venv_dir, requirements_source):
        python_executable = str((venv_dir / "bin" / "python").resolve())
        
        if isinstance(requirements_source, (Path, str)):
            pip_command = [python_executable, "-m", "pip", "install", "-r", str(requirements_source)]
        else:
            temp_reqs_path = venv_dir / "temp_reqs.txt"
            with open(temp_reqs_path, "w") as f: f.write("\n".join(requirements_source))
            pip_command = [python_executable, "-m", "pip", "install", "-r", str(temp_reqs_path)]
            
        _, stderr_install, returncode = run_command(pip_command)
        if returncode != 0:
            return False, None, f"Failed to install dependencies. Error: {stderr_install}"

        success, metrics, validation_output = validate_changes(python_executable, self.config, group_title="Running Validation on New Baseline")
        if not success:
            return False, None, validation_output
            
        installed_packages, _, _ = run_command([python_executable, "-m", "pip", "freeze"])
        return True, {"metrics": metrics, "packages": self._prune_pip_freeze(installed_packages)}, None

   # In agent_logic.py

    def run(self):
        if os.path.exists(self.config["METRICS_OUTPUT_FILE"]):
            os.remove(self.config["METRICS_OUTPUT_FILE"])
        
        # --- START OF "REPAIR THE BASELINE" ARCHITECTURE ---
        print("INFO: Performing Initial Baseline Health Check...")
        start_group("Initial Baseline Health Check")
        
        venv_dir = Path("./initial_check_venv")
        if venv_dir.exists(): shutil.rmtree(venv_dir)
        venv.create(venv_dir, with_pip=True)
        
        success, _, error_log = self._run_bootstrap_and_validate(venv_dir, self.requirements_path)
        end_group()
        
        if not success:
            print("WARNING: Initial baseline is broken. Activating 'Repair Mode'.", file=sys.stderr)
            repair_succeeded = self._run_repair_mode(error_log)
            if not repair_succeeded:
                sys.exit("CRITICAL ERROR: The initial baseline is broken and could not be automatically repaired.")
            else:
                print("\nINFO: Baseline successfully repaired. Proceeding to normal update pass.")
        else:
            print("Initial baseline is valid and stable.")
        # --- END OF "REPAIR THE BASELINE" ARCHITECTURE ---

        final_successful_updates, final_failed_updates = {}, {}
        pass_num = 0
        
        if not hasattr(self, 'dependency_graph_metrics'):
            print("INFO: Dependency graph metrics not found. Entanglement scores will be 0.")
            self.dependency_graph_metrics = {}

        while pass_num < self.config["MAX_RUN_PASSES"]:
            pass_num += 1
            start_group(f"UPDATE PASS {pass_num}/{self.config['MAX_RUN_PASSES']}")
            
            progressive_baseline_path = Path(f"./pass_{pass_num}_progressive_reqs.txt")
            shutil.copy(self.requirements_path, progressive_baseline_path)
            changed_packages_this_pass = set()

            packages_to_update = []
            with open(progressive_baseline_path, 'r') as f: lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            for line in lines:
                if '==' not in line or line.startswith('-e'): continue
                package, current_version = self._get_package_name_from_spec(line.split('==')[0]), line.split('==')[1].split(';')[0]
                latest_version = self.get_latest_version(package)
                if latest_version and parse_version(latest_version) > parse_version(current_version):
                    packages_to_update.append((package, current_version, latest_version))

            if not packages_to_update:
                print("\nNo further updates are available. The system has successfully converged.")
                if progressive_baseline_path.exists(): progressive_baseline_path.unlink()
                break 

            update_plan = []
            for pkg, cur, target in packages_to_update:
                components = self._calculate_update_risk_components(pkg, cur, target)
                update_plan.append({'pkg': pkg, 'cur': cur, 'target': target, 'components': components})
            max_ecosystem = max(p['components']['ecosystem'] for p in update_plan) if update_plan else 0
            max_depth = max(p['components']['depth'] for p in update_plan) if update_plan else 0
            W_ECOSYSTEM, W_DEPTH = 1.0, 0.5
            for p in update_plan:
                comps = p['components']; norm_ecosystem = (comps['ecosystem'] / max_ecosystem) if max_ecosystem > 0 else 0
                norm_depth = (comps['depth'] / max_depth) if max_depth > 0 else 0
                entanglement_score = (W_ECOSYSTEM * norm_ecosystem) + (W_DEPTH * norm_depth)
                p['final_score'] = (comps['severity'] * 10) + entanglement_score
                p['code_impact_score'] = comps['usage'] + (comps['criticality'] * 10)
            
            update_plan.sort(key=lambda p: (p['final_score'], p['code_impact_score']), reverse=True)
            
            total_score_sum = sum(p['final_score'] for p in update_plan) if update_plan else 0
            for p in update_plan:
                 if total_score_sum == 0: p['risk_percent_display'] = 0.0
                 else: p['risk_percent_display'] = (p['final_score'] / total_score_sum) * 100.0

            print("\nPrioritized Update Plan for this Pass (Highest Risk First):")
            print(f"{'Rank':<5} | {'Package':<30} | {'% of Total Risk':<18} | {'Change'}")
            print(f"{'-'*5} | {'-'*30} | {'-'*18} | {'-'*20}")
            for i, p in enumerate(update_plan): print(f"{i+1:<5} | {p['pkg']:<30} | {p['risk_percent_display']:<18.2f}% | {p['cur']} -> {p['target']}")
            
            for i, p_data in enumerate(update_plan):
                package, current_ver, target_ver = p_data['pkg'], p_data['cur'], p_data['target']
                print(f"\n" + "-"*80); print(f"PULSE: [PASS {pass_num} | ATTEMPT {i+1}/{len(update_plan)}] Processing '{package}'"); print(f"PULSE: Changed packages this pass so far: {changed_packages_this_pass}"); print("-"*80)
                
                success, reason_or_new_version, _ = self.attempt_update_with_healing(package, current_ver, target_ver, [], progressive_baseline_path)
                
                if success:
                    reached_version, is_a_real_change = "", False
                    if "skipped" in str(reason_or_new_version): reached_version, is_a_real_change = current_ver, False
                    else:
                        reached_version = reason_or_new_version
                        if current_ver != reached_version: is_a_real_change = True
                    final_successful_updates[package] = (target_ver, reached_version)
                    if is_a_real_change:
                        changed_packages_this_pass.add(package)
                        print(f"  -> SUCCESS. Locking in {package}=={reached_version} into the progressive baseline for this pass.")
                        with open(progressive_baseline_path, "r") as f: temp_lines = f.readlines()
                        with open(progressive_baseline_path, "w") as f:
                            for line in temp_lines:
                                if self._get_package_name_from_spec(line.split(';')[0]) == package:
                                    marker_part = f" ;{line.split(';')[1]}" if ';' in line else ""
                                    f.write(f"{package}=={reached_version}{marker_part}\n")
                                else: f.write(line)
                else:
                    final_failed_updates[package] = (target_ver, reason_or_new_version)
            
            end_group()

            if changed_packages_this_pass:
                print("\nPass complete with changes.")
                shutil.copy(progressive_baseline_path, self.requirements_path)
            if progressive_baseline_path.exists(): progressive_baseline_path.unlink()
            if not changed_packages_this_pass:
                print("\nNo effective version changes were possible in this pass. The system has converged.")
                break
        
        if final_successful_updates:
            self._print_final_summary(final_successful_updates, final_failed_updates)
            print("\nRun complete. The final requirements.txt is the latest provably working version.")
    

    # In agent_logic.py

    def _run_repair_mode(self, error_log):
        """
        Attempts a targeted, "surgical" repair of a broken baseline by letting pip
        re-resolve only the conflicting packages.
        """
        start_group("REPAIR MODE: Attempting surgical re-resolution of baseline")
        
        # --- Step 1: Diagnose the conflicting packages using the Expert Agent ---
        print("--> Step 1: Delegating diagnosis to the Expert Agent (CORE)...")
        conflicting_packages = self.expert.diagnose_conflict_from_log(error_log)
        
        if not conflicting_packages:
            print("--> DIAGNOSIS FAILED: Could not determine conflicting packages. Aborting repair.", file=sys.stderr)
            end_group()
            return False

        print(f"--> Expert Diagnosis: Conflict involves {conflicting_packages}")

        # --- Step 2: Prepare the surgical pip install command ---
        print("\n--> Step 2: Preparing surgical installation plan...")
        
        good_reqs_path = Path("./good_reqs.txt")
        unpinned_conflicts = []
        
        with open(self.requirements_path, 'r') as f_read, open(good_reqs_path, 'w') as f_write:
            for line in f_read:
                pkg_name = self._get_package_name_from_spec(line)
                if pkg_name and pkg_name.lower() in (p.lower() for p in conflicting_packages):
                    print(f"     - Unpinning '{pkg_name}' for re-resolution.")
                    unpinned_conflicts.append(pkg_name)
                else:
                    f_write.write(line)

        # --- Step 3: Execute the repair and validation in a single bootstrap run ---
        print("\n--> Step 3: Attempting to install the repaired dependency set...")
        
        venv_dir = Path("./repair_check_venv")
        if venv_dir.exists(): shutil.rmtree(venv_dir)
        venv.create(venv_dir, with_pip=True)
        python_executable = str((venv_dir / "bin" / "python").resolve())
        project_dir = self.config.get("VALIDATION_CONFIG", {}).get("project_dir")

        # First, install the "good" pinned dependencies
        pip_command_good = [python_executable, "-m", "pip", "install", "-r", str(good_reqs_path.resolve())]
        _, stderr_good, returncode_good = run_command(pip_command_good, cwd=project_dir)
        if returncode_good != 0:
            print(f"--> REPAIR FAILED: Could not install the 'good' part of the requirements.", file=sys.stderr)
            print(stderr_good, file=sys.stderr)
            end_group(); return False

        # Now, ask pip to install the unpinned, conflicting packages, letting it resolve them.
        pip_command_bad = [python_executable, "-m", "pip", "install"] + unpinned_conflicts
        _, stderr_bad, returncode_bad = run_command(pip_command_bad, cwd=project_dir)
        if returncode_bad != 0:
            print(f"--> REPAIR FAILED: pip could not find a compatible resolution for the conflicting packages.", file=sys.stderr)
            print(stderr_bad, file=sys.stderr)
            end_group(); return False

        print("--> Installation of repaired set was successful.")
        
        # --- Step 4: Validate the new environment and save the result ---
        print("\n--> Step 4: Validating the new, repaired environment...")
        success, metrics, validation_output = validate_changes(python_executable, self.config, group_title="Validating Repaired Baseline")
        
        if success:
            print("--> REPAIR SUCCESSFUL! The baseline is now provably working.")
            # Freeze the newly solved environment to create the new golden record.
            installed_packages_output, _, _ = run_command([python_executable, "-m", "pip", "freeze"])
            # (Assuming you have a _prune_pip_freeze helper)
            repaired_packages = self._prune_pip_freeze(installed_packages_output)
            with open(self.requirements_path, "w") as f:
                f.write(repaired_packages)
            end_group()
            return True
        else:
            print("--> REPAIR FAILED: The repaired environment failed the validation suite.", file=sys.stderr)
            print(validation_output, file=sys.stderr)
            end_group()
            return False
        
    def _get_conflicting_packages_from_log(self, error_log):
        """A helper to extract package names from a pip 'ResolutionImpossible' error."""
        # This regex is designed to find package names in "Cannot install A, B, and C" messages.
        match = re.search(r"Cannot install ([\s\S]+?) because", error_log)
        if not match: return []
        
        # Split by comma and 'and', then clean up and extract the name
        raw_packages = match.group(1).replace(' and ', ',').split(',')
        packages = [self._get_package_name_from_spec(p.strip()) for p in raw_packages]
        return [p for p in packages if p] # Filter out any None values

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

    def get_latest_version(self, package_name):
        try:
            package_info = self.pypi.get_project_page(package_name)
            if not (package_info and package_info.packages): return None
            stable_versions = [p.version for p in package_info.packages if p.version and not parse_version(p.version).is_prerelease]
            return max(stable_versions, key=parse_version) if stable_versions else max([p.version for p in package_info.packages if p.version], key=parse_version)
        except Exception: return None

    def _try_install_and_validate(self, package_to_update, new_version, dynamic_constraints, baseline_reqs_path, is_probe):
        start_group(f"Probe: Attempting install & validation for {package_to_update}=={new_version}")
        
        venv_dir = Path("./temp_venv")
        if venv_dir.exists(): shutil.rmtree(venv_dir)
        venv.create(venv_dir, with_pip=True)
        python_executable = str((venv_dir / "bin" / "python").resolve())
        
        old_version = "N/A"
        with open(baseline_reqs_path, "r") as f:
            for line in f:
                if self._get_package_name_from_spec(line.split(';')[0]) == package_to_update and '==' in line:
                    old_version = line.split(';')[0].strip().split('==')[1]
                    break
        
        if new_version == old_version:
            print(f"--> Version is unchanged ({old_version}). Skipping probe.")
            end_group()
            return True, "Validation skipped (no change)", ""

        print(f"--> Preparing test environment: Updating '{package_to_update}' from {old_version} to {new_version}")

        # --- START OF CRITICAL CHANGE: Pass requirements directly to command line ---
        requirements_list = []
        with open(baseline_reqs_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if self._get_package_name_from_spec(line.split(';')[0]) == package_to_update:
                    marker_part = f" ;{line.split(';')[1]}" if ';' in line else ""
                    requirements_list.append(f"{package_to_update}=={new_version}{marker_part}")
                else:
                    requirements_list.append(line)
        
        # This command is now much cleaner and will produce better error messages
        pip_command = [python_executable, "-m", "pip", "install"] + requirements_list
        # --- END OF CRITICAL CHANGE ---

        _, stderr_install, returncode = run_command(pip_command)
        
        if returncode != 0:
            print("--> ERROR: Installation failed. Analyzing conflict...")
            # The error from pip is now directly readable, so we can use a simpler regex
            conflict_match = re.search(r"because these package versions have conflicting dependencies.\s*([\s\S]*)The conflict is caused by:", stderr_install, re.MULTILINE)
            reason = "Installation conflict"
            if conflict_match:
                try:
                    conflict_lines = conflict_match.group(1).strip().split('\n')
                    packages_involved = [line.split(' ')[0].strip() for line in conflict_lines if line.strip()]
                    reason = f"Conflict involves: {', '.join(packages_involved)}"
                except Exception:
                    reason = "Conflict (Could not parse details)"
            else: # Fallback for other error types
                 summary = self._get_error_summary(stderr_install)
                 reason = f"Installation failed: {summary}"

            print(f"--> DIAGNOSIS: {reason}")
            end_group()
            return False, reason, stderr_install
        
        print("--> Installation successful. Running validation suite...")
        success, metrics, validation_output = validate_changes(python_executable, self.config, group_title=f"Running Validation on {package_to_update}=={new_version}")

        if not success:
            print("--> ERROR: Validation script failed.")
            end_group()
            return False, "Validation script failed", validation_output

        print("--> SUCCESS: Installation and validation passed.")
        end_group()
        return True, metrics, ""

    def attempt_update_with_healing(self, package, current_version, target_version, dynamic_constraints, baseline_reqs_path):
        """
        The "Triage" function. Now implements the Tiered Healing strategy.
        Level 1: Deterministic Filter-Then-Scan.
        Level 2: Escalates to the Expert Agent for Co-Resolution.
        """
        print(f"\n--> Toplevel Attempt: Trying direct update to {package}=={target_version}")
        
        success, result_data, stderr = self._try_install_and_validate(
            package, target_version, dynamic_constraints, baseline_reqs_path, is_probe=False
        )
        
        if success:
            print(f"--> Toplevel Result: Direct update SUCCEEDED.")
            return True, result_data if "skipped" in str(result_data) else target_version, None

        print(f"\n--> Toplevel Result: Direct update FAILED. Reason: '{result_data}'")
        
        # --- LEVEL 1 HEALING ---
        print("--> Action: Entering Level 1 Healing with 'Filter-Then-Scan' strategy.")
        healed_version = self._heal_with_filter_and_scan(package, current_version, target_version, baseline_reqs_path)
        
        if healed_version and healed_version != current_version:
             print(f"--> Healing Result: Level 1 succeeded. New working version: {healed_version}")
             return True, healed_version, None

        # --- ESCALATION TO LEVEL 2 ---
        print(f"\n--> Action: Level 1 Healing failed. Escalating to Level 2 'Co-Resolution Expert'.")
        
        # AURA gathers information for the expert.
        available_updates = self.get_available_updates_from_plan() # You'll need to implement this helper
        
        # AURA delegates the complex problem to CORE.
        co_resolution_plan = self.expert.propose_co_resolution(package, stderr, available_updates)

        if co_resolution_plan and co_resolution_plan.get("plausible"):
            print(f"--> Expert's Advice: A co-resolution is plausible. Proposed plan: {co_resolution_plan['proposed_plan']}")
            # AURA now attempts the "moonshot" probe.
            # (This logic would require extending _try_install_and_validate to handle multiple packages)
            # For now, we will just log it. This is a major new feature.
            print("--> Action: Executing multi-package probe... (feature to be fully implemented)")
            # In a full implementation, you'd run the probe here and handle success/failure.
        else:
            print("--> Expert's Advice: No plausible co-resolution found.")

        # If we reach here, all attempts have failed.
        print(f"--> Healing Result: All healing strategies failed. Reverting to {current_version}.")
        return True, current_version, None
        
    def _heal_with_filter_and_scan(self, package, last_good_version, failed_version, baseline_reqs_path):
        start_group(f"Healing '{package}': Filter-Then-Scan Strategy")

        # --- Phase 1: The High-Speed Compatibility Filter ---
        print("\n--- Phase 1: Filtering for compatible versions ---")
        candidate_versions = self.get_all_versions_between(package, last_good_version, failed_version)
        if not candidate_versions:
            print("No intermediate versions to test."); end_group()
            return last_good_version

        fixed_constraints = []
        with open(baseline_reqs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and self._get_package_name_from_spec(line) != package:
                    fixed_constraints.append(line)

        installable_versions = []
        venv_dir = Path("./temp_pip_check")
        if venv_dir.exists(): shutil.rmtree(venv_dir)
        venv.create(venv_dir, with_pip=True)
        python_executable = str((venv_dir / "bin" / "python").resolve())
        
        for version in reversed(candidate_versions):
            print(f"  -> Checking compatibility of {package}=={version}...")
            requirements_list_for_check = fixed_constraints + [f"{package}=={version}"]
            pip_command = [python_executable, "-m", "pip", "install", "--dry-run"] + requirements_list_for_check
            
            _, stderr, returncode = run_command(pip_command, display_command=False)

            if returncode == 0:
                print(f"     -- Compatible.")
                installable_versions.append(version)
            else:
                # --- START OF NEW, CONTEXTUAL LOGGING ---
                summary = self._get_error_summary(stderr)
                # This new log format is much clearer
                print(f"     -- Incompatible. The attempt to add this version revealed an underlying conflict.")
                print(f"        Diagnosis: {summary}")
                # --- END OF NEW, CONTEXTUAL LOGGING ---
        
        if venv_dir.exists(): shutil.rmtree(venv_dir)

        # --- Phase 2: The Linear Validation Scan ---
        print("\n--- Phase 2: Validating compatible versions (newest first) ---")
        if not installable_versions:
            print("Result: No compatible versions were found. Reverting to last known good version.")
            end_group()
            return last_good_version
        
        print(f"Found {len(installable_versions)} compatible versions to test: {installable_versions}")

        for version_to_test in installable_versions:
            success, _, _ = self._try_install_and_validate(
                package, version_to_test, [], baseline_reqs_path, is_probe=True
            )
            if success:
                print(f"\nSUCCESS: Found latest working version: {package}=={version_to_test}")
                end_group()
                return version_to_test
        
        print("\nResult: No compatible version passed validation. Reverting to last known good version.")
        end_group()
        return last_good_version
    
    def get_all_versions_between(self, package_name, start_ver_str, end_ver_str):
        try:
            package_info = self.pypi.get_project_page(package_name)
            if not (package_info and package_info.packages): return []
            start_v, end_v = parse_version(start_ver_str), parse_version(end_ver_str)
            all_versions = {p.version for p in package_info.packages if p.version and not parse_version(p.version).is_prerelease}
            candidate_versions = [v_str for v_str in all_versions if start_v < parse_version(v_str) <= end_v]
            return sorted(candidate_versions, key=parse_version)
        except Exception: return []

    def _prune_pip_freeze(self, freeze_output):
        lines = freeze_output.strip().split('\n')
        return "\n".join([line for line in lines if '==' in line and not line.startswith('-e')])

    def _get_error_summary(self, error_message: str) -> str:
        """Delegates the task of summarizing an error to the Expert Agent."""
        return self.expert.summarize_error(error_message)
    