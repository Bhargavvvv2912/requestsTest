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
        self.usage_scores = self._calculate_update_risk_components()
        self.exclusions_from_this_run = set()


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
        
        is_pinned, _ = self._get_requirements_state()
        if not is_pinned:
            self._bootstrap_unpinned_requirements()
            is_pinned, _ = self._get_requirements_state()
            if not is_pinned:
                sys.exit("CRITICAL: Bootstrap process failed to produce a fully pinned requirements file.")

        final_successful_updates, final_failed_updates = {}, {}
        pass_num = 0
        
        # --- Placeholder for dependency graph parsing ---
        # For HURM 4.0, you would run `pipdeptree --json-tree` here on the initial
        # environment and parse the output into self.dependency_graph_metrics.
        if not hasattr(self, 'dependency_graph_metrics'):
            print("INFO: Dependency graph metrics not found. Entanglement scores will be 0.")
            self.dependency_graph_metrics = {}
        # --- End of placeholder ---

        while pass_num < self.config["MAX_RUN_PASSES"]:
            pass_num += 1
            start_group(f"UPDATE PASS {pass_num}/{self.config['MAX_RUN_PASSES']}")
            
            progressive_baseline_path = Path(f"./pass_{pass_num}_progressive_reqs.txt")
            shutil.copy(self.requirements_path, progressive_baseline_path)
            changed_packages_this_pass = set()

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

            if not packages_to_update:
                if pass_num == 1:
                    print("\nInitial baseline is already fully up-to-date. Running a final health check.")
                    self._run_final_health_check()
                else:
                    print("\nNo further updates are available. The system has successfully converged.")
                if progressive_baseline_path.exists(): progressive_baseline_path.unlink()
                break 

            # --- START of HURM 4.0 Convergence-Focused Risk Calculation ---
            update_plan = []
            for pkg, cur, target in packages_to_update:
                components = self._calculate_update_risk_components(pkg, cur, target)
                update_plan.append({'pkg': pkg, 'cur': cur, 'target': target, 'components': components})

            max_ecosystem = max(p['components']['ecosystem'] for p in update_plan) if update_plan else 0
            max_depth = max(p['components']['depth'] for p in update_plan) if update_plan else 0

            W_ECOSYSTEM, W_DEPTH = 1.0, 0.5
            for p in update_plan:
                comps = p['components']
                norm_ecosystem = (comps['ecosystem'] / max_ecosystem) if max_ecosystem > 0 else 0
                norm_depth = (comps['depth'] / max_depth) if max_depth > 0 else 0
                entanglement_score = (W_ECOSYSTEM * norm_ecosystem) + (W_DEPTH * norm_depth)
                p['final_score'] = (comps['severity'] * 10) + entanglement_score
                p['code_impact_score'] = comps['usage'] + (comps['criticality'] * 10)

            update_plan.sort(key=lambda p: (p['final_score'], p['code_impact_score']))
            
            min_display_score = update_plan[0]['final_score'] if update_plan else 0
            max_display_score = update_plan[-1]['final_score'] if update_plan else 0
            for p in update_plan:
                 if max_display_score == min_display_score: p['norm_score_display'] = 0.0
                 else: p['norm_score_display'] = ((p['final_score'] - min_display_score) / (max_display_score - min_display_score)) * 100.0
            # --- END of HURM 4.0 Logic ---

            print("\nPrioritized Update Plan for this Pass (Lowest Risk First):")
            for i, p in enumerate(update_plan):
                print(f"  {i+1}/{len(update_plan)}: {p['pkg']} (Risk: {p['norm_score_display']:.2f}) -> {p['target']}")
            
            for i, p_data in enumerate(update_plan):
                package, current_ver, target_ver = p_data['pkg'], p_data['cur'], p_data['target']
                print(f"\n" + "-"*80); print(f"PULSE: [PASS {pass_num} | ATTEMPT {i+1}/{len(update_plan)}] Processing '{package}'"); print(f"PULSE: Changed packages this pass so far: {changed_packages_this_pass}"); print("-"*80)
                
                success, reason_or_new_version, _ = self.attempt_update_with_healing(
                    package, current_ver, target_ver, [], progressive_baseline_path
                )
                
                if success:
                    reached_version, is_a_real_change = "", False
                    if "skipped" in str(reason_or_new_version):
                        reached_version, is_a_real_change = current_ver, False
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
            self._run_final_health_check()

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
        python_executable = str((venv_dir / "bin" / "python").resolve())
        _, stderr, returncode = run_command([python_executable, "-m", "pip", "install", "-r", str(self.requirements_path)])
        if returncode != 0:
            print("CRITICAL ERROR: Final installation of combined dependencies failed!", file=sys.stderr); return
        success, metrics, _ = validate_changes(python_executable, self.config, group_title="Final System Health Check")
        if success and metrics and "not available" not in metrics:
            print("\n" + "="*70); print("=== FINAL METRICS FOR THE FULLY UPDATED ENVIRONMENT ==="); print("\n".join([f"  {line}" for line in metrics.split('\n')])); print("="*70)
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

    def _try_install_and_validate(self, package_to_update, new_version, dynamic_constraints, baseline_reqs_path, is_probe):
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
            if not is_probe:
                print(f"--> Change analysis: '{package_to_update}' version remains at {old_version}. Validation skipped.")
            return True, "Validation skipped (no change)", ""

        if not is_probe: print(f"\nChange analysis: Updating '{package_to_update}' from {old_version} -> {new_version}")

        temp_reqs_path = venv_dir / "temp_requirements.txt"
        with open(baseline_reqs_path, "r") as f_read, open(temp_reqs_path, "w") as f_write:
            lines_for_file = []
            for line in f_read:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if self._get_package_name_from_spec(line.split(';')[0]) == package_to_update:
                    marker_part = f" ;{line.split(';')[1]}" if ';' in line else ""
                    lines_for_file.append(f"{package_to_update}=={new_version}{marker_part}")
                else: lines_for_file.append(line)
            f_write.write("\n".join(lines_for_file))

        _, stderr_install, returncode = run_command([python_executable, "-m", "pip", "install", "-r", str(temp_reqs_path)])
        
        if returncode != 0:
            # THIS IS THE RESTORED DETAILED DIAGNOSTIC LOGIC
            print("INFO: Main installation failed. Analyzing conflict...")
            conflict_match = re.search(r"Cannot install(?P<packages>[\s\S]+?)because", stderr_install)
            reason = ""
            if conflict_match:
                conflicting_packages = ' '.join(conflict_match.group('packages').split()).replace(' and ', ', ').replace(',', ', ')
                reason = f"Installation conflict: {conflicting_packages}"
                print(f"DIAGNOSIS: {reason}")
            else:
                reason = "Installation conflict (Could not parse specific packages)"
            return False, reason, stderr_install
        
        group_title = f"Validation for {package_to_update}=={new_version}"
        success, metrics, validation_output = validate_changes(python_executable, self.config, group_title=group_title)

        if not success:
            return False, "Validation script failed", validation_output
        return True, metrics, ""

    def attempt_update_with_healing(self, package, current_version, target_version, dynamic_constraints, baseline_reqs_path):
        print(f"\nAttempting to validate {package}=={target_version}...")
        
        success, result_data, stderr = self._try_install_and_validate(
            package, target_version, dynamic_constraints, baseline_reqs_path, is_probe=False
        )
        
        if success:
            print(f"Direct update to {package}=={target_version} succeeded.")
            return True, result_data if "skipped" in str(result_data) else target_version, None

        print(f"\nINFO: Initial update for '{package}' failed. Reason: '{result_data}'")
        start_group("View Full Error Log for Initial Failure"); print(stderr); end_group()
        print("INFO: Entering healing mode with 'Filter-Then-Scan' strategy.")

        healed_version = self._heal_with_filter_and_scan(package, current_version, target_version, baseline_reqs_path)
        
        if healed_version:
            return True, healed_version, None
        
        return False, f"All healing attempts for {package} failed.", None

    def _heal_with_filter_and_scan(self, package, last_good_version, failed_version, baseline_reqs_path):
        start_group(f"Healing '{package}' with Filter-Then-Scan")

        print("\n--- Phase 1: Filtering for installable versions ---")
        candidate_versions = self.get_all_versions_between(package, last_good_version, failed_version)
        if not candidate_versions:
            print("No intermediate versions found to test."); end_group()
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
            print(f"  -> Checking installability of {package}=={version}...")
            temp_reqs_for_check = fixed_constraints + [f"{package}=={version}"]
            temp_reqs_path = venv_dir / "check_reqs.txt"
            with open(temp_reqs_path, "w") as f: f.write("\n".join(temp_reqs_for_check))
            pip_command = [python_executable, "-m", "pip", "install", "--dry-run", "--quiet", "-r", str(temp_reqs_path)]
            _, _, returncode = run_command(pip_command)
            if returncode == 0:
                print(f"     -- OK: {package}=={version} is installable.")
                installable_versions.append(version)
        
        if venv_dir.exists(): shutil.rmtree(venv_dir)

        print("\n--- Phase 2: Validating installable versions (newest first) ---")
        if not installable_versions:
            print("No installable versions found in range."); end_group()
            return last_good_version

        for version_to_test in installable_versions:
            print(f"  -> Validating {package}=={version_to_test}...")
            success, _, _ = self._try_install_and_validate(
                package, version_to_test, [], baseline_reqs_path, is_probe=True
            )
            if success:
                print(f"\nSUCCESS: Found latest working version: {package}=={version_to_test}")
                end_group()
                return version_to_test
        
        print("\nINFO: No installable version passed validation."); end_group()
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