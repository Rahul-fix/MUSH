import subprocess
import sys
import yaml
import itertools
import json
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_combinations(parameters):
    """Generate all combinations of parameters"""
    param_names = []
    param_values = []
    
    for param_name, param_config in parameters.items():
        param_names.append(param_name)
        if 'values' in param_config:
            param_values.append(param_config['values'])
        elif 'value' in param_config:
            param_values.append([param_config['value']])
    
    combinations = list(itertools.product(*param_values))
    return [(dict(zip(param_names, combo))) for combo in combinations]

def run_single_experiment(config_params, project_name, run_name, log_path):
    """Run a single experiment with given parameters"""
    # Print run details before execution
    print(f"\n[RUN] Starting experiment: {run_name}")
    print(f"[RUN] Parameters: {config_params}")
    cmd = [
        "accelerate", "launch",
        "--config_file", "accelerate_config.yaml",
        "train.py"
    ]
    for param, value in config_params.items():
        cmd.extend([f"--{param}", str(value)])
    cmd.extend(["--project_name", project_name])
    cmd.extend(["--run_name", run_name])
    print(f"[RUN] Command: {' '.join(cmd)}")
    max_retries = 2
    for attempt in range(1, max_retries + 2):
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"Experiment completed successfully: {run_name}")
            update_experiment_log(log_path, run_name, "finished")
            return True
        else:
            print(f"Experiment failed with return code: {result.returncode} (Attempt {attempt})")
            if attempt < max_retries + 1:
                print(f"Retrying experiment: {run_name}")
            else:
                print(f"Experiment failed after {max_retries + 1} attempts: {run_name}")
                update_experiment_log(log_path, run_name, "failed")
                return False

def update_experiment_log(log_path, run_name, status):
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
    except Exception:
        log = {}
    log[run_name] = status
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

def main():
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        print("No config file provided. Defaulting to 'grid_config.yaml'.")
        config_file = "grid_config.yaml"
    config = load_config(config_file)
    parameters = config.get('parameters', {})
    project_name = config.get('project')
    if not project_name:
        print("ERROR: 'project' field missing in config file. Please add 'project: <project_name>' to your config YAML.")
        sys.exit(1)
    combinations = generate_combinations(parameters)
    log_path = "experiment_log.json"
    try:
        with open(log_path, 'r') as f:
            experiment_log = json.load(f)
    except Exception:
        experiment_log = {}
    print(f"Found {len(combinations)} parameter combinations to run")
    successful_runs = 0
    for i, combo in enumerate(combinations, 1):
        run_name = "_".join([f"{k}{v}" for k, v in combo.items()])
        status = experiment_log.get(run_name)
        if status == "finished":
            print(f"[SKIP] Experiment already finished: {run_name}")
            continue
        print(f"\n--- Running experiment {i}/{len(combinations)} ---")
        print(f"Parameters: {combo}")
        success = run_single_experiment(combo, project_name, run_name, log_path)
        if success:
            successful_runs += 1
        else:
            print(f"Stopping grid search due to failure in run {i}")
            break
    print(f"\n--- Grid Search Complete ---")
    print(f"Successful runs: {successful_runs}/{len(combinations)}")

if __name__ == "__main__":
    main()
