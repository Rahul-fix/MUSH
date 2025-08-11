import subprocess
import sys
import yaml
import itertools
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

def run_single_experiment(config_params, project_name, base_run_name="experiment"):
    """Run a single experiment with given parameters"""
    # Generate run name based only on parameter values for uniqueness
    run_name = "_".join([f"{k}{v}" for k, v in config_params.items()])

    # Print run details before execution
    print(f"\n[RUN] Starting experiment: {run_name}")
    print(f"[RUN] Parameters: {config_params}")
    
    cmd = [
        "accelerate", "launch",
        "--config_file", "accelerate_config.yaml",
        "train.py"
    ]
    
    # Add parameters as command line arguments
    for param, value in config_params.items():
        cmd.extend([f"--{param}", str(value)])
    
    cmd.extend(["--project_name", project_name])
    cmd.extend(["--run_name", run_name])
    
    print(f"[RUN] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Experiment failed with return code: {result.returncode}")
        return False
    
    print(f"Experiment completed successfully: {run_name}")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_grid_search.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]  # Fixed: Added index 1 to get the argument
    config = load_config(config_file)
    parameters = config.get('parameters', {})
    project_name = config.get('project', 'pepper-segmentation-grid')
    combinations = generate_combinations(parameters)
    
    print(f"Found {len(combinations)} parameter combinations to run")
    
    successful_runs = 0
    for i, combo in enumerate(combinations, 1):
        print(f"\n--- Running experiment {i}/{len(combinations)} ---")
        print(f"Parameters: {combo}")
        
        success = run_single_experiment(combo, project_name)
        if success:
            successful_runs += 1
        else:
            print(f"Stopping grid search due to failure in run {i}")
            break
    
    print(f"\n--- Grid Search Complete ---")
    print(f"Successful runs: {successful_runs}/{len(combinations)}")

if __name__ == "__main__":
    main()
