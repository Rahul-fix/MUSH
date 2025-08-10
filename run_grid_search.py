import subprocess
import sys
import yaml
import itertools
from pathlib import Path
import wandb

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
    run_name = f"{base_run_name}_" + "_".join([f"{k}{v}" for k, v in config_params.items()])
    return run_name

def experiment_already_ran(run_name, project_name):
    """Check if a run with the given run_name already exists in the wandb project."""
    try:
        api = wandb.Api()
        runs = api.runs(project_name)
        for run in runs:
            if run.name == run_name:
                return True
        return False
    except Exception as e:
        print(f"Warning: Could not check wandb for existing runs due to: {e}")
        return False
    
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
    
    print(f"Executing: {' '.join(cmd)}")
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
        run_name = f"run_{i}_" + "_".join([f"{k}{v}" for k, v in combo.items()])
        if experiment_already_ran(run_name, project_name):
            print(f"Run {run_name} already exists in wandb project {project_name}. Skipping.")
            continue
        attempt = 1
        while True:
            print(f"Attempt {attempt} for experiment {i}")
            # Actually run the experiment
            # Use the original run_single_experiment logic, but skip run_name generation
            cmd = [
                "accelerate", "launch",
                "--config_file", "accelerate_config.yaml",
                "train.py"
            ]
            for param, value in combo.items():
                cmd.extend([f"--{param}", str(value)])
            cmd.extend(["--project_name", project_name])
            cmd.extend(["--run_name", run_name])
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode == 0:
                successful_runs += 1
                print(f"Experiment completed successfully: {run_name}")
                break
            else:
                print(f"Experiment {i} failed on attempt {attempt}. Retrying...")
                attempt += 1
    print(f"\n--- Grid Search Complete ---")
    print(f"Successful runs: {successful_runs}/{len(combinations)}")

if __name__ == "__main__":
    main()
