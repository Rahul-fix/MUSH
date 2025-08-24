import os
import re
import shutil

# Path to the Output directory containing model files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Output')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Create the models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Regex to match run_name in filenames (e.g., 551ouxf3 in any *_551ouxf3.pt)
run_name_pattern = re.compile(r'_([a-z0-9]{8})\.pt$')

# Collect all .pt files in Output directory
model_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')]

# Map run_name to list of files
run_name_to_files = {}
for filename in model_files:
    match = run_name_pattern.search(filename)
    if match:
        run_name = match.group(1)
        run_name_to_files.setdefault(run_name, []).append(filename)

# For each run_name, create a folder and move files
for run_name, files in run_name_to_files.items():
    run_folder = os.path.join(MODELS_DIR, run_name)
    os.makedirs(run_folder, exist_ok=True)
    for file in files:
        src = os.path.join(OUTPUT_DIR, file)
        dst = os.path.join(run_folder, file)
        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

print("Model files have been organized by run_name in the 'models/' directory.")
