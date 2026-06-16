import os
import shutil

# Source and destination directories
src_root = r"d:\Mental"
dest_root = r"d:\Mental\Health-Sence-AI"

print("="*60)
print("HealthSense AI - Preparing Hugging Face Space Repository")
print("="*60)

# Create destination folders
os.makedirs(os.path.join(dest_root, "src"), exist_ok=True)
os.makedirs(os.path.join(dest_root, "models"), exist_ok=True)
os.makedirs(os.path.join(dest_root, "assets"), exist_ok=True)


# Copy and adjust config.py paths for the new flat layout
config_dest = os.path.join(dest_root, "config.py")
shutil.copy(os.path.join(src_root, "config.py"), config_dest)

with open(config_dest, "r") as f:
    config_content = f.read()

config_content = config_content.replace("'artifacts', 'models'", "'models'")
config_content = config_content.replace("'artifacts', 'scalers'", "'models'")
config_content = config_content.replace("'artifacts', 'plots'", "'assets'")

with open(config_dest, "w") as f:
    f.write(config_content)
    
print("Copied and optimized config.py paths")


# Copy src folders (data, evaluation, fusion)
for folder in ["data", "evaluation", "fusion"]:
    src_folder = os.path.join(src_root, "src", folder)
    dest_folder = os.path.join(dest_root, "src", folder)
    if os.path.exists(src_folder):
        try:
            shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
            print(f"Copied src/{folder}")
        except Exception as e:
            print(f"Warning: Failed to copy src/{folder}: {e}")

# Copy models and scalers to the models/ folder in destination
models_src = os.path.join(src_root, "artifacts", "models")
scalers_src = os.path.join(src_root, "artifacts", "scalers")
dest_models = os.path.join(dest_root, "models")

for f in os.listdir(models_src):
    if f.endswith(".h5") or f.endswith(".pkl"):
        try:
            shutil.copy(os.path.join(models_src, f), os.path.join(dest_models, f))
            print(f"Copied model/scaler: {f}")
        except Exception as e:
            print(f"Warning: Failed to copy model/scaler {f}: {e}")

for f in os.listdir(scalers_src):
    if f.endswith(".pkl"):
        try:
            shutil.copy(os.path.join(scalers_src, f), os.path.join(dest_models, f))
            print(f"Copied scaler/tokenizer: {f}")
        except Exception as e:
            print(f"Warning: Failed to copy scaler/tokenizer {f}: {e}")

# Remove unnecessary files from destination
unnecessary_files = [
    os.path.join(dest_root, "Dockerfile"),
    os.path.join(dest_root, "docker-compose.yml"),
    os.path.join(dest_root, "src", "streamlit_app.py") # we will use app.py in root
]
for f in unnecessary_files:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Removed unnecessary file: {f}")
        except Exception as e:
            print(f"Warning: Failed to remove file {f}: {e}")

# Remove unnecessary directories from destination
unnecessary_folders = [
    os.path.join(dest_root, "python_portable"),
    os.path.join(dest_root, "venv"),
    os.path.join(dest_root, ".venv"),
    os.path.join(dest_root, "node_modules"),
    os.path.join(dest_root, ".pytest_cache"),
]
for d in unnecessary_folders:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f"Removed unnecessary directory: {d}")
        except Exception as e:
            print(f"Warning: Failed to remove directory {d}: {e}")

# Walk through destination to find any __pycache__ folders to delete
for dirpath, dirnames, _ in os.walk(dest_root):
    if "__pycache__" in dirnames:
        pycache_path = os.path.join(dirpath, "__pycache__")
        try:
            shutil.rmtree(pycache_path)
            print(f"Cleaned __pycache__ in {dirpath}")
        except Exception as e:
            print(f"Warning: Failed to clean __pycache__ in {dirpath}: {e}")

print("\nSuccess! Repository folder is clean and prepared.")

print("="*60)
