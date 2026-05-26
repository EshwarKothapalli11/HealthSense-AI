import os
import shutil
import pathlib

def get_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

def main():
    base_dir = r"d:\Mental"
    
    # 1. Folders to completely remove (caches, redundant envs)
    to_remove = [
        os.path.join(base_dir, ".venv"),
        os.path.join(base_dir, ".pytest_cache"),
        os.path.join(base_dir, ".gradio"),
        os.path.join(base_dir, "artifacts", "plots"),
        os.path.join(base_dir, "data", "raw"), # datasets not needed for inference
    ]
    
    # Also find all __pycache__ folders
    for dirpath, dirnames, _ in os.walk(base_dir):
        if "__pycache__" in dirnames:
            to_remove.append(os.path.join(dirpath, "__pycache__"))
            
    # 2. Folders to move to /training
    training_dir = os.path.join(base_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    
    to_move = [
        os.path.join(base_dir, "train_all.py"),
        os.path.join(base_dir, "retrain_mental.py"),
        os.path.join(base_dir, "tests"),
    ]
    
    # move src/data, src/models, src/evaluation
    for folder in ["data", "models", "evaluation"]:
        src_path = os.path.join(base_dir, "src", folder)
        if os.path.exists(src_path):
            to_move.append(src_path)

    print("="*50)
    print("HealthSense AI - Optimization & Cleanup Script")
    print("="*50)

    # Calculate before size (excluding python_portable to avoid long scan)
    total_before = 0
    for item in os.listdir(base_dir):
        if item != "python_portable":
            total_before += get_size(os.path.join(base_dir, item))
    print(f"Project Size Before (excluding portable env): {format_size(total_before)}")

    # Remove
    print("\n[Removing Caches, Broken Envs & Training Data]")
    for path in to_remove:
        if os.path.exists(path):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
                print(f"  - Deleted: {path}")
            except Exception as e:
                print(f"  ! Failed to delete {path}: {e}")

    # Move
    print("\n[Moving Training Files to /training]")
    for path in to_move:
        if os.path.exists(path):
            target = os.path.join(training_dir, os.path.basename(path))
            try:
                shutil.move(path, target)
                print(f"  - Moved: {os.path.basename(path)} -> /training")
            except Exception as e:
                print(f"  ! Failed to move {path}: {e}")

    # Calculate after size
    total_after = 0
    for item in os.listdir(base_dir):
        if item != "python_portable":
            total_after += get_size(os.path.join(base_dir, item))
    
    print("\n" + "="*50)
    print(f"Project Size After (excluding portable env): {format_size(total_after)}")
    print(f"Saved Space: {format_size(total_before - total_after)}")
    print("="*50)
    print("Note: 'python_portable' was left untouched as it is currently running your server.")

if __name__ == "__main__":
    main()
