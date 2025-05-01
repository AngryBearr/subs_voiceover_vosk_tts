from pathlib import Path
def ensure_folder_exists(folder_path):
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")