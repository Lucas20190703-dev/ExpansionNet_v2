
from pathlib import Path

def scan_files_recursively(directory, extensions):
    path = Path(directory)
    files = []
    for ext in extensions:
        for img in path.rglob(f'*.{ext}'):
            files.append(img.as_uri())
    
    return files
