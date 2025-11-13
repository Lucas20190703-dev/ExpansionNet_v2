import os
from pathlib import Path


def remove_prefix(uri : str, prefix = "file:///"):
    if uri.startswith(prefix):
        return uri[len(prefix):]
    return uri


def generate_qrc(folder_path : Path, prefix : str) -> Path:
    """
    Scans a folder and returns a list of all files in it.

    Args:
        folder_path (str): The path to the folder to scan.

    Returns:
        list: A list of file names in the folder.
    """
    files = []
    try:
        items = os.listdir(folder_path)
        for item in items:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                files.append(item)
    except FileNotFoundError:
        print(f"Error: Folder not found: {folder_path}")
    except NotADirectoryError:
        print(f"Error: Not a directory: {folder_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    qrc_file = folder_path.parent / "icons.qrc"
    
    with open(qrc_file, 'w') as f:
        f.write("<!DOCTYPE RCC>\n")
        f.write("<RCC version=\"1.0\">\n")
        f.write(f"\t<qresource prefix=\"{prefix}\">\n")
        
        for ele in files:
            file_path = folder_path.as_uri() + "/" + ele
            f.write(f"\t\t<file alias=\"{ele}\">{remove_prefix(file_path)}</file>\n")
            
        f.write("</qresource>\n")
        f.write("</RCC>\n")
    
    return qrc_file

def generate_rcc(qrc_file : Path) -> Path:
    
    rcc_file = qrc_file.resolve().parent / "icons.py"
    
    source = remove_prefix(qrc_file.as_uri())
    target = remove_prefix(rcc_file.as_uri())
        
    os.system(f"pyside2-rcc {source} -o {target}")

def test_remove_prefix():
    file = "file:///F:/source/something.jpg"
    rfile = remove_prefix(file)
    return "file:///" + rfile == file

if __name__ == "__main__":
    
    if not test_remove_prefix():
        print("Test: 'remove_prefix' failed.")
        exit(-1)
    
    folder = Path(__file__).resolve().parent / "icons"
    
    qrc_file = generate_qrc(folder, "icons")
    rcc_file = generate_rcc(qrc_file)