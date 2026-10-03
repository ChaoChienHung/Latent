"""
Maintain and sanitize pipeline notebooks in notebooks/.
Strips cell outputs and ensures dual-environment (Colab / Local) setup cell and badge.
"""
import json
import os
from pathlib import Path

REPO_USER = "ChaoChienHung"
REPO_NAME = "Latent"
NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"

SETUP_CODE = f"""# ══════════════════════════════════════════
# Environment Setup (Auto-Detect Colab / Local)
# ══════════════════════════════════════════
import os
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print('Running on Google Colab. Setting up environment...')
    from google.colab import drive
    drive.mount('/content/drive')
    
    PROJECT_ROOT = '/content/drive/MyDrive/Latent'
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    os.makedirs(f'{{PROJECT_ROOT}}/intermediate_data', exist_ok=True)
    os.makedirs(f'{{PROJECT_ROOT}}/utilities', exist_ok=True)
    
    if not os.path.exists('/content/Latent'):
        !git clone https://github.com/{REPO_USER}/{REPO_NAME}.git /content/Latent
        %cd /content/Latent
    else:
        %cd /content/Latent
        !git pull
        
    !pip install -q -r requirements.txt
    !python -m spacy download en_core_web_sm -q
    print(f'Working directory: {{os.getcwd()}}')
    print(f'Data directory: {{PROJECT_ROOT}}')
else:
    PROJECT_ROOT = '.'
    print('Running locally. Ensure dependencies are installed via: pip install -r requirements.txt')
"""

def strip_and_format_notebook(nb_path: Path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Strip outputs and execution counts
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    badge_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO_USER}/{REPO_NAME}/blob/main/notebooks/{nb_path.name})\\n",
            "\\n",
            "---\\n"
        ]
    }

    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in SETUP_CODE.split("\\n")]
    }

    # Filter out existing badge and setup cells
    clean_cells = []
    for cell in nb.get("cells", []):
        src_text = "".join(cell.get("source", []))
        if "Open In Colab" in src_text or "Google Colab Setup" in src_text or "Environment Setup (Auto-Detect" in src_text:
            continue
        clean_cells.append(cell)

    nb["cells"] = [badge_cell, setup_cell] + clean_cells

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Sanitized: {nb_path.name}")

def main():
    notebook_files = [f for f in NOTEBOOKS_DIR.glob("*.ipynb")]
    if not notebook_files:
        print(f"No notebooks found in {NOTEBOOKS_DIR}")
        return
    for nb in sorted(notebook_files):
        strip_and_format_notebook(nb)
    print(f"Successfully processed {len(notebook_files)} notebooks.")

if __name__ == "__main__":
    main()
