"""
Convert existing Stage notebooks into clean local + Colab versions.
Strips output cells and adds Colab-specific setup cells.
"""
import json
import os
import copy

NOTEBOOKS = [
    ("Stage_0_Introduction.ipynb", "Stage_0_Introduction.ipynb"),
    ("Stage_1_Data_Collection_and_Data_Cleaning.ipynb", "Stage_1_Data_Collection.ipynb"),
    ("Stage_2_POS_and_NER_Tagging.ipynb", "Stage_2_POS_NER_Tagging.ipynb"),
    ("Stage_3_Singlish_Normalisation.ipynb", "Stage_3_Singlish_Normalisation.ipynb"),
    ("Stage_4_Singlish_to_English_Conversion.ipynb", "Stage_4_Singlish_to_English.ipynb"),
    ("Stage_5_Common_Normalisation.ipynb", "Stage_5_Text_Normalisation.ipynb"),
    ("Stage_6_Vector_Space_Model_and_Inverted_Index.ipynb", "Stage_6_Vector_Space_Model.ipynb"),
    ("Stage_8_Clustering_and_Visualization.ipynb", "Stage_8_Clustering.ipynb"),
    ("Stage_9_Document_Search.ipynb", "Stage_9_Document_Search.ipynb"),
    ("Step_Appendix_1_Sentiment Labelling (Benchmark).ipynb", "Appendix_1_Sentiment_Benchmark.ipynb"),
]

# Colab setup cell to prepend
COLAB_SETUP_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ══════════════════════════════════════════\n",
        "# Google Colab Setup\n",
        "# ══════════════════════════════════════════\n",
        "# Run this cell first to set up the environment in Google Colab.\n",
        "\n",
        "import os\n",
        "import sys\n",
        "\n",
        "# Check if running in Colab\n",
        "IN_COLAB = 'google.colab' in sys.modules\n",
        "\n",
        "if IN_COLAB:\n",
        "    # Mount Google Drive for data persistence\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    \n",
        "    # Set project root on Google Drive\n",
        "    PROJECT_ROOT = '/content/drive/MyDrive/Latent'\n",
        "    os.makedirs(PROJECT_ROOT, exist_ok=True)\n",
        "    os.makedirs(f'{PROJECT_ROOT}/intermediate_data', exist_ok=True)\n",
        "    os.makedirs(f'{PROJECT_ROOT}/utilities', exist_ok=True)\n",
        "    \n",
        "    # Clone or update the repo\n",
        "    if not os.path.exists('/content/Latent'):\n",
        "        !git clone https://github.com/YOUR_USERNAME/Latent.git /content/Latent\n",
        "    \n",
        "    # Install dependencies\n",
        "    !pip install -q spacy nltk ftfy emoji langdetect sentence-transformers rank-bm25 tqdm\n",
        "    !python -m spacy download en_core_web_sm\n",
        "    \n",
        "    # Set working directory\n",
        "    os.chdir('/content/Latent')\n",
        "    print(f'Working directory: {os.getcwd()}')\n",
        "    print(f'Data directory: {PROJECT_ROOT}')\n",
        "else:\n",
        "    PROJECT_ROOT = '.'\n",
        "    print('Running locally. Ensure dependencies are installed via: pip install -r requirements.txt')\n"
    ]
}

COLAB_BADGE_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Latent/blob/main/notebooks/colab/{notebook_name})\n",
        "\n",
        "---\n"
    ]
}

def strip_outputs(nb_data):
    """Remove all output cells and execution counts."""
    for cell in nb_data.get("cells", []):
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb_data

def update_project_structure_refs(cells):
    """Replace old CS5246Project references with Latent."""
    for cell in cells:
        new_source = []
        for line in cell.get("source", []):
            line = line.replace("CS5246Project/", "Latent/")
            line = line.replace("CS5246Project", "Latent")
            new_source.append(line)
        cell["source"] = new_source
    return cells

def create_local_version(src_path, dst_path):
    """Create a clean local version: stripped outputs, updated refs."""
    with open(src_path) as f:
        nb = json.load(f)
    
    nb = strip_outputs(nb)
    nb["cells"] = update_project_structure_refs(nb["cells"])
    
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  Local: {dst_path}")

def create_colab_version(src_path, dst_path, notebook_name):
    """Create a Colab version: badge + setup cell + stripped outputs."""
    with open(src_path) as f:
        nb = json.load(f)
    
    nb = strip_outputs(nb)
    nb["cells"] = update_project_structure_refs(nb["cells"])
    
    # Create badge cell
    badge = copy.deepcopy(COLAB_BADGE_CELL)
    badge["source"] = [s.replace("{notebook_name}", notebook_name) for s in badge["source"]]
    
    # Prepend badge + setup
    nb["cells"] = [badge, copy.deepcopy(COLAB_SETUP_CELL)] + nb["cells"]
    
    # Ensure Colab-compatible kernel spec
    nb.setdefault("metadata", {})
    nb["metadata"]["colab"] = {
        "name": notebook_name,
        "provenance": [],
        "gpuType": "T4"
    }
    nb["metadata"]["accelerator"] = "GPU"
    
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  Colab: {dst_path}")

def main():
    src_dir = "."
    local_dir = "notebooks/local"
    colab_dir = "notebooks/colab"
    
    for src_name, dst_name in NOTEBOOKS:
        src_path = os.path.join(src_dir, src_name)
        if not os.path.exists(src_path):
            print(f"SKIP (not found): {src_path}")
            continue
        
        print(f"Converting: {src_name}")
        create_local_version(src_path, os.path.join(local_dir, dst_name))
        create_colab_version(src_path, os.path.join(colab_dir, dst_name), dst_name)
    
    print("\nDone! Notebooks created in:")
    print(f"  Local:  {local_dir}/")
    print(f"  Colab:  {colab_dir}/")

if __name__ == "__main__":
    main()
