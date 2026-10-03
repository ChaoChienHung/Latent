# Pipeline Notebooks

This directory contains the NLP pipeline notebooks in two versions:

## 📁 `local/` — Local Python (Jupyter)

For users with **their own GPU** or sufficient CPU resources. These notebooks run in a standard Jupyter environment.

### Prerequisites
```bash
pip install -r requirements.txt
# or
uv sync
```

### Run
```bash
jupyter notebook notebooks/local/
```

## ☁️ `colab/` — Google Colab (Free GPU)

For users who want to use **Google's free T4 GPU**. Each notebook includes:
- An "Open in Colab" badge at the top
- Auto-setup cells (Google Drive mount, dependency installation, repo clone)
- GPU-optimized runtime settings

### Run
1. Open any notebook from `notebooks/colab/`
2. Click the **"Open in Colab"** badge
3. Set runtime type to **GPU** (`Runtime → Change runtime type → T4 GPU`)
4. Run all cells sequentially

> **Note**: Colab notebooks save data to Google Drive for persistence between sessions.

---

## Pipeline Stages

| Stage | Notebook | Description |
|---|---|---|
| 0 | `Stage_0_Introduction` | Project overview and data summary |
| 1 | `Stage_1_Data_Collection` | Scraping + data cleaning |
| 2 | `Stage_2_POS_NER_Tagging` | Part-of-Speech and Named Entity Recognition |
| 3 | `Stage_3_Singlish_Normalisation` | Singlish expression standardization |
| 4 | `Stage_4_Singlish_to_English` | Singlish → English conversion |
| 5 | `Stage_5_Text_Normalisation` | Slang expansion, spelling, lemmatization |
| 6 | `Stage_6_Vector_Space_Model` | TF-IDF, BM25, Sentence-BERT indices |
| 7 | `Stage_7_Sentiment_Analysis` | 7-class emotion classification |
| 8 | `Stage_8_Clustering` | K-Means topic clustering + t-SNE |
| 9 | `Stage_9_Document_Search` | Search & recommendation engine |
| A1 | `Appendix_1_Sentiment_Benchmark` | Sentiment model evaluation |

Run stages **in order** — each stage's output is the next stage's input.
