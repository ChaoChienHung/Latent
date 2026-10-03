# Pipeline Notebooks

This directory contains the NLP pipeline notebooks for Latent. Each notebook is **dual-environment compatible**, featuring auto-detection for both **local Jupyter** and **Google Colab** runtimes.

## 🚀 Running the Notebooks

### 💻 Local Execution (Own GPU / CPU)

For running locally with your own Python environment:

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start Jupyter
jupyter notebook notebooks/
```

When run locally, the setup cell automatically recognizes your environment, uses your local project directory as `PROJECT_ROOT`, and loads models and data accordingly.

### ☁️ Google Colab Execution (Free Cloud GPU)

Every notebook includes an **"Open In Colab"** badge at the top:

1. Click the **"Open In Colab"** badge at the top of any notebook.
2. Ensure GPU acceleration is enabled: `Runtime → Change runtime type → T4 GPU`.
3. Run the first cell — it will automatically:
   - Mount your Google Drive for data persistence.
   - Clone or pull the Latent repository to `/content/Latent`.
   - Install required packages and the spaCy language model.
   - Configure working directories seamlessly.

---

## 📋 Pipeline Stages

| Stage | Notebook | Description |
|---|---|---|
| **0** | [`Stage_0_Introduction.ipynb`](file:///notebooks/Stage_0_Introduction.ipynb) | Project overview, architecture, and dataset summary |
| **1** | [`Stage_1_Data_Collection.ipynb`](file:///notebooks/Stage_1_Data_Collection.ipynb) | Scraping ingestion, schema canonicalization, and deduplication |
| **2** | [`Stage_2_POS_NER_Tagging.ipynb`](file:///notebooks/Stage_2_POS_NER_Tagging.ipynb) | Part-of-Speech tagging and Named Entity Recognition via spaCy |
| **3** | [`Stage_3_Singlish_Normalisation.ipynb`](file:///notebooks/Stage_3_Singlish_Normalisation.ipynb) | Singlish particle & colloquial word normalization |
| **4** | [`Stage_4_Singlish_to_English.ipynb`](file:///notebooks/Stage_4_Singlish_to_English.ipynb) | Lexicon-guided Singlish to standard English conversion |
| **5** | [`Stage_5_Text_Normalisation.ipynb`](file:///notebooks/Stage_5_Text_Normalisation.ipynb) | Slang expansion, emoji translation, lemmatization, stop words |
| **6** | [`Stage_6_Vector_Space_Model.ipynb`](file:///notebooks/Stage_6_Vector_Space_Model.ipynb) | TF-IDF, BM25Okapi, Sentence-BERT embeddings, Inverted Index |
| **7** | [`Stage_7_Sentiment_Analysis.ipynb`](file:///notebooks/Stage_7_Sentiment_Analysis.ipynb) | RoBERTa 7-class emotion classification & sentiment metrics |
| **8** | [`Stage_8_Clustering.ipynb`](file:///notebooks/Stage_8_Clustering.ipynb) | TruncatedSVD dimensionality reduction, K-Means clustering, t-SNE |
| **9** | [`Stage_9_Document_Search.ipynb`](file:///notebooks/Stage_9_Document_Search.ipynb) | Multi-model document retrieval & recommendation engine |
| **A1** | [`Appendix_1_Sentiment_Benchmark.ipynb`](file:///notebooks/Appendix_1_Sentiment_Benchmark.ipynb) | Zero-shot & fine-tuned sentiment benchmark comparisons |

> **Note**: For production pipelines and batch execution, equivalent modular Python scripts are available in [`preprocessing/`](../preprocessing/), [`modelling/`](../modelling/), and [`search/`](../search/).
