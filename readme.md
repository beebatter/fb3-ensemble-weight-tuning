# **FB3 DeBERTa Family Ensemble & Weight Tuning**

## **1. Project Overview**
This repository automates the end-to-end ensemble inference and weight tuning workflow used in my Kaggle “Feedback Prize – English Language Learning” competition submission with DeBERTa family models. The process includes:
- Running each pretrained model to generate predictions.
- Searching for optimal linear ensemble weights on a held-out validation set.
- Applying those weights to produce a final submission.
Final result: [**Silver medal (Rank 126 / 2,654 teams)**](https://www.kaggle.com/code/easkwon/fb3-deberta-family-inference-weight-tune).

## **2. Competition & Dataset**
**Competition:** [Feedback Prize – English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)  
**Objective:** Predict six language proficiency scores for 8th–12th grade English Language Learners (ELLs) from their essay text:
- **cohesion**, **syntax**, **vocabulary**, **phraseology**, **grammar**, **conventions**

| Split       | Essays | Labels Provided? |
|-------------|--------|------------------|
| Training    | ~10,000| Yes              |
| Validation* | –      | Yes              |
| Test        | ~8,000 | No               |

\* Validation set labels must be prepared (e.g. from `data/df_folds.csv`).

## **3. Methods & Components**

### **3.1 Notebook: Baseline Inference**
- **Path:** `Notebook/fb3-deberta-v3-large-baseline-inference.ipynb`  
- Loads a pretrained DeBERTa-v3-large model via Hugging Face Transformers.  
- Processes test essays in batches and writes out `data/submission_<model>.csv`.

### **3.2 Notebook: Weight Tuning**
- **Path:** `Notebook/fb3-deberta-family-inference-weight-tune.ipynb`  
- Reads each model’s validation-set predictions (`data/submission_1.csv` … `submission_6.csv`).  
- Reads true labels for the validation set (from `data/df_folds.csv`).  
- Performs grid or random search over linear ensemble weights.  
- Visualizes MSE or Spearman score as a function of weights.  
- Saves optimal weights to `results/best_weights.json`.

### **3.3 Script: Final Ensemble Submission**
- **Path:** `Scripts/run_ensemble.py`  
- Loads test-set predictions (`data/submission_1.csv` … `submission_6.csv`).  
- Loads optimal weights from `results/best_weights.json`.  
- Computes weighted average for each of the six score columns.  
- Outputs final `data/submission.csv` ready for Kaggle submission.

## **4. Repository Structure**
    FB3-ENSEMBLE-WEIGHT-TUNING/
    ├── data/
    │   ├── df_folds.csv           # Optional: validation folds & true labels
    │   ├── inference.log          # Inference & tuning logs
    │   ├── submission_1.csv … submission_6.csv   # Model predictions
    │   └── submission.csv         # Final ensemble submission
    │
    ├── Notebook/
    │   ├── fb3-deberta-v3-large-baseline-inference.ipynb
    │   └── fb3-deberta-family-inference-weight-tune.ipynb
    │
    ├── Scripts/
    │   └── run_ensemble.py
    │
    ├── Tokenizer/
    │   ├── added_tokens.json
    │   ├── config.json
    │   ├── special_tokens_map.json
    │   ├── spm.model
    │   ├── tokenizer_config.json
    │   └── tokenizer.json
    │
    ├── .gitignore
    └── README.md

## **5. Installation & Dependencies**

```bash
git clone <your-repo-url>
cd FB3-ENSEMBLE-WEIGHT-TUNING
pip install -r requirements.txt
``` 


## **6. Usage**

### **6.1 Prepare Data**
1.	Place your six model prediction CSVs in data/ and name them submission_1.csv … submission_6.csv.
2.	Ensure you have validation-set true labels in data/df_folds.csv (or adjust the notebook accordingly).

### **6.2 Tune Weights on Validation Set**

Launch the tuning notebook:

```bash
jupyter notebook Notebook/fb3-deberta-family-inference-weight-tune.ipynb
```

## **7. Results & Evaluation**

- Kaggle Final Rank: Silver medal (126 / 2,654).

- Public & Private LB scores: See data/inference.log for details.

## **8. Acknowledgments & References**

- Competition Hosts: Vanderbilt University & The Learning Agency Lab
- Funding: Bill & Melinda Gates Foundation, Schmidt Futures, Chan Zuckerberg Initiative
- Libraries & Models:
  - Hugging Face Transformers (DeBERTa)
  - Scikit-learn, pandas, numpy
- Additional Resources:
  - [Feedback Prize Competition Page](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)