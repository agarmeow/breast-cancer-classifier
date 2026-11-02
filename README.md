# Breast Tumour Classifier - CSE3141 Predictive Analytics Mini Project

Predicts whether a tumour is **malignant or benign** from cell-nucleus measurements
(Breast Cancer Wisconsin Diagnostic dataset, 569 samples). The dataset is built into
scikit-learn, so **nothing needs to be downloaded**.

The project has two parts:
1. **Analysis pipeline** (`run_analysis.py`): EDA, preprocessing, model comparison, tuning, evaluation. Saves charts and tables.
2. **Web app** (`app.py`): a usable interface where you enter measurements and get a prediction with probabilities.

## How to run (VS Code terminal, inside this folder)
```
pip install -r requirements.txt            # web app + model training
python train_model.py                      # trains and saves the final model (run once, takes seconds)
python app.py                              # starts the web app

pip install -r requirements-analysis.txt   # only if you want the analysis charts
python run_analysis.py                     # charts in outputs/plots, tables in outputs/tables
```
Then open **http://127.0.0.1:5000** in your browser. Press Ctrl+C in the terminal to stop it.

In the app, use **Load benign example** or **Load malignant example** for a quick demo,
or type your own ten measurements and press **Predict**.

## Project structure
| File | Purpose |
|---|---|
| `run_analysis.py` | Runs the full analysis pipeline |
| `train_model.py` | Trains the final tuned Logistic Regression and saves it for the app |
| `app.py` | Flask web app (page, input validation, prediction, explanation) |
| `templates/index.html`, `public/` | The web interface (HTML, CSS, JavaScript) |
| `src/config.py` | Settings (random seed, split size, folders) |
| `src/data_loader.py` | Loads the dataset |
| `src/preprocessing.py` | Missing values, duplicates, IQR outliers |
| `src/eda.py` | Statistics and plots |
| `src/features.py` | Feature selection (ANOVA F-test) and train-test split |
| `src/models.py` | Logistic Regression, LDA, QDA, Random Forest, cross-validation, grid and random search |
| `src/evaluate.py` | Accuracy, precision, recall, F1, AUC-ROC, confusion matrix, learning curve |

## How the web app predicts
1. The browser sends the ten values to the server.
2. The server validates them (numbers only, not negative, not wildly outside the training range).
3. The saved pipeline (StandardScaler then Logistic Regression) outputs the probability of malignant.
4. Malignant is shown when that probability is 50% or higher.
5. The three biggest contributors are listed (coefficient x standardised value).

## Syllabus coverage
EDA (mean, median, variance, std), outliers (IQR, box plot), visualisation, feature selection,
logistic regression, LDA, QDA, train-test split, cross-validation, F1, AUC-ROC,
grid search, random search, overfitting check, ethical considerations.

## Ethics note
Educational demo, not a medical tool. The dataset is small and from one source, so results may
not generalise. A missed malignant case (false negative) is the costly error, so recall and F1
are emphasised over accuracy.

## Deployment
The trained model (`model/model.joblib`, `model/meta.json`) is committed to the repository so the
hosted app starts instantly. `.python-version` pins Python 3.14 and `requirements.txt` pins
scikit-learn so the saved model loads on the server.
- **Render:** Build command `pip install -r requirements.txt`, start command `gunicorn app:app`.
- **Vercel:** detected automatically from `app.py`; the CSS and JavaScript are served from `public/`.
