# 🩺 Insurance Cost Predictor

**A Machine Learning project that predicts medical insurance charges using regression algorithms and an interactive Streamlit web application.**

## 📌 Overview

The Insurance Cost Predictor estimates annual medical insurance charges based on personal and lifestyle-related information. It uses Python and Scikit-learn to preprocess data, compare regression models, and generate predictions through a simple web interface.

This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model evaluation to interactive prediction.

## ✨ Features

- 🧑 Collect user details through an interactive input form.
- 🤖 Compare Linear Regression, Random Forest, and Gradient Boosting models.
- 📊 Evaluate models using MAE, RMSE, and R².
- 🔄 Use 5-fold cross-validation for model evaluation.
- ⚙️ Apply numerical and categorical data preprocessing.
- 💰 Predict estimated annual medical insurance charges.
- 📓 Explore data preprocessing in a Jupyter Notebook.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Data preprocessing and exploration |

## 📂 Project Structure

```text
InsuranceCostPredictor/
│
├── app.py                # Streamlit application and ML workflow
├── preprocessing.ipynb   # Data preprocessing and exploration
├── insurance.csv         # Medical insurance dataset
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Files excluded from Git
│
└── artifacts/             # Generated model files
    ├── best_model.joblib
    ├── model_metadata.json
    └── model_comparison.csv
```

*Note: The `artifacts/` directory is generated when the application successfully trains and saves the models.*

## 📊 Dataset

The project uses the **Medical Cost Personal Dataset**, containing information about individuals and their medical insurance charges.

### Input Features

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `sex` | Sex of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of dependent children |
| `smoker` | Smoking status |
| `region` | Residential region |

**Target variable:** `charges` — medical insurance charges.

## 🧠 Machine Learning Models

The application compares three regression algorithms:

1. **Linear Regression:** Provides a baseline for predicting insurance charges.
2. **Random Forest Regressor:** Uses an ensemble of decision trees to model nonlinear relationships.
3. **Gradient Boosting Regressor:** Builds trees sequentially to improve predictive performance.

### Data Preprocessing

- Missing numerical values are handled using median imputation.
- Missing categorical values are handled using most-frequent imputation.
- Numerical features are standardized.
- Categorical features are converted using one-hot encoding.

Preprocessing is integrated into the machine learning pipeline to ensure consistent transformations during model evaluation and prediction.

### Model Evaluation

The models are evaluated using 5-fold cross-validation.

- **MAE — Mean Absolute Error:** Measures the average absolute prediction error.
- **RMSE — Root Mean Squared Error:** Penalizes larger prediction errors more heavily.
- **R² Score:** Measures how well the model explains the variation in insurance charges.

The model with the lowest cross-validation RMSE is selected as the best-performing model.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/PrajwalUlhasJagdale/InsuranceCostPredictor.git
cd InsuranceCostPredictor
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly in the following steps.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The terminal will display a local URL, usually `http://localhost:8501`. Open it in your browser to access the application.

## 🚀 How to Use

1. Start the Streamlit application.
2. Enter your age, sex, BMI, number of children, smoking status, and region.
3. Click **Predict insurance charges**.
4. View the estimated annual medical insurance charges.

The application also provides information about the selected model and its evaluation process.

## 📁 Generated Artifacts

After successful execution, the application can save:

- `best_model.joblib` — Trained machine learning pipeline.
- `model_metadata.json` — Information about the selected model.
- `model_comparison.csv` — Comparative model evaluation results.

These files are generated outputs and may not be present in a fresh repository clone.

## 🔮 Future Enhancements

- Deploy the application using Streamlit Community Cloud.
- Add interactive data visualizations.
- Perform hyperparameter tuning to improve model performance.
- Add more regression algorithms for comparison.
- Evaluate model performance across different demographic groups.

## ⚠️ Disclaimer

This project is intended for educational purposes only. Predictions are estimates based on patterns in the training dataset and are not official insurance quotes, medical advice, or financial advice. Actual insurance costs may differ significantly.

## 👨‍💻 Author

**Prajwal Jagdale**

Computer Science Engineering — Artificial Intelligence & Machine Learning

Interested in Machine Learning, Data Science, and developing practical AI applications.

- GitHub: [@PrajwalUlhasJagdale](https://github.com/PrajwalUlhasJagdale)
- Project Repository: [InsuranceCostPredictor](https://github.com/PrajwalUlhasJagdale/InsuranceCostPredictor)

---

⭐ If you find this project useful, consider giving the repository a star!
