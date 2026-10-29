
from pathlib import Path
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

TARGET = "charges"

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = set(FEATURES + [TARGET])


# =========================================================
# 2. CUSTOM DARK THEME
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.hero {
    padding: 28px 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #172554, #312e81);
    border: 1px solid #3730a3;
    margin-bottom: 22px;
}

.hero h1 {
    color: #ffffff;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #263247;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>💰 Insurance Cost Predictor</h1>
    <p>Explore medical insurance data, compare regression models,
    and predict estimated insurance charges.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD AND VALIDATE DATA
# =========================================================

def validate_data(df):
    df = df.copy()

    # Normalize column names
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[FEATURES + [TARGET]].copy()

    # Convert numeric columns
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical columns
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip().str.lower()
        df[col] = df[col].replace(
            {"": pd.NA, "nan": pd.NA, "none": pd.NA}
        )

    # Target cannot be missing
    df = df.dropna(subset=[TARGET])

    if df.empty:
        raise ValueError("No valid insurance charges were found.")

    if (df[TARGET] < 0).any():
        raise ValueError("Insurance charges cannot be negative.")

    if len(df) < 5:
        raise ValueError(
            "At least five valid rows are required for 5-fold CV."
        )

    return df


def get_fingerprint(df):
    """Create a fingerprint so changed data triggers retraining."""
    hashed = pd.util.hash_pandas_object(
        df, index=True
    ).values.tobytes()

    return hashlib.sha256(hashed).hexdigest()


# =========================================================
# 4. PREPROCESSING AND MODEL TRAINING
# =========================================================

@st.cache_resource(show_spinner=False)
def train_models(fingerprint, df):
    # fingerprint is part of the cache key
    X = df[FEATURES]
    y = df[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
    }

    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    pipelines = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator),
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        mae = -scores["test_mae"].mean()

        # Calculate RMSE for each fold, then average
        rmse = np.sqrt(-scores["test_mse"]).mean()

        r2 = scores["test_r2"].mean()

        results.append({
            "Model": name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R²": float(r2),
        })

        pipelines[name] = pipeline

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    # Select model with the lowest CV RMSE
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = pipelines[best_name]

    # Fit the selected model on the complete dataset
    best_pipeline.fit(X, y)

    # Save the fitted pipeline
    joblib.dump(
        best_pipeline,
        ARTIFACTS_DIR / "best_model.joblib",
    )

    # Save comparison metrics
    results_df.to_csv(
        ARTIFACTS_DIR / "model_comparison.csv",
        index=False,
    )

    # Save model metadata
    metadata = {
        "best_model": best_name,
        "features": FEATURES,
        "target": TARGET,
        "cv_folds": 5,
        "selection_metric": "Lowest mean CV RMSE",
        "metrics": {
            key: (
                float(value)
                if isinstance(value, (np.floating, np.integer))
                else value
            )
            for key, value in results_df.iloc[0].to_dict().items()
        },
    }

    with open(
        ARTIFACTS_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    return best_pipeline, results_df, best_name


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard",
        "Explore Data",
        "Model Comparison",
        "Predict Insurance Cost",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning | Scikit-learn | Streamlit"
)

st.sidebar.caption(
    "5-fold cross-validation | Random seed: 42"
)


# =========================================================
# 6. LOAD THE DATASET
# =========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload your insurance CSV",
    type=["csv"],
)

DATA_PATH = BASE_DIR / "insurance.csv"

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name

    elif DATA_PATH.exists():
        raw_df = pd.read_csv(DATA_PATH)
        dataset_name = "insurance.csv"

    else:
        st.info(
            "Upload your Medical Cost Personal Dataset CSV "
            "using the sidebar, or place insurance.csv "
            "beside app.py."
        )
        st.stop()

    df = validate_data(raw_df)

except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

st.sidebar.success(f"Dataset loaded: {dataset_name}")
st.sidebar.caption(f"{len(df):,} valid records")


# =========================================================
# 7. TRAIN AND COMPARE MODELS
# =========================================================

with st.spinner(
    "Training and comparing models using 5-fold CV..."
):
    try:
        fingerprint = get_fingerprint(df)

        best_model, results_df, best_name = train_models(
            fingerprint,
            df,
        )

    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()


# =========================================================
# 8. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df):,}",
    )

    col2.metric(
        "Average Charges",
        f"₹{df['charges'].mean():,.0f}",
    )

    col3.metric(
        "Best Model",
        best_name,
    )

    col4.metric(
        "Mean CV R²",
        f"{results_df.iloc[0]['R²']:.3f}",
    )

    st.subheader("Distribution of Insurance Charges")

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.histplot(
        data=df,
        x="charges",
        kde=True,
        ax=ax,
    )

    ax.set_xlabel("Insurance Charges (₹)")
    ax.set_ylabel("Number of People")

    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Charges by Smoking Status")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.boxplot(
            data=df,
            x="smoker",
            y="charges",
            ax=ax,
        )

        ax.set_xlabel("Smoker")
        ax.set_ylabel("Charges (₹)")

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### Average Charges by Age Group")

        age_df = df.assign(
            AgeGroup=(df["age"] // 10) * 10
        ).groupby(
            "AgeGroup",
            as_index=False
        )["charges"].mean()

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.barplot(
            data=age_df,
            x="AgeGroup",
            y="charges",
            ax=ax,
        )

        ax.set_xlabel("Age Group")
        ax.set_ylabel("Average Charges (₹)")

        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# 9. EXPLORE DATA
# =========================================================

elif page == "Explore Data":

    st.subheader("Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Dataset Preview",
        "Statistics",
        "Data Quality",
    ])

    with tab1:
        st.dataframe(
            df.head(200),
            use_container_width=True,
        )

        st.download_button(
            label="Download Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="insurance_data.csv",
            mime="text/csv",
        )

    with tab2:
        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True,
        )

        st.write(
            "Charges skewness:",
            round(df["charges"].skew(), 3),
        )

    with tab3:
        quality = pd.DataFrame({
            "Missing Values": df.isna().sum(),
            "Data Type": df.dtypes.astype(str),
            "Unique Values": df.nunique(),
        })

        st.dataframe(
            quality,
            use_container_width=True,
        )

        st.write(
            "Duplicate rows:",
            int(df.duplicated().sum()),
        )

        numeric_df = df.select_dtypes(
            include=np.number
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
        )

        ax.set_title("Numeric Feature Correlations")

        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# 10. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.subheader("5-Fold Cross-Validation Results")

    st.caption(
        "Lower MAE and RMSE are better. Higher R² is better. "
        "The best model is selected using mean CV RMSE."
    )

    display_df = results_df.copy()

    display_df["MAE"] = display_df["MAE"].map(
        lambda value: f"₹{value:,.2f}"
    )

    display_df["RMSE"] = display_df["RMSE"].map(
        lambda value: f"₹{value:,.2f}"
    )

    display_df["R²"] = display_df["R²"].map(
        lambda value: f"{value:.4f}"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    metric = st.selectbox(
        "Select metric to compare",
        ["RMSE", "MAE", "R²"],
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        data=results_df,
        x="Model",
        y=metric,
        ax=ax,
    )

    ax.set_title(f"Model Comparison — {metric}")
    ax.tick_params(axis="x", rotation=10)

    if metric in ["RMSE", "MAE"]:
        ax.set_ylabel(f"{metric} (₹)")
    else:
        ax.set_ylabel("R² Score")

    st.pyplot(fig)
    plt.close(fig)

    st.success(f"Best model: {best_name}")

    st.download_button(
        label="Download Comparison CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    st.caption(
        "The selected model is refitted on the full dataset. "
        "Cross-validation metrics are estimates, not an "
        "independent holdout test score."
    )


# =========================================================
# 11. PREDICT INSURANCE COST
# =========================================================

elif page == "Predict Insurance Cost":

    st.subheader("Predict Annual Insurance Charges")

    st.write(
        "Enter the details below to estimate insurance charges."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=25.0,
                step=0.1,
            )

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            sex_options = sorted(
                df["sex"].dropna().unique().tolist()
            )

            if not sex_options:
                st.error("No valid sex categories in the dataset.")
                st.stop()

            sex = st.selectbox(
                "Sex",
                sex_options,
            )

        with col2:
            smoker_options = sorted(
                df["smoker"].dropna().unique().tolist()
            )

            region_options = sorted(
                df["region"].dropna().unique().tolist()
            )

            if not smoker_options or not region_options:
                st.error(
                    "The smoker or region column has no valid categories."
                )
                st.stop()

            smoker = st.selectbox(
                "Smoking Status",
                smoker_options,
            )

            region = st.selectbox(
                "Region",
                region_options,
            )

        submitted = st.form_submit_button(
            "Predict Insurance Charges",
            use_container_width=True,
        )

    if submitted:

        input_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }])[FEATURES]

        prediction = float(
            best_model.predict(input_df)[0]
        )

        st.divider()

        st.markdown("### Estimated Annual Insurance Charges")

        st.metric(
            label="Predicted Cost",
            value=f"₹{max(0, prediction):,.2f}",
        )

        st.caption(
            f"Model used: {best_name}. This is a statistical "
            "estimate based on historical data, not an official "
            "insurance quote."
        )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "Insurance Cost Predictor | Python | Pandas | "
    "Scikit-learn | Streamlit"
)
