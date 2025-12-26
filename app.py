import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import time

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBClassifier, XGBRegressor

# AI & Profiling
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

from huggingface_hub import InferenceClient

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Nexus AI: Universal AutoML", layout="wide", page_icon="🧬")

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1 {color: #FF4B4B; text-align: center;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #FF4B4B; color: white; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #262730; border-radius: 5px 5px 0 0; color: white;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #FF4B4B;}
</style>
""", unsafe_allow_html=True)

# --- 1. ROBUST DATA LOADER ---
@st.cache_data
def load_data(file):
    """Smartly loads CSV, JSON, Excel, XML, Parquet"""
    try:
        filename = file.name.lower()
        if filename.endswith('.csv'): return pd.read_csv(file)
        elif filename.endswith('.json'):
            try: return pd.read_json(file)
            except: import json; return pd.json_normalize(json.load(file))
        elif filename.endswith(('.xls', '.xlsx')): return pd.read_excel(file)
        elif filename.endswith('.xml'): return pd.read_xml(file)
        elif filename.endswith('.parquet'): return pd.read_parquet(file)
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# --- 2. FAIL-SAFE AI BRAIN (Updated) ---
def llm_advisor(df, token):
    """
    Tries multiple free models to ensure the user gets an answer
    even if one model is down or restricted.
    """
    if not token: return "⚠️ Please enter your Hugging Face Token in the sidebar."

    client = InferenceClient(token=token)

    # Prepare data summary (limit size to prevent errors)
    buffer = StringIO(); df.info(buf=buffer)
    info_str = buffer.getvalue(); head_str = df.head(3).to_string()

    prompt = f"""
    You are a Senior Data Scientist. Analyze this dataset:
    {head_str}

    Info:
    {info_str}

    Report:
    1. Best ML Algo & Why.
    2. 3 Business Insights.
    3. Data Quality Issues.
    """

    # List of models to try (Primary -> Backup -> Fallback)
    # Zephyr is currently the most reliable free model on HF Spaces
    models_to_try = [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "tiiuae/falcon-7b-instruct"
    ]

    for model in models_to_try:
        try:
            # We use a lower max_tokens count to ensure faster response on free tier
            response = client.text_generation(prompt, model=model, max_new_tokens=512)
            return f"✅ Analysis by **{model}**:\n\n{response}"
        except Exception as e:
            # If model fails, print error to logs and try the next one
            print(f"Model {model} failed: {e}")
            continue

    return "❌ All AI models are currently busy or restricted on the free tier. Please try again in 5 minutes."

# --- 3. SMART PREPROCESSING ENGINE ---
def smart_preprocess(df_input):
    """
    The Brain of the App:
    - Handles Dates -> Year/Month/Day
    - Handles Text -> Label Encoding
    - Handles NaNs -> Mean/Mode
    """
    df = df_input.copy()

    # A. Handle Date Columns
    for col in df.columns:
        # Check if column is object but looks like a date (common in CSVs)
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass # If it fails, keep as object

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_Year"] = df[col].dt.year
            df[f"{col}_Month"] = df[col].dt.month
            df[f"{col}_Day"] = df[col].dt.day
            df = df.drop(columns=[col])

    # B. Handle Missing Values (Imputation)
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    if len(cat_cols) > 0:
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # C. Handle Text (Label Encoding)
    # We re-check cat_cols after date conversion
    cat_cols = df.select_dtypes(exclude=np.number).columns

    if len(cat_cols) > 0:
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))

    return df

# --- MAIN APPLICATION UI ---

st.title("🧬 Nexus AI: Universal AutoML")
st.markdown("### 🤖 From Raw Data to AI Predictions in Minutes")

with st.sidebar:
    st.header("⚙️ Control Panel")
    uploaded_file = st.file_uploader("📂 Upload Data", type=["csv", "json", "xml", "xlsx", "parquet"])
    hf_token = st.text_input("🔑 Hugging Face Token", type="password", help="Required for the AI Brain tab")
    st.info("Get your free token from Hugging Face Settings -> Access Tokens")

# Only proceed if file is uploaded
if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Check if dataframe is valid and not None
    if df is not None:
        st.session_state['original_df'] = df

        # Create Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 EDA",
            "🧠 AI Brain",
            "🛠️ Manual Clean",
            "🚀 Training Studio",
            "🔮 Prediction"
        ])

        # --- TAB 1: EDA ---
        with tab1:
            st.header("Exploratory Data Analysis")
            if st.button("Run Deep Profiling"):
                if PROFILING_AVAILABLE:
                    with st.spinner("Analyzing Data..."):
                        # Fix: Removed dark_mode=True to prevent pydantic errors
                        pr = ProfileReport(df, explorative=True)
                        st_profile_report(pr)
                else:
                    st.error("Profiling library missing. Check requirements.txt")

        # --- TAB 2: AI BRAIN ---
        with tab2:
            st.header("🤖 AI Consultant")
            st.write("Click below to get Model Advice & Business Insights.")
            if st.button("Consult AI Models"):
                with st.spinner("AI is reading your data (Trying multiple models)..."):
                    advice = llm_advisor(df, hf_token)
                    st.success("Analysis Complete")
                    st.markdown(advice)

        # --- TAB 3: MANUAL CLEANING (Optional) ---
        with tab3:
            st.header("🛠️ Manual Data Processing")
            st.markdown("Use this tab if you want specific control. Otherwise, the 'Training Studio' cleans automatically.")

            df_manual = df.copy()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Missing Values")
                if st.button("Auto-Fill Missing"):
                    df_manual = df_manual.fillna(method='ffill').fillna(method='bfill')
                    st.success("Filled missing values!")

            with col2:
                st.subheader("Drop Columns")
                cols_to_drop = st.multiselect("Select columns to drop", df_manual.columns)
                if st.button("Drop Selected"):
                    df_manual = df_manual.drop(columns=cols_to_drop)
                    st.success("Dropped columns!")

            st.session_state['df_manual'] = df_manual
            st.dataframe(df_manual.head())

        # --- TAB 4: TRAINING STUDIO (Robust Engine) ---
        with tab4:
            st.header("🚀 Model Training Arena")

            # Use manually cleaned data if available, else original
            train_df_raw = st.session_state.get('df_manual', df)

            learning_type = st.radio("Choose Learning Type", ["Supervised (Prediction)", "Unsupervised (Clustering)"])

            if learning_type == "Supervised (Prediction)":
                c1, c2 = st.columns(2)
                target = c1.selectbox("Target Variable (Y)", train_df_raw.columns)
                features = c2.multiselect("Feature Variables (X)", [c for c in train_df_raw.columns if c != target])

                # The dropdown options
                algo_type = st.selectbox("Problem Type", ["Regression (Predict Number)", "Classification (Predict Category)"])

                # FIX: Keys now match the dropdown options EXACTLY
                model_map = {
                    "Regression (Predict Number)": {
                        "Random Forest": RandomForestRegressor(),
                        "XGBoost": XGBRegressor(),
                        "Linear Regression": LinearRegression()
                    },
                    "Classification (Predict Category)": {
                        "Random Forest": RandomForestClassifier(),
                        "XGBoost": XGBClassifier(),
                        "Logistic Regression": LogisticRegression()
                    }
                }

                # Now this line will work because the key exists
                model_name = st.selectbox("Select Model", list(model_map[algo_type].keys()))

                if st.button("Train Model"):
                    if not features or not target:
                        st.warning("Please select at least one Feature and a Target.")
                    else:
                        try:
                            # 1. Prepare Data
                            X_raw = train_df_raw[features]
                            y_raw = train_df_raw[target]

                            # 2. Smart Preprocessing
                            X_clean = smart_preprocess(X_raw)

                            # Handle Target Encoding if Classification
                            if "Classification" in algo_type and y_raw.dtype == 'object':
                                le_y = LabelEncoder()
                                y_clean = le_y.fit_transform(y_raw)
                            else:
                                y_clean = y_raw

                            # 3. Train Test Split
                            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

                            # 4. Train
                            model = model_map[algo_type][model_name]
                            model.fit(X_train, y_train)

                            # 5. Score
                            score = model.score(X_test, y_test)
                            metric_name = "R2 Score" if "Regression" in algo_type else "Accuracy"

                            st.success(f"✅ Training Successful! {metric_name}: {score:.4f}")

                            # Save to session state
                            st.session_state['model'] = model
                            st.session_state['model_features'] = features
                            st.session_state['model_type'] = algo_type

                        except Exception as e:
                            st.error(f"Training Failed: {e}")

            elif learning_type == "Unsupervised (Clustering)":
                cluster_algo = st.selectbox("Clustering Algorithm", ["K-Means", "DBSCAN"])
                features = st.multiselect("Select Features to Cluster", train_df_raw.columns)

                if st.button("Run Clustering"):
                    if features:
                        try:
                            # 1. Preprocess
                            X_clean = smart_preprocess(train_df_raw[features])

                            # 2. Run Algo
                            if cluster_algo == "K-Means":
                                k = st.slider("K (Clusters)", 2, 10, 3)
                                model = KMeans(n_clusters=k)
                                labels = model.fit_predict(X_clean)
                            else:
                                eps = st.slider("Epsilon", 0.1, 5.0, 0.5)
                                model = DBSCAN(eps=eps)
                                labels = model.fit_predict(X_clean)

                            # 3. Visualize
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_clean)

                            fig = px.scatter(
                                x=X_pca[:,0], y=X_pca[:,1],
                                color=labels.astype(str),
                                title=f"{cluster_algo} Clustering Results (PCA Projection)",
                                labels={'x': 'PCA 1', 'y': 'PCA 2'}
                            )
                            st.plotly_chart(fig)

                        except Exception as e:
                            st.error(f"Clustering Error: {e}")
                    else:
                        st.warning("Select features first.")

        # --- TAB 5: PREDICTION (Make use of trained model) ---
        with tab5:
            st.header("🔮 Prediction Engine")

            if 'model' in st.session_state:
                st.write(f"Using trained model: **{st.session_state.get('model_type', 'Unknown')}**")

                # Dynamic Input Form
                input_data = {}
                st.subheader("Enter values:")

                # Create input fields for original features
                cols = st.columns(2)
                for i, col in enumerate(st.session_state['model_features']):
                    with cols[i % 2]:
                        val = st.text_input(f"Value for {col}", key=f"pred_{col}")
                        input_data[col] = val

                if st.button("Predict Result"):
                    try:
                        # Convert input dict to DataFrame
                        input_df = pd.DataFrame([input_data])

                        # Apply EXACT same preprocessing as training
                        # (We need to force types because text_input is always string)
                        processed_input = smart_preprocess(input_df)

                        prediction = st.session_state['model'].predict(processed_input)
                        st.success(f"🎯 Prediction: {prediction[0]}")

                    except Exception as e:
                        st.error(f"Prediction Error: {e}. (Make sure inputs match the data types used in training)")
            else:
                st.info("⚠️ Please train a supervised model in the 'Training Studio' tab first.")

    else:
        st.info("👋 Welcome! Please upload a dataset to begin.")

else:
    st.write("waiting for file upload...")