---
title: My Data Analyst
emoji: 🔥
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🧬 Nexus AI: Universal AutoML

**From Raw Data to AI Predictions in Minutes.**

Nexus AI is a powerful, no-code Machine Learning assistant built with **Streamlit**. It allows users to upload any dataset, automatically clean it, get AI-powered business insights (via Hugging Face LLMs), and train professional-grade ML models without writing a single line of code.

## 🚀 Key Features

### 1. 📂 Smart Data Loader
* Supports **CSV, Excel, JSON, XML, and Parquet**.
* Auto-detects file formats and handles parsing errors gracefully.

### 2. 📊 Automated EDA (Exploratory Data Analysis)
* Integrated **YData Profiling** for deep statistical analysis.
* Visualizes correlations, missing values, and data distributions instantly.

### 3. 🧠 The AI Brain (LLM Consultant)
* Powered by **Hugging Face Inference API** (Zephyr-7b, Mistral-7B).
* Sends a summary of your data to an LLM to get:
    * The best ML algorithms to use.
    * 3 key business insights hidden in the data.
    * Data quality warnings.

### 4. 🛠️ Smart Preprocessing Engine
* **Auto-Cleaning:** Automatically handles missing values (Mean/Mode imputation).
* **Date Parsing:** Converts date columns into Year, Month, and Day features.
* **Encoding:** Automatically Label Encodes categorical text variables.

### 5. 🤖 Training Studio
* **Supervised Learning:**
    * **Regression:** Random Forest, XGBoost, Linear Regression.
    * **Classification:** Random Forest, XGBoost, Logistic Regression.
* **Unsupervised Learning:**
    * K-Means & DBSCAN Clustering with 2D PCA Visualization.
* **Performance Metrics:** Real-time calculation of Accuracy and R2 Score.

### 6. 🔮 Prediction Engine
* Generate predictions on new data using your trained model directly in the UI.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* A Hugging Face Access Token (Free) for the AI Brain feature.

### 1. Clone the Repository
```bash
git clone [https://github.com/Shubham231005/Nexus-Universal-Auto-ML.git](https://github.com/Shubham231005/Nexus-Universal-Auto-ML.git)
cd Nexus-Universal-Auto-ML
2. Install Dependencies
Bash

pip install -r requirements.txt
3. Run the App
Bash

streamlit run app.py
📦 Tech Stack
Frontend: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn, XGBoost

LLM Integration: Hugging Face Hub (InferenceClient)

Visualization: Plotly, YData Profiling

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📜 License
MIT