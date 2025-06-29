# 🏡 Real Estate Explorer

A smart Streamlit-powered dashboard for **scraping**, **analyzing**, **predicting**, and **clustering** real estate listings from [Aqarmap](https://aqarmap.com).  
This project leverages **NLP**, **machine learning**, and **unsupervised clustering** to give valuable insights into property trends and pricing.

---

## 🚀 Features

- 🔍 **Web Scraping** — Collect real-time listings from Aqarmap
- ⚙️ **Preprocessing** — Clean and structure raw data
- 📊 **EDA** — Interactive charts and heatmaps for data exploration
- 🤖 **Price Prediction** — Using Random Forest, XGBoost, and CatBoost
- 🧠 **NLP Embeddings** — Extract semantic meaning from title and location
- 🧩 **Clustering** — Auto-select KMeans clusters with silhouette analysis
- 📈 **Interactive UI** — Powered by Streamlit for fast insights

---

### 📁 Project Structure

```text
real-estate-explorer/
│
├── 📂 data/                  # Folder for scraped or preprocessed CSV data
│   └── aqarmap_listings.csv
│
├── 📂 visuals/               # Folder for saved plots, charts, or screenshots
│   └── dashboard_screenshot.png
│
├── 📂 .venv/                 # (Optional) Your virtual environment folder (usually in .gitignore)
│
├── 📜 main.py                # Streamlit app – runs the full dashboard
├── 📜 scraper.py             # Handles web scraping logic from Aqarmap
├── 📜 preprocessing.py       # Cleans and processes raw data
├── 📜 model.py               # Machine learning models (RandomForest, XGBoost, etc.)
├── 📜 clustering.py          # Clustering logic using NLP features
├── 📜 nlp_features.py        # NLP feature extraction (SBERT, TF-IDF)
├── 📜 EDA.py                 # Exploratory Data Analysis visuals
├── 📜 requirements.txt       # Python dependencies
├── 📜 README.md              # Project overview and usage guide
│
└── 📜 .gitignore             # Files/folders to ignore in version control



## 📸 Visuals

### 📌 Dashboard Overview
![Screenshot 2025-06-29 154554](https://github.com/user-attachments/assets/7c92eb7b-2e21-477b-a836-84b5f0773257)

### 📈 Price Per Meter Squared Distribution
![494a73abfb91e3a10b4f86f800c57f10f044ac87e5b15a4ec00bd5c3](https://github.com/user-attachments/assets/41dd260c-9abb-4dff-baea-1bfba2be9551)

### 🤖 Model Performance
  ![Screenshot 2025-06-29 155555](https://github.com/user-attachments/assets/d39e6013-17fe-46ce-9d43-799a9e3b11cf)


### 🧩 Cluster Visualization
![Screenshot 2025-06-29 161215](https://github.com/user-attachments/assets/12d45913-dac7-4c44-80a8-bd895c405ea9)

Some Extra Visuals

### Correlation Heatmap
![4383517be88830be17540744d3724ffacbc72510944d15f5d179e2d5](https://github.com/user-attachments/assets/81320aeb-9c39-419d-9b19-d3cb8db0e360)

### Distribution of Bathrooms
![9edf0334b553da9ea6ce256428028fe7b0ca19cc48faeddaf9efa8d9](https://github.com/user-attachments/assets/ea365bd0-bdcf-49be-b7dc-ef171fd8ffd4)

### Distribution of Area
![34024546f7d02bf4f2999c74c924417594aac4f26332848396d5c170](https://github.com/user-attachments/assets/3f69474b-fc15-4fa8-8740-5f65d7caed0a)


---

## 🧠 Technologies Used

- `Python 3.10+`
- `Streamlit`
- `BeautifulSoup` (for web scraping)
- `Scikit-learn`, `XGBoost`, `CatBoost`
- `Pandas`, `Matplotlib`, `Seaborn`
- `SentenceTransformers` (for NLP embeddings)

---
