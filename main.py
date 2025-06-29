import streamlit as st
import pandas as pd
from scraper import fetch_listings_from_page
from preprocessing import preprocess
from model import run_models
from clustering import run_clustering, plot_clusters
from EDA import run_eda
from nlp_features import enrich_with_nlp

st.set_page_config(page_title="Aqarmap Explorer", layout="wide")

st.title("🏡 Aqarmap Real Estate Explorer")
st.markdown("A smart dashboard for scraping, analyzing, modeling and visualizing real estate data from Aqarmap.")

# --- Scraping Section ---
st.sidebar.header("📥 Data Collection")
if st.sidebar.button("Scrape Latest Listings"):
    with st.spinner("Scraping data from Aqarmap..."):
        all_listings = []
        for page in range(1, 20):
            listings = fetch_listings_from_page(page)
            all_listings.extend(listings)

        df_raw = pd.DataFrame(all_listings)
        df_raw.to_csv("data/aqarmap_listings.csv", index=False)
        st.success("✅ Scraping completed and data saved to 'data/aqarmap_listings.csv'.")
        st.dataframe(df_raw.head())

# --- Load & Preprocess Section ---
df = None
nlp_df = None  # Holds enriched version
st.sidebar.header("⚙️ Data Loading & Preprocessing")

if st.sidebar.checkbox("Load and preprocess saved data"):
    with st.spinner("Preprocessing data..."):
        df = preprocess("data/aqarmap_listings.csv")
        st.success("✅ Data loaded and preprocessed successfully.")
        st.dataframe(df.head())

        with st.spinner("Embedding NLP features..."):
            nlp_df = enrich_with_nlp(df.copy())
            st.success("✅ NLP enrichment complete.")

# --- Main Analysis Tabs ---
if df is not None and nlp_df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧠 Prediction", "🧩 Clustering", "📁 Raw Data"])

    # --- EDA Tab ---
    with tab1:
        st.subheader("📊 Exploratory Data Analysis")
        eda_figures = run_eda(df)

        st.subheader("🔹 Summary Statistics")
        st.pyplot(eda_figures["summary"])

        st.subheader("🔹 Distributions")
        for name, fig in eda_figures["distributions"]:
            st.markdown(f"**{name}**")
            st.pyplot(fig)

        st.subheader("🔹 Correlation Heatmap")
        st.pyplot(eda_figures["correlation"])

        st.subheader("🔹 Scatter Relationships")
        for name, fig in eda_figures["scatter"]:
            st.markdown(f"**{name}**")
            st.pyplot(fig)

        st.subheader("🔹 Boxplot by Location")
        st.pyplot(eda_figures["boxplot_location"])

        st.subheader("🔹 Bedroom / Bathroom Counts")
        for name, fig in eda_figures["room_counts"]:
            st.markdown(f"**{name}**")
            st.pyplot(fig)

    # --- Modeling Tab ---
    with tab2:
        st.subheader("🧠 Predictive Modeling")
        if st.button("Run Machine Learning Models"):
            results = run_models(nlp_df, target="Price")
            for metrics, fig in results:
                st.markdown(f"### {metrics['Model']}")
                st.write(f"**MAE:** {metrics['MAE']:.2f}")
                st.write(f"**RMSE:** {metrics['RMSE']:.2f}")
                st.write(f"**R² Score:** {metrics['R2']:.3f}")
                st.pyplot(fig)

    # --- Clustering Tab ---
    with tab3:
        st.subheader("🧩 Cluster Real Estate Listings")

        if st.button("Run Clustering Algorithm"):
            nlp_cols = [col for col in nlp_df.columns if col.startswith("title_") or col.startswith("location_")]

            if not nlp_cols:
                st.warning("⚠️ NLP features not found. Please run the modeling step first.")
            else:
                try:
                    clustered_df, model, best_k, best_score = run_clustering(nlp_df.copy(), nlp_cols=nlp_cols)
                    fig = plot_clusters(clustered_df, nlp_cols)
                    st.success(f"✅ Clustering complete. Best number of clusters: **{best_k}** (Silhouette Score: {best_score:.2f})")
                    st.pyplot(fig)
                    st.dataframe(clustered_df.head())
                except Exception as e:
                    st.error(f"❌ Clustering failed: {str(e)}")

    # --- Raw Data Tab ---
    with tab4:
        st.subheader("📁 Preview Raw Preprocessed Data")
        st.dataframe(nlp_df)

else:
    st.info("☝️ Load or scrape data first to continue.")
