import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Aesthetic settings
sns.set(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (12, 6)


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.replace("\t", "").strip() for col in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.replace("\t", "").str.strip()
    return df


def plot_statistical_summary(df: pd.DataFrame):
    stats = df.describe().T.round(2)
    fig, ax = plt.subplots()
    sns.heatmap(stats, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=0.5, ax=ax)
    ax.set_title("Statistical Summary of Numerical Features")
    return fig


def plot_distributions(df: pd.DataFrame):
    numeric_cols = ["Price", "Price/m²", "Area", "Bedrooms", "Bathrooms"]
    figures = []
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        figures.append((col, fig))
    return figures


def plot_correlation(df: pd.DataFrame):
    numeric_cols = ["Price", "Price/m²", "Area", "Bedrooms", "Bathrooms"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def scatter_relationships(df: pd.DataFrame):
    figs = []

    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="Area", y="Price", hue="Bedrooms", alpha=0.7, ax=ax1)
    ax1.set_title("Price vs Area (Hue: Bedrooms)")
    ax1.set_xlabel("Area (m²)")
    ax1.set_ylabel("Price (EGP)")
    figs.append(("Price vs Area", fig1))

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="Area", y="Price/m²", hue="Bathrooms", alpha=0.7, ax=ax2)
    ax2.set_title("Price/m² vs Area (Hue: Bathrooms)")
    ax2.set_xlabel("Area (m²)")
    ax2.set_ylabel("Price/m² (EGP)")
    figs.append(("Price/m² vs Area", fig2))

    return figs


def boxplots_by_location(df: pd.DataFrame, top_n=10):
    top_locations = df["Location"].value_counts().head(top_n).index
    filtered_df = df[df["Location"].isin(top_locations)]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=filtered_df, x="Location", y="Price/m²", ax=ax)
    ax.set_title(f"Price/m² by Top {top_n} Locations")
    ax.set_xlabel("Location")
    ax.set_ylabel("Price per m² (EGP)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig


def count_room_features(df: pd.DataFrame):
    figs = []
    for col in ["Bedrooms", "Bathrooms"]:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        figs.append((col, fig))
    return figs


def run_eda(df: pd.DataFrame):
    """
    Return all EDA figures as dictionary.
    """
    df = clean_strings(df)

    eda_figures = {
        "summary": plot_statistical_summary(df),
        "distributions": plot_distributions(df),
        "correlation": plot_correlation(df),
        "scatter": scatter_relationships(df),
        "boxplot_location": boxplots_by_location(df),
        "room_counts": count_room_features(df),
    }

    return eda_figures
