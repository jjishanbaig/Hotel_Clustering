from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "final_hotel_clustering_model.pkl"
SCALER_PATH = ROOT_DIR / "models" / "hotel_scaler.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "hotel_features.pkl"
PCA_PATH = ROOT_DIR / "models" / "hotel_pca.pkl"
HOTEL_DATA_PATH = ROOT_DIR / "results" / "hotel_clusters.csv"
CLUSTER_PROFILE_PATH = ROOT_DIR / "results" / "cluster_profile.csv"
ALGORITHM_COMPARISON_PATH = ROOT_DIR / "results" / "clustering_algorithm_comparison.csv"

DEFAULT_FEATURES = [
    "Average_Score",
    "Reviewer_Score",
    "sentiment_balance",
    "log_review_volume",
    "Total_Number_of_Reviews_Reviewer_Has_Given",
]


st.set_page_config(
    page_title="Hotel Clustering Dashboard",
    page_icon="🏨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #0f172a;
    }
    .subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #dfe7f1;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        font-weight: 600;
        color: #475569;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.45rem;
    }
    .section-header {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        color: #0f172a;
        font-weight: 700;
    }
    .cluster-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        border: 1px solid transparent;
    }
    .cluster-0 {
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
        border-color: rgba(59, 130, 246, 0.25);
    }
    .cluster-1 {
        background: rgba(16, 185, 129, 0.12);
        color: #047857;
        border-color: rgba(16, 185, 129, 0.25);
    }
    .info-note {
        background: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 0.9rem 1rem;
        border-radius: 0.5rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    pca = joblib.load(PCA_PATH)
    return model, scaler, feature_names, pca


@st.cache_data
def load_hotel_data() -> pd.DataFrame:
    df = pd.read_csv(HOTEL_DATA_PATH)
    if "Cluster" in df.columns:
        df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce")
    return df


def normalize_index_style_column(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    normalized = df.copy()
    if target_column in normalized.columns:
        return normalized

    unnamed_columns = [col for col in normalized.columns if str(col).startswith("Unnamed:")]
    if unnamed_columns:
        normalized = normalized.rename(columns={unnamed_columns[0]: target_column})

    if target_column not in normalized.columns:
        normalized.insert(0, target_column, normalized.index.to_numpy())

    return normalized


@st.cache_data
def load_cluster_profile() -> pd.DataFrame:
    df = pd.read_csv(CLUSTER_PROFILE_PATH)
    return normalize_index_style_column(df, "Cluster")


@st.cache_data
def load_algorithm_comparison() -> pd.DataFrame:
    df = pd.read_csv(ALGORITHM_COMPARISON_PATH)
    return normalize_index_style_column(df, "Algorithm")


def format_cluster_label(cluster_value) -> str:
    return f"Cluster {int(cluster_value)}"


def render_metric_card(label: str, value: str, columns_container):
    with columns_container:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def forecast_cluster(df: pd.DataFrame, scaler, model, feature_order: list[str]):
    missing_columns = [col for col in feature_order if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    working = df.loc[:, feature_order].copy()
    for col in feature_order:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if working.isnull().any().any():
        invalid = [col for col in feature_order if working[col].isnull().any()]
        raise ValueError(f"Invalid or missing numeric values found in: {', '.join(invalid)}")

    scaled = scaler.transform(working)
    prediction = model.predict(scaled)
    return prediction.astype(int)


def render_dashboard_page():
    hotel_df = load_hotel_data()
    algorithm_df = load_algorithm_comparison()
    kmeans_row = algorithm_df[algorithm_df["Algorithm"] == "K-Means"].iloc[0]

    total_hotels = int(hotel_df["Hotel_Name"].nunique())
    cluster_counts = hotel_df["Cluster"].value_counts().sort_index()
    cluster_distribution = pd.DataFrame(
        {
            "Cluster": [format_cluster_label(cluster) for cluster in cluster_counts.index],
            "Count": cluster_counts.values.astype(int),
        }
    )

    st.markdown('<div class="app-title">Hotel Clustering Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive Hotel Segmentation using Unsupervised Machine Learning</div>', unsafe_allow_html=True)

    metric_columns = st.columns(5)
    render_metric_card("Total Hotels", f"{total_hotels:,}", metric_columns[0])
    render_metric_card("Number of Clusters", "2", metric_columns[1])
    render_metric_card("Algorithm", "K-Means", metric_columns[2])
    render_metric_card("Silhouette Score", f"{float(kmeans_row['Silhouette']):.4f}", metric_columns[3])
    render_metric_card("Davies-Bouldin Score", f"{float(kmeans_row['Davies_Bouldin']):.4f}", metric_columns[4])

    st.markdown("<div class='section-header'>Cluster Distribution</div>", unsafe_allow_html=True)
    fig = px.bar(
        cluster_distribution,
        x="Cluster",
        y="Count",
        color="Cluster",
        title="Cluster Distribution",
        text="Count",
        color_discrete_sequence=["#3B82F6", "#10B981"],
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Cluster Comparison</div>", unsafe_allow_html=True)
    cluster_summary = hotel_df.groupby("Cluster")[DEFAULT_FEATURES].mean().round(4).reset_index()
    cluster_summary["Cluster"] = cluster_summary["Cluster"].map(format_cluster_label)
    st.dataframe(cluster_summary, use_container_width=True)

    st.markdown("<div class='section-header'>Download Results</div>", unsafe_allow_html=True)
    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            label="Download Final Clustered Dataset",
            data=hotel_df.to_csv(index=False).encode("utf-8"),
            file_name="hotel_clusters.csv",
            mime="text/csv",
        )
    with download_cols[1]:
        st.download_button(
            label="Download Cluster Profile",
            data=load_cluster_profile().to_csv(index=False).encode("utf-8"),
            file_name="cluster_profile.csv",
            mime="text/csv",
        )
    with download_cols[2]:
        st.download_button(
            label="Download Algorithm Comparison",
            data=load_algorithm_comparison().to_csv(index=False).encode("utf-8"),
            file_name="clustering_algorithm_comparison.csv",
            mime="text/csv",
        )


def render_hotel_explorer_page():
    hotel_df = load_hotel_data()
    hotel_names = sorted(hotel_df["Hotel_Name"].dropna().tolist())

    st.markdown('<div class="section-header">Hotel Explorer</div>', unsafe_allow_html=True)
    selected_hotel = st.selectbox("Search and select a hotel", hotel_names, index=0)
    row = hotel_df[hotel_df["Hotel_Name"] == selected_hotel].iloc[0]
    cluster_value = int(row["Cluster"])
    cluster_name = format_cluster_label(cluster_value)

    cluster_badge = f"<div class='cluster-badge cluster-{cluster_value}'>{cluster_name}</div>"
    st.markdown(cluster_badge, unsafe_allow_html=True)

    feature_columns = st.columns(3)
    metrics = [
        ("Hotel Name", selected_hotel),
        ("Cluster", cluster_name),
        ("Average Score", f"{row['Average_Score']:.4f}"),
        ("Reviewer Score", f"{row['Reviewer_Score']:.4f}"),
        ("Sentiment Balance", f"{row['sentiment_balance']:.4f}"),
        ("Log Review Volume", f"{row['log_review_volume']:.4f}"),
        (
            "Total Number of Reviews Reviewer Has Given",
            f"{row['Total_Number_of_Reviews_Reviewer_Has_Given']:.4f}",
        ),
    ]

    for idx, (label, value) in enumerate(metrics):
        with feature_columns[idx % 3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.1rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    cluster_means = hotel_df.groupby("Cluster")[DEFAULT_FEATURES].mean()
    selected_values = row[DEFAULT_FEATURES]
    cluster_average = cluster_means.loc[cluster_value]
    comparison_df = pd.DataFrame(
        {
            "Selected Hotel": selected_values.round(4),
            "Cluster Average": cluster_average.round(4),
            "Difference": (selected_values - cluster_average).round(4),
        }
    )
    st.markdown("<div class='section-header'>Position Relative to Cluster Average</div>", unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True)


def render_cluster_analysis_page():
    hotel_df = load_hotel_data()
    cluster_profile = load_cluster_profile()
    feature_heatmap = cluster_profile.set_index("Cluster").T
    feature_heatmap.columns = [format_cluster_label(col) for col in feature_heatmap.columns]

    st.markdown('<div class="section-header">Cluster Profile</div>', unsafe_allow_html=True)
    st.dataframe(cluster_profile, use_container_width=True)

    st.markdown('<div class="section-header">Feature Heatmap</div>', unsafe_allow_html=True)
    heatmap_fig = px.imshow(
        feature_heatmap,
        labels=dict(x="Cluster", y="Feature", color="Value"),
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    heatmap_fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown('<div class="section-header">Feature Comparison by Cluster</div>', unsafe_allow_html=True)
    selected_feature = st.selectbox("Select feature to visualize", DEFAULT_FEATURES)
    box_fig = px.box(
        hotel_df,
        x="Cluster",
        y=selected_feature,
        color="Cluster",
        category_orders={"Cluster": sorted(hotel_df["Cluster"].dropna().unique().tolist())},
        title=f"{selected_feature} by Cluster",
        template="plotly_white",
    )
    box_fig.update_layout(height=420)
    st.plotly_chart(box_fig, use_container_width=True)

    st.markdown('<div class="section-header">Detailed Feature Views</div>', unsafe_allow_html=True)
    feature_columns = st.columns(5)
    for idx, metric in enumerate(DEFAULT_FEATURES):
        with feature_columns[idx]:
            temp_fig = px.box(
                hotel_df,
                x="Cluster",
                y=metric,
                color="Cluster",
                category_orders={"Cluster": sorted(hotel_df["Cluster"].dropna().unique().tolist())},
                template="plotly_white",
                title=metric,
                color_discrete_sequence=["#3B82F6", "#10B981"],
            )
            temp_fig.update_layout(showlegend=False, height=250, margin=dict(t=25, r=10, b=10, l=10))
            st.plotly_chart(temp_fig, use_container_width=True)


def render_geographic_analysis_page():
    hotel_df = load_hotel_data().copy()
    hotel_df = hotel_df.dropna(subset=["lat", "lng"]).copy()
    hotel_df["Cluster"] = pd.to_numeric(hotel_df["Cluster"], errors="coerce")

    cluster_options = sorted(hotel_df["Cluster"].dropna().unique().tolist())
    selected_clusters = st.multiselect("Select clusters", cluster_options, default=cluster_options)
    score_min, score_max = st.slider("Hotel Score range", 0.0, 10.0, (0.0, 10.0), step=0.1)

    geo_df = hotel_df[
        hotel_df["Cluster"].isin(selected_clusters)
        & hotel_df["Average_Score"].between(score_min, score_max)
    ].copy()

    st.markdown('<div class="section-header">Geographic Distribution of Hotels</div>', unsafe_allow_html=True)
    fig = px.scatter_geo(
        geo_df,
        lat="lat",
        lon="lng",
        color="Cluster",
        hover_name="Hotel_Name",
        hover_data={"Average_Score": True, "Reviewer_Score": True, "Cluster": True},
        projection="natural earth",
        category_orders={"Cluster": sorted(selected_clusters)},
        template="plotly_white",
        title="Hotel Locations by Cluster",
    )
    fig.update_layout(height=600, margin={"r": 0, "t": 25, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def render_algorithm_comparison_page():
    comparison_df = load_algorithm_comparison().copy()
    comparison_df["Algorithm"] = comparison_df["Algorithm"].astype(str)
    comparison_df["Clusters"] = pd.to_numeric(comparison_df["Clusters"], errors="coerce")

    st.markdown('<div class="section-header">Algorithm Comparison</div>', unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown(
        """
        <div class="info-note">
            Silhouette and Calinski-Harabasz are higher-is-better metrics. Davies-Bouldin is lower-is-better.
            K-Means was selected because it provided a strong balance of clustering quality and balanced cluster distribution,
            even though Mean Shift reported a higher Silhouette score but produced a highly imbalanced cluster pattern.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for metric in ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]:
        legend_text = "Higher is generally better" if metric in ["Silhouette", "Calinski_Harabasz"] else "Lower is generally better"
        fig = px.bar(
            comparison_df,
            x="Algorithm",
            y=metric,
            color="Algorithm",
            text=metric,
            title=f"{metric.replace('_', ' ')} Comparison",
            template="plotly_white",
        )
        fig.update_layout(height=360, showlegend=False)
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=legend_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(148,163,184,0.3)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_prediction_page():
    model, scaler, feature_names, _ = load_artifacts()
    feature_names = [str(item) for item in feature_names]

    st.markdown('<div class="section-header">Hotel Cluster Prediction</div>', unsafe_allow_html=True)
    st.caption("Use the saved model and scaler to predict a hotel cluster without retraining.")

    average_score = st.number_input("Average Score", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
    reviewer_score = st.number_input("Reviewer Score", min_value=0.0, max_value=10.0, value=8.8, step=0.1)
    sentiment_balance = st.number_input("Sentiment Balance", min_value=-1.0, max_value=1.0, value=0.1, step=0.01)
    log_review_volume = st.number_input("Log Review Volume", min_value=0.0, max_value=30.0, value=6.5, step=0.1)
    total_reviews_given = st.number_input(
        "Total Number of Reviews Reviewer Has Given",
        min_value=0,
        max_value=1000,
        value=8,
        step=1,
    )

    if st.button("Predict Cluster"):
        payload = pd.DataFrame(
            [{
                "Average_Score": average_score,
                "Reviewer_Score": reviewer_score,
                "sentiment_balance": sentiment_balance,
                "log_review_volume": log_review_volume,
                "Total_Number_of_Reviews_Reviewer_Has_Given": total_reviews_given,
            }],
            columns=DEFAULT_FEATURES,
        )

        try:
            predicted = forecast_cluster(payload, scaler, model, DEFAULT_FEATURES)[0]
            cluster_name = format_cluster_label(predicted)
            st.markdown(
                f"""
                <div class='cluster-badge cluster-{predicted}'>{cluster_name}</div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:  # pragma: no cover - user-input validation path
            st.error(f"Prediction failed: {exc}")

    st.markdown("<div class='section-header'>Upload CSV for Bulk Prediction</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if uploaded_df.empty:
                st.error("The uploaded CSV is empty. Please upload a CSV with hotel feature data.")
            else:
                missing_columns = [col for col in DEFAULT_FEATURES if col not in uploaded_df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    try:
                        predictions = forecast_cluster(uploaded_df, scaler, model, DEFAULT_FEATURES)
                        prediction_output = uploaded_df.copy()
                        prediction_output["Cluster"] = predictions
                        st.dataframe(prediction_output.head(20), use_container_width=True)

                        cluster_counts = prediction_output["Cluster"].value_counts().sort_index()
                        cluster_chart = px.bar(
                            pd.DataFrame({
                                "Cluster": [format_cluster_label(cluster) for cluster in cluster_counts.index],
                                "Count": cluster_counts.values,
                            }),
                            x="Cluster",
                            y="Count",
                            color="Cluster",
                            template="plotly_white",
                            text="Count",
                            color_discrete_sequence=["#3B82F6", "#10B981"],
                        )
                        cluster_chart.update_layout(showlegend=False, height=320)
                        st.plotly_chart(cluster_chart, use_container_width=True)

                        csv_payload = prediction_output.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download hotel_cluster_predictions.csv",
                            data=csv_payload,
                            file_name="hotel_cluster_predictions.csv",
                            mime="text/csv",
                        )
                    except ValueError as exc:
                        st.error(str(exc))
        except Exception as exc:  # pragma: no cover - CSV read validation path
            st.error(f"Could not read the uploaded CSV: {exc}")


def render_about_page():
    st.markdown('<div class="section-header">About Project</div>', unsafe_allow_html=True)
    st.write(
        "The goal is to segment hotels using unsupervised machine learning based on review and hotel-level characteristics."
    )
    st.write(
        "The dataset includes 515,738 review records aggregated into approximately 1,492 hotels for clustering analysis."
    )

    st.markdown("<div class='section-header'>Feature Engineering</div>", unsafe_allow_html=True)
    st.write(
        "The clustering model uses the engineered sentiment balance metric, review volume, and log review volume to measure customer perception and hotel activity."
    )
    st.latex(r"sentiment\_balance = \frac{Positive - Negative + 1}{Positive + Negative + 1}")
    st.write("review_volume = Total_Number_of_Reviews + Additional_Number_of_Scoring")
    st.write("log_review_volume = np.log1p(review_volume)")

    st.markdown("<div class='section-header'>Algorithms Compared</div>", unsafe_allow_html=True)
    st.write("K-Means, Hierarchical, DBSCAN, GMM, Mean Shift, and Spectral clustering were evaluated.")

    st.markdown("<div class='section-header'>Final Model</div>", unsafe_allow_html=True)
    st.write(
        "The final model is K-Means with 2 clusters. It was selected based on a combination of clustering metrics, cluster balance, and interpretability."
    )

    st.markdown("<div class='section-header'>Technologies</div>", unsafe_allow_html=True)
    st.write("Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Joblib, and Jupyter Notebook")


def main():
    model, scaler, feature_names, _ = load_artifacts()
    if not feature_names:
        feature_names = DEFAULT_FEATURES

    page_names = [
        "Dashboard",
        "Hotel Explorer",
        "Cluster Analysis",
        "Geographic Analysis",
        "Algorithm Comparison",
        "Prediction",
        "About Project",
    ]
    st.sidebar.markdown("# Hotel Clustering Intelligence Dashboard")
    st.sidebar.markdown("**Final Algorithm:** K-Means")
    st.sidebar.markdown("**Clusters:** 2")
    st.sidebar.markdown("**Hotels:** 1,492")
    page = st.sidebar.radio("Navigation", page_names)

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Hotel Explorer":
        render_hotel_explorer_page()
    elif page == "Cluster Analysis":
        render_cluster_analysis_page()
    elif page == "Geographic Analysis":
        render_geographic_analysis_page()
    elif page == "Algorithm Comparison":
        render_algorithm_comparison_page()
    elif page == "Prediction":
        render_prediction_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
