import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path




# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "Dataset-selected-columns cars.csv"
MODEL_PATH = BASE_DIR / "used_car_price_model.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATASET_PATH)

    df = df.drop(
        columns=["Unnamed: 0", "Id"],
        errors="ignore"
    )

    return df


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not DATASET_PATH.exists():

    st.error(
        "Dataset file not found: "
        "Dataset-selected-columns cars.csv"
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        "Machine learning model not found: "
        "used_car_price_model.pkl"
    )

    st.stop()


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = load_data()
model = load_model()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚗 Car Analytics")

st.sidebar.write("AI/ML Used Car Analysis")

st.sidebar.divider()

st.sidebar.subheader("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Price Prediction",
        "Market Analysis",
        "Market Segmentation",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("📁 Project")

st.sidebar.write(
    "Used Car Price Prediction\n"
    "& Market Segmentation"
)

st.sidebar.subheader("🛠️ Technologies")

st.sidebar.write(
    "Python • Scikit-learn • Streamlit"
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Project**

    Used Car Price Prediction  
    & Market Segmentation

    **Technologies**

    Python • Scikit-learn • Streamlit
    """
)


# ============================================================
# OVERVIEW PAGE
# ============================================================

# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.title("🚗 Used Car Price Prediction & Market Segmentation")

    st.write(
        "An AI/ML application for predicting used-car prices "
        "and analyzing used-car market segments."
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Cars",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Car Brands",
            f"{df['brand'].nunique()}"
        )

    with col3:
        st.metric(
            "Cities",
            f"{df['city'].nunique()}"
        )

    with col4:
        st.metric(
            "Average Price",
            f"₹{df['price'].mean() / 100000:.2f} L"
        )

    st.divider()

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        """
        This project uses Artificial Intelligence and Machine
        Learning techniques to predict the prices of used cars
        and identify meaningful market segments.

        The system combines supervised learning for price
        prediction with unsupervised learning for market
        segmentation.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("📋 Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown(
            """
            **Dataset Characteristics**

            - 🚗 1,725 used cars
            - 🏷️ 31 car brands
            - 🏙️ 15 cities
            - ⛽ Multiple fuel types
            - 📅 Multiple manufacturing years
            - 💰 Used-car prices
            """
        )

    with info_col2:

        st.markdown(
            """
            **Important Features**

            - Manufacturing year
            - Brand
            - Model
            - Price
            - Distance travelled
            - Fuel type
            - City
            - Car age
            - Distance per year
            """
        )

    st.divider()

    # --------------------------------------------------------
    # MACHINE LEARNING APPROACH
    # --------------------------------------------------------

    st.subheader("🤖 Machine Learning Approach")

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:

        st.markdown("### 💰 Price Prediction")

        st.write(
            """
            Supervised learning models are trained to estimate
            the price of a used car based on its characteristics.
            """
        )

        st.markdown(
            """
            **Models evaluated:**

            • Linear Regression  
            • Decision Tree  
            • Random Forest  
            • Gradient Boosting
            """
        )

    with ml_col2:

        st.markdown("### 🎯 Market Segmentation")

        st.write(
            """
            K-Means clustering is used to group cars with similar
            market characteristics.
            """
        )

        st.markdown(
            """
            **Techniques used:**

            • Elbow Method  
            • Silhouette Analysis  
            • K-Means Clustering  
            • PCA Visualization
            """
        )

    st.divider()

    # --------------------------------------------------------
    # KEY MARKET INSIGHTS
    # --------------------------------------------------------

    st.subheader("💡 Key Market Insights")

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:

        st.markdown("### 💰 Price")

        st.write(
            f"""
            Average used-car price:

            **₹{df['price'].mean() / 100000:.2f} L**

            Median price:

            **₹{df['price'].median() / 100000:.2f} L**
            """
        )

    with insight_col2:

        st.markdown("### 🚗 Popular Brands")

        top_brand = df["brand"].value_counts().index[0]
        top_brand_count = df["brand"].value_counts().iloc[0]

        st.write(
            f"""
            Most frequently represented brand:

            **{top_brand}**

            Cars in dataset:

            **{top_brand_count}**
            """
        )

    with insight_col3:

        st.markdown("### 🏙️ Popular City")

        top_city = df["city"].value_counts().index[0]
        top_city_count = df["city"].value_counts().iloc[0]

        st.write(
            f"""
            City with the most listings:

            **{top_city}**

            Cars in dataset:

            **{top_city_count}**
            """
        )

    st.divider()

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

   
# ============================================================
# PRICE PREDICTION PAGE
# ============================================================

elif page == "Price Prediction":

    st.title("💰 Used Car Price Prediction")

    st.write(
        "Enter the details of a used car and the trained "
        "Random Forest model will estimate its market price."
    )

    st.divider()

    st.subheader("🚗 Enter Car Details")

    # --------------------------------------------------------
    # FIRST COLUMN
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(
            "Manufacturing Year",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=int(df["year"].max()),
            step=1
        )

        brand = st.selectbox(
            "Brand",
            sorted(df["brand"].dropna().unique())
        )

        # Show only models belonging to selected brand
        available_models = sorted(
            df.loc[
                df["brand"] == brand,
                "model_name"
            ]
            .dropna()
            .unique()
        )

        model_name = st.selectbox(
            "Car Model",
            available_models
        )

    # --------------------------------------------------------
    # SECOND COLUMN
    # --------------------------------------------------------

    with col2:

        distance = st.number_input(
            "Distance Travelled (km)",
            min_value=0,
            max_value=int(
                df["distance_travelled(kms)"].max()
            ),
            value=50000,
            step=1000
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(
                df["fuel_type"]
                .dropna()
                .unique()
            )
        )

        city = st.selectbox(
            "City",
            sorted(
                df["city"]
                .dropna()
                .unique()
            )
        )

    st.divider()

    # --------------------------------------------------------
    # SELECTED CAR SUMMARY
    # --------------------------------------------------------

    st.subheader("📋 Selected Car")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.write(
        f"**Brand:** {brand}"
    )

    summary_col2.write(
        f"**Model:** {model_name}"
    )

    summary_col3.write(
        f"**City:** {city}"
    )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Car Price",
        type="primary",
        use_container_width=True
    ):

        # Feature engineering used during model training

        REFERENCE_YEAR = 2021

        car_age = REFERENCE_YEAR - year

        distance_per_year = (
            distance / max(car_age, 1)
        )

        # Create input DataFrame
        input_data = pd.DataFrame(
            {
                "year": [year],
                "car_age": [car_age],
                "distance_travelled(kms)": [distance],
                "distance_per_year": [distance_per_year],
                "brand": [brand],
                "model_name": [model_name],
                "fuel_type": [fuel_type],
                "city": [city]
            }
        )

        try:

            prediction = model.predict(
                input_data
            )[0]

            st.success(
                "Prediction generated successfully! 🎉"
            )

            st.subheader(
                "💰 Estimated Used Car Price"
            )

            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )

            result_col1.metric(
                "Predicted Price",
                f"₹{prediction:,.0f}"
            )

            result_col2.metric(
                "Price in Lakhs",
                f"₹{prediction / 100000:.2f} L"
            )

            result_col3.metric(
                "Car Age",
                f"{car_age} years"
            )

            st.info(
                "The predicted value is an estimate "
                "generated by the trained Random Forest "
                "machine learning model."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# MARKET ANALYSIS PAGE
# ============================================================
# ============================================================
# MARKET ANALYSIS PAGE
# ============================================================

elif page == "Market Analysis":

    st.title("📊 Used Car Market Analysis")

    st.write(
        "Explore price patterns and market trends across "
        "brands, fuel types, cities, manufacturing years "
        "and distance travelled."
    )

    st.divider()

    # ========================================================
    # KEY MARKET STATISTICS
    # ========================================================

    st.subheader("📌 Key Market Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Price",
        f"₹{df['price'].mean() / 100000:.2f} L"
    )

    col2.metric(
        "Median Price",
        f"₹{df['price'].median() / 100000:.2f} L"
    )

    col3.metric(
        "Minimum Price",
        f"₹{df['price'].min() / 100000:.2f} L"
    )

    col4.metric(
        "Maximum Price",
        f"₹{df['price'].max() / 100000:.2f} L"
    )

    st.divider()

    # ========================================================
    # 1. PRICE DISTRIBUTION
    # ========================================================

    st.subheader("💰 Price Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["price"],
        bins=40
    )

    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Number of Cars")
    ax.set_title("Distribution of Used Car Prices")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "The price distribution is right-skewed because a small "
        "number of premium and luxury cars have substantially "
        "higher prices than most cars."
    )

    st.divider()

    # ========================================================
    # 2. BRAND-WISE AVERAGE PRICE
    # ========================================================

    st.subheader("🚘 Brand-wise Average Price")

    brand_price = (
        df.groupby("brand")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    brand_price.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Average Price (₹)")
    ax.set_ylabel("Brand")
    ax.set_title("Top 10 Brands by Average Used Car Price")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ========================================================
    # 3. FUEL TYPE ANALYSIS
    # ========================================================

    st.subheader("⛽ Fuel Type Analysis")

    fuel_price = (
        df.groupby("fuel_type")["price"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    fuel_price.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Average Price (₹)")
    ax.set_title("Average Used Car Price by Fuel Type")

    plt.xticks(rotation=30)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ========================================================
    # 4. CITY-WISE ANALYSIS
    # ========================================================

    st.subheader("🏙️ City-wise Average Price")

    city_price = (
        df.groupby("city")["price"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    city_price.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Average Price (₹)")
    ax.set_ylabel("City")
    ax.set_title("Average Used Car Price by City")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ========================================================
    # 5. YEAR VS PRICE
    # ========================================================

    st.subheader("📅 Manufacturing Year vs Average Price")

    year_price = (
        df.groupby("year")["price"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        year_price.index,
        year_price.values,
        marker="o"
    )

    ax.set_xlabel("Manufacturing Year")
    ax.set_ylabel("Average Price (₹)")
    ax.set_title(
        "Relationship Between Manufacturing Year and Price"
    )

    ax.grid(True)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ========================================================
    # 6. DISTANCE VS PRICE
    # ========================================================

    st.subheader("🛣️ Distance Travelled vs Price")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df["distance_travelled(kms)"],
        df["price"],
        alpha=0.5
    )

    ax.set_xlabel("Distance Travelled (km)")
    ax.set_ylabel("Price (₹)")
    ax.set_title(
        "Distance Travelled vs Used Car Price"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ========================================================
    # MARKET INSIGHTS
    # ========================================================

    st.subheader("💡 Market Insights")

    highest_brand = (
        df.groupby("brand")["price"]
        .mean()
        .idxmax()
    )

    highest_brand_price = (
        df.groupby("brand")["price"]
        .mean()
        .max()
    )

    highest_city = (
        df.groupby("city")["price"]
        .mean()
        .idxmax()
    )

    highest_city_price = (
        df.groupby("city")["price"]
        .mean()
        .max()
    )

    st.markdown(
        f"""
        - **Highest average-priced brand:** {highest_brand}
          (approximately ₹{highest_brand_price / 100000:.2f} L)

        - **Highest average-priced city:** {highest_city}
          (approximately ₹{highest_city_price / 100000:.2f} L)

        - The used-car price distribution is strongly
          right-skewed because premium cars have much higher
          prices than mainstream cars.

        - Manufacturing year and distance travelled are important
          factors associated with used-car prices.

        - Brand, fuel type and city also contribute to differences
          in the observed used-car market.
        """
    )


# ============================================================
# MARKET SEGMENTATION PAGE
# ============================================================

# ============================================================
# MARKET SEGMENTATION PAGE
# ============================================================

elif page == "Market Segmentation":

    st.title("🎯 Used Car Market Segmentation")

    st.write(
        "K-Means clustering is used to divide the used-car "
        "market into meaningful groups based on price, "
        "distance travelled and car age."
    )

    st.divider()

    # ========================================================
    # LOAD CLUSTERING MODELS
    # ========================================================

    KMEANS_PATH = BASE_DIR / "used_car_kmeans.pkl"
    SCALER_PATH = BASE_DIR / "cluster_scaler.pkl"
    PCA_PATH = BASE_DIR / "pca_model.pkl"

    if not KMEANS_PATH.exists():
        st.error("used_car_kmeans.pkl not found.")
        st.stop()

    if not SCALER_PATH.exists():
        st.error("cluster_scaler.pkl not found.")
        st.stop()

    if not PCA_PATH.exists():
        st.error("pca_model.pkl not found.")
        st.stop()

    # --------------------------------------------------------
    # LOAD SAVED MODELS
    # --------------------------------------------------------

    kmeans_model = joblib.load(KMEANS_PATH)
    cluster_scaler = joblib.load(SCALER_PATH)
    pca_model = joblib.load(PCA_PATH)

    # ========================================================
    # PREPARE CLUSTERING DATA
    # ========================================================

    REFERENCE_YEAR = 2021

    clustering_df = df.copy()

    clustering_df["car_age"] = (
        REFERENCE_YEAR - clustering_df["year"]
    )

    clustering_features = [
        "price",
        "distance_travelled(kms)",
        "car_age"
    ]

    cluster_data = clustering_df[
        clustering_features
    ].copy()

    cluster_scaled = cluster_scaler.transform(
        cluster_data
    )

    clustering_df["cluster"] = (
        kmeans_model.predict(cluster_scaled)
    )

    # ========================================================
    # CLUSTER OVERVIEW
    # ========================================================

    st.subheader("📊 Cluster Overview")

    cluster_counts = (
        clustering_df["cluster"]
        .value_counts()
        .sort_index()
    )

    col1, col2, col3, col4 = st.columns(4)

    for col, cluster_number in zip(
        [col1, col2, col3, col4],
        range(4)
    ):

        count = cluster_counts.get(
            cluster_number,
            0
        )

        col.metric(
            f"Cluster {cluster_number}",
            f"{count:,} Cars"
        )

    st.divider()

    # ========================================================
    # CLUSTER PROFILE
    # ========================================================

    st.subheader("📋 Cluster Profiles")

    cluster_profile = (
        clustering_df
        .groupby("cluster")
        .agg(
            Cars=("price", "count"),
            Average_Price=("price", "mean"),
            Median_Price=("price", "median"),
            Average_Distance=(
                "distance_travelled(kms)",
                "mean"
            ),
            Average_Car_Age=(
                "car_age",
                "mean"
            )
        )
        .reset_index()
    )

    # Convert values for display

    display_profile = cluster_profile.copy()

    display_profile["Average_Price"] = (
        display_profile["Average_Price"]
        .apply(lambda x: f"₹{x:,.0f}")
    )

    display_profile["Median_Price"] = (
        display_profile["Median_Price"]
        .apply(lambda x: f"₹{x:,.0f}")
    )

    display_profile["Average_Distance"] = (
        display_profile["Average_Distance"]
        .apply(lambda x: f"{x:,.0f} km")
    )

    display_profile["Average_Car_Age"] = (
        display_profile["Average_Car_Age"]
        .apply(lambda x: f"{x:.1f} years")
    )

    display_profile.columns = [
        "Cluster",
        "Cars",
        "Average Price",
        "Median Price",
        "Average Distance",
        "Average Car Age"
    ]

    st.dataframe(
        display_profile,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # CLUSTER SIZE CHART
    # ========================================================

    st.subheader("📊 Cluster Size")

    fig, ax = plt.subplots(figsize=(8, 5))

    cluster_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Cars")
    ax.set_title("Number of Cars in Each Market Segment")

    plt.xticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ========================================================
    # AVERAGE PRICE BY CLUSTER
    # ========================================================

    st.subheader("💰 Average Price by Cluster")

    avg_cluster_price = (
        clustering_df
        .groupby("cluster")["price"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    avg_cluster_price.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Average Price (₹)")
    ax.set_title("Average Used Car Price by Market Segment")

    plt.xticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ========================================================
    # PCA VISUALIZATION
    # ========================================================

    st.subheader("🔬 PCA Visualization of Market Segments")

    pca_result = pca_model.transform(
        cluster_scaled
    )

    pca_df = pd.DataFrame(
        {
            "PCA1": pca_result[:, 0],
            "PCA2": pca_result[:, 1],
            "Cluster": clustering_df["cluster"].values
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster_number in sorted(
        pca_df["Cluster"].unique()
    ):

        cluster_points = pca_df[
            pca_df["Cluster"] == cluster_number
        ]

        ax.scatter(
            cluster_points["PCA1"],
            cluster_points["PCA2"],
            alpha=0.6,
            label=f"Cluster {cluster_number}"
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.set_title(
        "K-Means Market Segments in PCA Space"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "The first two PCA components explain approximately "
        "80.76% of the variance in the clustering features."
    )

    st.divider()

    # ========================================================
    # CLUSTER CHARACTERISTICS
    # ========================================================

    st.subheader("🔎 Segment Characteristics")

    cluster_descriptions = {
        0: (
            "Older / lower-priced / higher-use cars. "
            "This segment contains a large number of "
            "mainstream used vehicles."
        ),

        1: (
            "Premium / newer / lower-use cars. "
            "This segment is dominated by higher-priced "
            "premium brands."
        ),

        2: (
            "Extreme high-mileage segment. "
            "Only a small number of vehicles fall into "
            "this segment."
        ),

        3: (
            "Mainstream newer / mid-priced cars. "
            "This is the largest market segment in the dataset."
        )
    }

    for cluster_number in sorted(
        clustering_df["cluster"].unique()
    ):

        st.markdown(
            f"### Cluster {cluster_number}"
        )

        st.write(
            cluster_descriptions.get(
                cluster_number,
                "Market segment identified using K-Means."
            )
        )

    st.divider()

    # ========================================================
    # TOP BRANDS IN EACH CLUSTER
    # ========================================================

    st.subheader("🚘 Top Brands in Each Segment")

    for cluster_number in sorted(
        clustering_df["cluster"].unique()
    ):

        cluster_data = clustering_df[
            clustering_df["cluster"] == cluster_number
        ]

        top_brands = (
            cluster_data["brand"]
            .value_counts()
            .head(5)
        )

        st.write(
            f"**Cluster {cluster_number}**"
        )

        st.dataframe(
            top_brands.rename(
                "Number of Cars"
            ).reset_index().rename(
                columns={
                    "index": "Brand"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ========================================================
    # INDIVIDUAL CAR SEGMENT PREDICTION
    # ========================================================

    st.subheader(
        "🚗 Find the Market Segment of a Car"
    )

    st.write(
        "Enter price, distance travelled and manufacturing "
        "year to determine which K-Means market segment "
        "the car belongs to."
    )

    seg_col1, seg_col2, seg_col3 = st.columns(3)

    with seg_col1:

        segment_price = st.number_input(
            "Car Price (₹)",
            min_value=0,
            value=800000,
            step=50000
        )

    with seg_col2:

        segment_distance = st.number_input(
            "Distance Travelled (km)",
            min_value=0,
            value=50000,
            step=5000
        )

    with seg_col3:

        segment_year = st.number_input(
            "Manufacturing Year",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=2018,
            step=1
        )

    if st.button(
        "🎯 Identify Market Segment",
        type="primary",
        use_container_width=True
    ):

        segment_age = (
            REFERENCE_YEAR - segment_year
        )

        segment_input = pd.DataFrame(
            {
                "price": [segment_price],
                "distance_travelled(kms)": [
                    segment_distance
                ],
                "car_age": [segment_age]
            }
        )

        segment_scaled = cluster_scaler.transform(
            segment_input
        )

        predicted_cluster = int(
            kmeans_model.predict(
                segment_scaled
            )[0]
        )

        st.success(
            f"This car belongs to **Cluster "
            f"{predicted_cluster}**."
        )

        st.info(
            cluster_descriptions.get(
                predicted_cluster,
                "Market segment identified using K-Means."
            )
        )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.write(
        "Comparison and evaluation of machine learning models "
        "used for used-car price prediction."
    )

    st.divider()

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.subheader("📊 Model Comparison")

    comparison_data = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting"
            ],

            "MAE (₹)": [
                451000,
                287000,
                315446,
                482000
            ],

            "RMSE (₹)": [
                1014000,
                969000,
                966439,
                1064000
            ],

            "R² Score": [
                0.694,
                0.720,
                0.722,
                0.663
            ]
        }
    )

    display_comparison = comparison_data.copy()

    display_comparison["MAE (₹)"] = (
        display_comparison["MAE (₹)"]
        .apply(lambda x: f"₹{x:,.0f}")
    )

    display_comparison["RMSE (₹)"] = (
        display_comparison["RMSE (₹)"]
        .apply(lambda x: f"₹{x:,.0f}")
    )

    display_comparison["R² Score"] = (
        display_comparison["R² Score"]
        .apply(lambda x: f"{x:.3f}")
    )

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Higher R² indicates better explanatory performance, "
        "while lower MAE and RMSE indicate lower prediction error."
    )

    st.divider()

    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader("🏆 Selected Final Model")

    best_col1, best_col2, best_col3 = st.columns(3)

    best_col1.metric(
        "Final Model",
        "Random Forest"
    )

    best_col2.metric(
        "Test R²",
        "0.722"
    )

    best_col3.metric(
        "Test MAE",
        "₹3.15 L"
    )

    st.write(
        """
        Random Forest was selected as the final model because it
        provided strong predictive performance while handling
        nonlinear relationships between car characteristics and
        used-car prices.
        """
    )

    st.divider()

    # ========================================================
    # 5-FOLD CROSS VALIDATION
    # ========================================================

    st.subheader("🔄 5-Fold Cross-Validation")

    cv_data = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Linear Regression",
                "Decision Tree",
                "Gradient Boosting"
            ],

            "Mean CV R²": [
                0.7680,
                0.7430,
                0.7269,
                0.7249
            ],

            "Std CV R²": [
                0.1053,
                0.1126,
                0.0969,
                0.1086
            ]
        }
    )

    st.dataframe(
        cv_data,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        """
        Five-fold cross-validation provides a more reliable
        estimate of model performance by evaluating the models
        across multiple training and validation splits.
        """
    )

    st.divider()

    # ========================================================
    # RECREATE TEST SET
    # ========================================================

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    import numpy as np

    REFERENCE_YEAR = 2021

    performance_df = df.copy()

    performance_df["car_age"] = (
        REFERENCE_YEAR - performance_df["year"]
    )

    performance_df["distance_per_year"] = (
        performance_df["distance_travelled(kms)"]
        /
        performance_df["car_age"].replace(0, 1)
    )

    performance_features = [
        "year",
        "car_age",
        "distance_travelled(kms)",
        "distance_per_year",
        "brand",
        "model_name",
        "fuel_type",
        "city"
    ]

    X_performance = performance_df[
        performance_features
    ]

    y_performance = performance_df["price"]

    X_train_perf, X_test_perf, y_train_perf, y_test_perf = (
        train_test_split(
            X_performance,
            y_performance,
            test_size=0.20,
            random_state=42
        )
    )

    y_pred_perf = model.predict(
        X_test_perf
    )

    # ========================================================
    # FINAL MODEL METRICS
    # ========================================================

    final_mae = mean_absolute_error(
        y_test_perf,
        y_pred_perf
    )

    final_rmse = np.sqrt(
        mean_squared_error(
            y_test_perf,
            y_pred_perf
        )
    )

    final_r2 = r2_score(
        y_test_perf,
        y_pred_perf
    )

    st.subheader("📌 Final Random Forest Metrics")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "MAE",
        f"₹{final_mae:,.0f}"
    )

    metric2.metric(
        "RMSE",
        f"₹{final_rmse:,.0f}"
    )

    metric3.metric(
        "R² Score",
        f"{final_r2:.3f}"
    )

    st.divider()

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader("🎯 Actual vs Predicted Prices")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        y_test_perf,
        y_pred_perf,
        alpha=0.6
    )

    min_value = min(
        y_test_perf.min(),
        y_pred_perf.min()
    )

    max_value = max(
        y_test_perf.max(),
        y_pred_perf.max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")

    ax.set_title(
        "Actual vs Predicted Used Car Prices"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "Points closer to the diagonal line indicate more "
        "accurate predictions."
    )

    st.divider()

    # ========================================================
    # RESIDUAL ANALYSIS
    # ========================================================

    st.subheader("📉 Residual Analysis")

    residuals = (
        y_test_perf.values -
        y_pred_perf
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        y_pred_perf,
        residuals,
        alpha=0.6
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_xlabel("Predicted Price (₹)")
    ax.set_ylabel("Residual (₹)")

    ax.set_title(
        "Residuals vs Predicted Prices"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        f"Mean residual: ₹{residuals.mean():,.0f}"
    )

    st.info(
        "Residual analysis helps identify systematic prediction "
        "errors and unusual observations."
    )

    st.divider()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader("🌳 Random Forest Feature Importance")

    try:

        fitted_preprocessor = (
            model.named_steps["preprocessor"]
        )

        random_forest = (
            model.named_steps["model"]
        )

        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        feature_names = [
            name.replace("cat__", "")
                .replace("num__", "")
            for name in feature_names
        ]

        importances = (
            random_forest
            .feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        importance_df.sort_values(
            "Importance"
        ).plot(
            kind="barh",
            x="Feature",
            y="Importance",
            ax=ax,
            legend=False
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Top 15 Random Forest Features"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.warning(
            f"Feature importance could not be displayed: {e}"
        )

    st.divider()

    # ========================================================
    # MODEL CONCLUSION
    # ========================================================

    st.subheader("📝 Model Evaluation Conclusion")

    st.markdown(
        """
        **Random Forest** was selected as the final price
        prediction model.

        The model achieved approximately:

        - **R² = 0.722**
        - **MAE ≈ ₹3.15 lakh**
        - **RMSE ≈ ₹9.66 lakh**

        The five-fold cross-validation produced a mean R² of
        approximately **0.768**, indicating that the model
        performs consistently across different data splits.

        Random Forest was preferred because it can capture
        nonlinear relationships between vehicle characteristics
        such as manufacturing year, brand, model, distance
        travelled, fuel type and city.
        """
    )