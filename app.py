import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
import joblib
import fontstyle

st.set_page_config(page_title="Pune Property Intelligence", layout="wide")

# ------------------ BACKGROUND IMAGE ------------------
page_bg = """

<style>

/* ---------------- Background ---------------- */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1651326127741-0b1bc583f7f7?q=80&w=1743&auto=format&fit=crop&ixlib=rb-4.1.0");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Remove white container box */
.block-container {
    background: transparent !important;
}

/* ---------------- MAIN PAGE TEXT WHITE ---------------- */
h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

.st-emotion-cache-104fm5o h2 {
    font-size: 1.55rem;
    font-weight: 600;
    padding: 1rem 0px;
}

.st-emotion-cache-h1zhg5 p{
    font-size: 16px;
}
/* -----------------TITLE --------*/
.st-emotion-cache-1frkdi4 h1 {
    font-size: 2.75rem;
    font-weight: 700;
    padding: 0.25rem 0px 2rem;
}
/* ---------------- SIDEBAR GLASS EFFECT ---------------- */

section[data-testid="stSidebar"] {
    background-color: rgba(71, 255, 221, 0.08) !important;
    backdrop-filter: blur(150px);
    -webkit-backdrop-filter: blur(50px);
}

/* SIDEBAR TEXT BLACK */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: black !important;
}

/* ---------------- INCREASE PARA SIZE ---------------- */

.block-container p {
    font-size: 18px !important;   /* Change size here */
    line-height: 1.8 !important;  /* Better readability */
    font-weight: 400;
}

block-container h1 {
    font-size: 42px !important;
}


/* Sidebar text black */
section[data-testid="stSidebar"] * {
    color: black !important;
}

/* ---------------- Fix Dropdown ---------------- */

/* Select box closed state */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* Dropdown menu options */
ul[role="listbox"] {
    background-color: white !important;
    color: black !important;
}

/* Dropdown individual option text */
li[role="option"] {
    color: black !important;
}

/* Number input box */
section[data-testid="stSidebar"] input {
    background-color: white !important;
    color: black !important;
}

/* ---------------Scroll bar--------------------*/
.st-bb {
    background-color: rgb(28 88 75);
}

/* --------------- Predict Button --------------------*/


/* Center the button */
div.stButton {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}

/* Button main style */
.stButton>button {
    width: 300px;             
    height: 60px;            
    background: rgba(255,255,255,0.3);
    color: white;
    border-radius: 12px;
    border: 5px solid white;
    transition: all 0.3s ease;
}

/* Hover effect */
.stButton>button:hover {
    background: #3B9797;
    color: white;
    border: 5px solid #3B9797;
}

/* Remove top white header */
[data-testid="stHeader"] {
    background: transparent !important;
}

/*------------------TOGGLE BUTTON-----------------*/

/* Sidebar toggle button (top-left ☰ icon) */
.st-emotion-cache-5r6ut5 {
    color: rgb(255 255 255);
    font-size: 1.5rem;}
    
/* ----------------- KPI Glass Card ---------------- */
.kpi-card {
    background: rgba(0, 84, 97, 0.65);  /* #005461 transparent */
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1.5px solid rgba(255,255,255,0.4);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.kpi-title {
    font-size: 20px;
    font-weight: 500;
    margin-bottom: 10px;
}

.kpi-value {
    font-size: 26px;
    font-weight: bold;
}

/*  ------------------------ Custom Divider --------------------------------------- */
hr {
    border: none !important;
    height: 2px !important;      /* thickness */
    background-color: white !important;
    margin-top: 30px !important;
    margin-bottom: 30px !important;
    opacity: 0.4 !important;;
}



</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- LOAD MODELS ---------------- #


@st.cache_resource
def load_models():

    unfurnished_model = joblib.load(r"models/1_rf_unfurnished_final_production_model.pkl")

    furnished_model = joblib.load(r"models/2_rf_furnished_full_production_pipeline.pkl")        

    with open(r"models/3_unfurnished_mae.json","r") as f:
        unfurnished_mae = json.load(f)["mae"]

    with open(r"models/4_furnished_mae.json", "r") as f:
        furnished_mae = json.load(f)["mae"]

    return unfurnished_model, furnished_model, unfurnished_mae, furnished_mae


@st.cache_data
def load_data():
    return pd.read_csv(r"cleaned_data/cleaned_pune_property_ds.csv")


unfurnished_model, furnished_model, unfurnished_mae, furnished_mae = load_models()
df = load_data()

# ---------------- INDIAN FORMAT FUNCTION ---------------- #

def format_indian(number):
    s = str(int(number))
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    remaining = s[:-3]
    formatted_remaining = ",".join(
        [remaining[max(i-2, 0):i] for i in range(len(remaining), 0, -2)][::-1]
    )
    return f"{formatted_remaining},{last_three}"


# ---------------- HEADER ---------------- #

st.title("🏠 Pune Property Price Predictor & Analytics Dashboard")

st.markdown("""
🏙️ Pune Property Intelligence is a Machine Learning–powered real estate analytics and price prediction platform.

🔮 It predicts prices for furnished and unfurnished flats while offering smart recommendations based on 📍 locality, 📐 area, and 💰 budget.

📊 The interactive dashboard delivers clear insights into price trends, area patterns, and property availability across Pune — helping users make confident, data-driven decisions.""")

st.divider()

# ---------------- SIDEBAR INPUT ---------------- #

st.sidebar.header("Enter Property Details 🍂🌷")

property_type = st.sidebar.selectbox("Property Type", ["Furnished", "Unfurnished"])

balconies = st.sidebar.number_input("Balconies", 0, 10, 1)
bathrooms = st.sidebar.number_input("Bathrooms", 0, 10, 2)
total_area = st.sidebar.number_input("Total Area (sqft)", 200, 5000, 1200)
total_rooms = st.sidebar.number_input("Total BHK", 1, 10, 3)
additional_rooms = st.sidebar.number_input("Additional Rooms", 0, 5, 0)

house_type = st.sidebar.selectbox("House Type", ["New", "Old"])
car_parking = st.sidebar.selectbox("Car Parking", ["Yes", "No"])
power_backup = st.sidebar.selectbox("Power Backup", ["Yes", "No"])

# Furnished extra inputs
if property_type == "Furnished":
    AC = st.sidebar.selectbox("AC", ["Yes", "No"])
    TV = st.sidebar.selectbox("TV", ["Yes", "No"])
    Refrigerator = st.sidebar.selectbox("Refrigerator", ["Yes", "No"])
    Sofa = st.sidebar.selectbox("Sofa", ["Yes", "No"])
    Wardrobe = st.sidebar.selectbox("Wardrobe", ["Yes", "No"])
    Washing_Machine = st.sidebar.selectbox("Washing Machine", ["Yes", "No"])
    Gas_connection = st.sidebar.selectbox("Gas Connection", ["Yes", "No"])
    BED = st.sidebar.selectbox("BED", ["Yes", "No"])

# ---------------- PREDICTION ---------------- #

if st.sidebar.button("Predict Price"):

    user = {
        "balconies": float(balconies),
        "bathroom": float(bathrooms),
        "house type": house_type,
        "no. of additional rooms": float(additional_rooms),
        "total area": float(total_area),
        "total rooms": float(total_rooms)+ 2.0,
        "Car Parking": car_parking,
        "Power Backup": power_backup
    }

    if property_type == "Furnished":
        user.update({
            "AC": AC,
            "TV": TV,
            "Refrigerator": Refrigerator,
            "Sofa": Sofa,
            "Wardrobe": Wardrobe,
            "Washing Machine": Washing_Machine,
            "Gas connection": Gas_connection,
            "BED": BED
        })
        model = furnished_model
        mae = furnished_mae
    else:
        model = unfurnished_model
        mae = unfurnished_mae

    # Create DataFrame
    user_df = pd.DataFrame([user])

    # Ensure numeric columns are float
    user_df[user_df.select_dtypes(include=["int64", "float64"]).columns] = \
        user_df.select_dtypes(include=["int64", "float64"]).astype(float)

    # Predict in log space
    predicted_log_price = model.predict(user_df)[0]

    # Calculate range in log space
    lower_log = predicted_log_price - mae
    upper_log = predicted_log_price + mae

    # Convert back to actual price
    pred_price = round(np.exp(predicted_log_price))
    lower_price = round(np.exp(lower_log))
    upper_price = round(np.exp(upper_log))

    # ---------------- SHOW KPI METRICS ---------------- #

    col1, col2, col3 = st.columns(3)

    # col1.metric("✨ Estimated Predicted Price", f"₹ {format_indian(pred_price)}")
    # col2.metric("💰 Lower Price Range", f"₹ {format_indian(lower_price)}")
    # col3.metric("💰 Upper price Range", f"₹ {format_indian(upper_price)}")
    
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">✨ Estimated Predicted Price</div>
            <div class="kpi-value">₹ {format_indian(pred_price)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💰 Lower Price Range</div>
            <div class="kpi-value">₹ {format_indian(lower_price)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💰 Upper Price Range</div>
            <div class="kpi-value">₹ {format_indian(upper_price)}</div>
        </div>
        """, unsafe_allow_html=True)


    st.divider()

    # ---------------- RECOMMENDATION ---------------- #

    recommended_flats = df[
        (df["price"] >= lower_price) &
        (df["price"] <= upper_price)
    ]

    area_tolerance = 150
    recommended_flats = recommended_flats[
        abs(recommended_flats["total area"] - total_area) <= area_tolerance
    ]

    if recommended_flats.empty:
        st.warning("No Matching Flats Found in Pune")
    else:
        st.subheader("🏠 Recommended Flats")

        avg_price = int(recommended_flats["price"].mean())
        min_price = int(recommended_flats["price"].min())
        max_price = int(recommended_flats["price"].max())
        total_flats = recommended_flats.shape[0]

        # c1, c2, c3, c4 = st.columns(4)

        # c1.metric("🏠 Total Flats", total_flats)
        # c2.metric("💵 Average Price", f"₹ {format_indian(avg_price)}")
        # c3.metric("💵 Cheapest Flat", f"₹ {format_indian(min_price)}")
        # c4.metric("💵 Most Expensive Flat", f"₹ {format_indian(max_price)}")
        
        col4, col5, col6, col7 = st.columns(4)

        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Flats</div>
                <div class="kpi-value">{total_flats}</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Price</div>
                <div class="kpi-value">₹ {format_indian(avg_price)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Cheapest Flat</div>
                <div class="kpi-value">₹ {format_indian(min_price)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col7:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Most Expensive Flat</div>
                <div class="kpi-value">₹ {format_indian(max_price)}</div>
            </div>
            """, unsafe_allow_html=True)
    
        st.divider()

        # ---------------- DASHBOARD GRAPHS ---------------- #

        # =========================
        # CENTERED COLUMN LAYOUT
        # =========================

        colA, colB = st.columns(2)

        # =========================
        # GRAPH 1 — LINE GRAPH
        # =========================

        with colA:
            
            price_sorted = np.sort(recommended_flats["price"]) 
            area_sorted = np.sort(recommended_flats["total area"])
        
            fig1 = go.Figure()

            fig1.add_trace(
                go.Scatter(
                    x=area_sorted,
                    y=price_sorted,
                    mode="lines",
                    line=dict(color="#018790", width=3),
                    name="Price Trend",
                    line_shape="spline"
                )
            )

            fig1.update_layout(
                title="Price Distribution with Total Area",
                title_x=0.5,
                plot_bgcolor="#9ABDBD",
                paper_bgcolor="#9ABDBD",
                font=dict(color="black", size=15),
                xaxis_title="Total Area",
                yaxis_title="Price",
                margin=dict(l=40, r=40, t=60, b=40)
            )

            fig1.update_xaxes(
                showgrid=True,
                gridcolor="white",
                tickfont=dict(color="black"),
                title_font=dict(color="black")
            )

            fig1.update_yaxes(
                showgrid=True,
                gridcolor="white",
                tickfont=dict(color="black"),
                title_font=dict(color="black")
            )
            st.plotly_chart(fig1, use_container_width=True)


        # =========================
        # GRAPH 2 — TOP 15 BAR GRAPH
        # =========================

        with colB:

            locality_count = (
                recommended_flats
                .groupby("locality")
                .size()
                .sort_values(ascending=False)
                .head(15)   # TOP 15 ONLY
                .reset_index(name="count")
            )

            fig2 = go.Figure()

            fig2.add_trace(
                go.Bar(
                    x=locality_count["count"],
                    y=locality_count["locality"],
                    orientation="h",
                    marker=dict(color="#018790"),
                )
            )

            fig2.update_layout(
                title="Top 15 Localities by Flat Count",
                title_x=0.5,
                plot_bgcolor="#9ABDBD",
                paper_bgcolor="#9ABDBD",
                font=dict(color="black", size=14),
                xaxis_title="Number of Flats",
                yaxis_title="Locality",
                margin=dict(l=40, r=40, t=60, b=40)
            )

            fig2.update_xaxes(
                showgrid=True,
                gridcolor="white",
                tickfont=dict(color="black"),
                title_font=dict(color="black")
            )

            fig2.update_yaxes(
                showgrid=False,
                autorange="reversed",
                tickfont=dict(color="black"),
                title_font=dict(color="black")
            )            
            
            st.plotly_chart(fig2, use_container_width=True)


        # =========================
        # PIE CHART SECTION
        # =========================

        st.divider()
        
        custom_colors = [
            "#014D4E",
            "#016B6B",
            "#018790",
            "#3AAFA9",
            "#5F9598",
            "#2B7A78",
            "#1F5F5B"
        ]


        top_localities = (
            recommended_flats
            .groupby("locality")
            .size()
            .sort_values(ascending=False)
            .head(7)
            .reset_index(name="count")
        )
        
        # TOTAL FLATS 

        total_flats = len(recommended_flats)

        # Center Title (Outside Box)
        st.markdown(
            """
            <h3 style='text-align:center; color:white; margin-bottom:15px;'>
                🏠 Total Flats Available Based On Your Filters
            </h3>
            """,
            unsafe_allow_html=True
        )

        # Centered KPI Card (Only Number Inside)
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            st.markdown(
                f"""
                <div style="
                    background: rgba(0, 84, 97, 0.65);
                    backdrop-filter: blur(12px);
                    padding: 25px 10px;
                    border: 1.5px solid rgba(255, 255, 255, 0.4);
                    border-radius: 30px;
                    text-align: center;
                    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
                ">
                    <h1 style="
                        color: white;
                        font-size: 50px;
                        margin: 0;
                        font-weight: 600;
                    ">
                        {total_flats}
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            ) 
            
        st.divider()
       
        fig3 = go.Figure(
            go.Pie(
                labels=top_localities["locality"],
                values=top_localities["count"],
                hole=0.45,
                marker=dict(colors=custom_colors)            )
        )

        fig3.update_traces(
            textinfo="percent+label",
            textfont_size=13,
            marker=dict(line=dict(color="white", width=2))

        )

        fig3.update_layout(
            title="Top 7 Locality Distribution",
            title_x=0.5,
            plot_bgcolor="#9ABDBD",
            paper_bgcolor="#9ABDBD",
            font=dict(color="black", size=14),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig3, use_container_width=True)
        
        
        # =========================
        # TABLE SECTION
        # =========================

        st.divider()

        st.subheader("Top 10 Matching Flats")

        st.dataframe(
            recommended_flats[["locality", "total area", "price"]]
                .sort_values("price")
                .head(10),
            use_container_width=True
        )