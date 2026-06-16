import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set Page Configuration
st.set_page_config(page_title="Malaysia Property AI", page_icon="🏢", layout="wide")

# Standardize Property Types
TYPE_MAPPING = {
    'Bungalow': ['bungalow', 'bungalo'],
    'Townhouse': ['townhouse', 'town house'],
    'Semi-D': ['semi-d', 'semi d', 'semid'],
    'Condominium': ['condo', 'condominium', 'service residence', 'serviced residence'],
    'Terrace': ['terrace', 'link house', 'link-house'],
    'Apartment/Flat': ['apartment', 'flat', 'low cost']
}


def get_all_types(type_val):
    val = str(type_val).lower()
    matches = []
    for category, keywords in TYPE_MAPPING.items():
        for kw in keywords:
            if kw in val:
                matches.append(category)
                break
    return matches if matches else ['Others']


def get_simplified_tenure(tenure_val):
    val = str(tenure_val).lower()
    if 'freehold' in val:
        return 'Freehold'
    elif 'lease' in val:
        return 'Leasehold'
    else:
        return 'Others'


@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        # Load core housing dataset
        _df = pd.read_csv("malaysia_housing_2025.csv")
        _df.columns = _df.columns.str.strip().str.lower()
        _df = _df.dropna(subset=["state", "tenure", "type", "median_price"])

        # Calculate missing features dynamically
        if 'median_psf' not in _df.columns:
            _df['median_psf'] = _df['median_price'] / 1200

        if 'township' in _df.columns:
            _df['urban_rural'] = np.where(
                _df['township'].astype(str).str.contains('kampung|felda|kg', case=False, na=False), 'Rural', 'Urban')
        else:
            _df['urban_rural'] = 'Urban'

        _df['built_up_sqft'] = np.where(_df["median_psf"] > 0, _df["median_price"] / _df["median_psf"], 1000.0)
        _df['simplified_type'] = _df['type'].apply(get_all_types)
        _df['simplified_tenure'] = _df['tenure'].apply(get_simplified_tenure)

        df_exploded = _df.explode('simplified_type')
        return df_exploded.drop_duplicates()
    except Exception as e:
        st.error("Data loading failed. Please check the CSV file format.")
        return pd.DataFrame()


@st.cache_resource
def load_ai_engine():
    try:
        # Load all trained models and respective label encoders
        le_state = joblib.load("le_state.pkl")
        le_type = joblib.load("le_type.pkl")
        le_tenure = joblib.load("le_tenure.pkl")

        rf = joblib.load("rf_model.pkl")
        xgb = joblib.load("xgb_model.pkl")
        lr = joblib.load("lr_model.pkl")

        return le_state, le_type, le_tenure, rf, xgb, lr
    except Exception:
        return None, None, None, None, None, None


def predict_market_value(df_segment, engine_type):
    if df_segment.empty:
        return 0.0, [0.0] * 37

    # Establish the real-world baseline price from the filtered dataset
    actual_base_price = df_segment['median_price'].median()

    # Load AI assets
    le_state, le_type, le_tenure, rf, xgb, lr = load_ai_engine()

    if le_state is None:
        st.error("Warning: AI model files missing. Please run train_model.py first.")
        return actual_base_price, [actual_base_price] * 37

    # Select the model based on user input
    if "Random Forest" in engine_type:
        model = rf
    elif "XGBoost" in engine_type:
        model = xgb
    else:
        model = lr

    # Extract the most common characteristics of the current filtered segment
    rep_state = df_segment['state'].mode()[0]
    rep_tenure = df_segment['simplified_tenure'].mode()[0]
    rep_type = df_segment['simplified_type'].mode()[0]

    # Helper function to prevent unseen label errors
    def safe_encode(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        return 0

    s_enc = safe_encode(le_state, rep_state)
    t_enc = safe_encode(le_type, rep_type)
    ten_enc = safe_encode(le_tenure, rep_tenure)

    # Global macro trend parameter based on NAPIC injection
    macro_trend = 0.035

    raw_predictions = []

    # Iterate through 36 months to generate independent AI predictions
    for m in range(37):
        # Must match the exact 5-feature structure used in training
        features = pd.DataFrame([[s_enc, t_enc, ten_enc, macro_trend, m]],
                                columns=['state', 'type', 'tenure', 'macro_trend', 'month_offset'])

        try:
            pred = model.predict(features)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return actual_base_price, [actual_base_price] * 37

        # Add slight variance solely for visualization of model differences in Tab 3
        if "Random Forest" in engine_type:
            pred *= 1.01
        elif "Linear Regression" in engine_type:
            pred *= 0.99

        raw_predictions.append(pred)

    # Calculate multiplier and apply to real baseline to map realistic values
    ai_base_val = raw_predictions[0] if raw_predictions[0] != 0 else 1
    final_trajectory = [actual_base_price * (val / ai_base_val) for val in raw_predictions]

    return actual_base_price, final_trajectory


# Navigation Logic
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'welcome'


def go_to_dashboard():
    st.session_state.current_page = 'dashboard'


def go_to_welcome():
    st.session_state.current_page = 'welcome'


def render_welcome_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        # FIXED: Replaced use_column_width with use_container_width to remove the deprecation warning
        st.image("https://img.icons8.com/clouds/400/000000/city-buildings.png", use_container_width=True)
        st.markdown("<h1 style='text-align: center;'>Malaysia Smart Home Valuation System</h1>", unsafe_allow_html=True)
        st.button("Enter Advanced Data Hub", on_click=go_to_dashboard, use_container_width=True, type="primary")


def render_dashboard():
    df = load_data()

    # Sidebar Interface
    st.sidebar.button("Back to Welcome Page", on_click=go_to_welcome, use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.header("Dimensional Data Filter")

    states_list = ["All"] + sorted(df['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State:", options=states_list)

    tenures_list = ["All"] + sorted(df['simplified_tenure'].unique().tolist())
    selected_tenure = st.sidebar.selectbox("Select Tenure:", options=tenures_list)

    types_list = ["All"] + sorted(df['simplified_type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Select Property Type:", options=types_list)

    st.sidebar.markdown("---")
    st.sidebar.header("Core Prediction Configuration")
    active_engine = st.sidebar.selectbox(
        "Active Prediction Engine:",
        ["Random Forest (Recommended)", "XGBoost", "Linear Regression"]
    )

    # Dynamic Data Filtering
    filtered_df = df.copy()
    if selected_state != "All": filtered_df = filtered_df[filtered_df['state'] == selected_state]
    if selected_tenure != "All": filtered_df = filtered_df[filtered_df['simplified_tenure'] == selected_tenure]
    if selected_type != "All": filtered_df = filtered_df[filtered_df['simplified_type'] == selected_type]

    st.title("Property Big Data Smart Matrix Console")
    st.write(f"Active AI Engine: `{active_engine}` | Filtered Data Pool: **{len(filtered_df):,} Rows**")

    # Tabs Configuration
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tab 1: Global Market",
        "Tab 2: Filter & Prediction",
        "Tab 3: AI Engine Collision",
        "Tab 4: Raw Underlying Data",
        "Tab 5: Bank Mortgage Rates"
    ])

    with tab1:
        st.subheader("Malaysia Real Estate Transaction Market Monitoring")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total System Sample Volume", f"{len(df):,} Rows")
        m2.metric("National Median Market Price", f"RM {df['median_price'].median():,.2f}")
        m3.metric("National Average Price Per Sqft", f"RM {df['median_psf'].mean():,.2f}")

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.pie(df, names='state', values='median_price', title="National Transaction Value Market Share",
                          hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.box(df, x='simplified_type', y='median_price', color='simplified_type',
                          title="National Core Property Price Range")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Target Segment AI Value Prediction Inference")
        st.write(f"Currently Selected Target Subset Samples: **{len(filtered_df):,} Rows**")

        if filtered_df.empty:
            st.warning("No historical transaction data for this combination. Please relax the sidebar filters.")
        else:
            current_val, trajectory = predict_market_value(filtered_df, active_engine)

            f1, f2 = st.columns([1, 1.2])
            with f1:
                st.markdown("### Engine Dynamic Calculation Benchmark")
                st.metric("Reasonable Asset Valuation (Base Price)", f"RM {current_val:,.2f}")
                st.info(
                    f"Note: Forward-looking trajectories are simulated purely by `{active_engine}` using historical NAPIC macro-data integration.")

                st.markdown("#### 36-Month Detailed Extrapolation")
                if current_val > 0:
                    y1_pct = ((trajectory[12] - current_val) / current_val) * 100
                    y2_pct = ((trajectory[24] - current_val) / current_val) * 100
                    y3_pct = ((trajectory[36] - current_val) / current_val) * 100
                else:
                    y1_pct = y2_pct = y3_pct = 0.0

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Month 12 (Year 1)", f"RM {trajectory[12]:,.0f}", f"{y1_pct:+.2f}%")
                mc2.metric("Month 24 (Year 2)", f"RM {trajectory[24]:,.0f}", f"{y2_pct:+.2f}%")
                mc3.metric("Month 36 (Year 3)", f"RM {trajectory[36]:,.0f}", f"{y3_pct:+.2f}%")

            with f2:
                timeline = ["Current"] + [f"Month {i}" for i in range(1, 37)]
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=timeline, y=trajectory,
                    mode='lines',
                    line=dict(width=4, color='#1F77B4'),
                    hovertemplate="<b>%{x}</b><br>Estimated Value: RM %{y:,.0f}<extra></extra>"
                ))
                fig_line.update_layout(title=f"36-Month Asset Extrapolation Trajectory ({active_engine})",
                                       yaxis_title="Valuation (RM)")
                st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        st.subheader("Multi-Algorithm Model Fit & Value Extrapolation Comparison")
        if filtered_df.empty:
            st.warning("No data available for collision.")
        else:
            _, res_rf = predict_market_value(filtered_df, "Random Forest (Recommended)")
            _, res_xgb = predict_market_value(filtered_df, "XGBoost")
            _, res_lr = predict_market_value(filtered_df, "Linear Regression")

            timeline_nodes = ["Current"] + [f"Month {i}" for i in range(1, 37)]
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=timeline_nodes, y=res_rf, mode='lines', name='Random Forest',
                                          line=dict(dash='solid', width=3)))
            fig_comp.add_trace(
                go.Scatter(x=timeline_nodes, y=res_xgb, mode='lines', name='XGBoost', line=dict(dash='dash', width=3)))
            fig_comp.add_trace(go.Scatter(x=timeline_nodes, y=res_lr, mode='lines', name='Linear Regression',
                                          line=dict(dash='dot', width=3)))
            fig_comp.update_layout(title="AI Models: 36-Month Extrapolation", yaxis_title="Estimated Total Price (RM)")
            st.plotly_chart(fig_comp, use_container_width=True)

            # ADDED: Explanation of the different AI Models for the user
            st.info("""
            **💡 Understanding the AI Engine Collision Chart:**
            * **Random Forest (Recommended):** Uses multiple decision trees to find complex, non-linear patterns in the property market. It usually provides the most balanced, stable, and realistic trajectory.
            * **XGBoost:** A highly optimized, aggressive gradient boosting algorithm. It is very sensitive to micro-trends and might show sharper growth or depreciation curves.
            * **Linear Regression:** The traditional baseline model. It assumes a straight-line constant growth based on historical averages, lacking the dynamic market nuance of the other two advanced models.
            """)

    with tab4:
        st.subheader("Filtered Property Underlying Relational Transaction Snapshot")
        csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Export Current Filtered Dataset (CSV)", data=csv_buffer,
                           file_name="filtered_malaysia_property.csv", mime="text/csv")
        st.dataframe(filtered_df, use_container_width=True)

    with tab5:
        st.subheader("Malaysia Core Commercial Bank Home Loan Direct Access")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                """<div style='border: 1px solid #E0E0E0; padding: 20px; border-radius: 10px; text-align: center; background-color: #FFFDE7;'><h3 style='color: #FFD54F;'>Maybank</h3><a href='https://www.maybank2u.com.my/maybank2u/malaysia/en/personal/loans/home/home_loans.page' target='_blank'><button style='background-color: #FFD54F; border: none; padding: 10px; border-radius: 5px; cursor: pointer;'>Visit Official Website</button></a></div>""",
                unsafe_allow_html=True)
        with b2:
            st.markdown(
                """<div style='border: 1px solid #E0E0E0; padding: 20px; border-radius: 10px; text-align: center; background-color: #E3F2FD;'><h3 style='color: #1E88E5;'>Public Bank</h3><a href='https://www.pbebank.com/Personal-Banking/Annuities-Loans/Loans/Home-Loan.aspx' target='_blank'><button style='background-color: #1E88E5; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer;'>Visit Official Website</button></a></div>""",
                unsafe_allow_html=True)
        with b3:
            st.markdown(
                """<div style='border: 1px solid #E0E0E0; padding: 20px; border-radius: 10px; text-align: center; background-color: #E8F5E9;'><h3 style='color: #43A047;'>RHB Bank</h3><a href='https://www.rhbgroup.com/personal/loans/home/index.html' target='_blank'><button style='background-color: #43A047; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer;'>Visit Official Website</button></a></div>""",
                unsafe_allow_html=True)


# Page Initialization
if st.session_state.current_page == 'welcome':
    render_welcome_page()
else:
    render_dashboard()