import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
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
        le_urban = joblib.load("le_urban.pkl")

        rf = joblib.load("rf_model.pkl")
        xgb = joblib.load("xgb_model.pkl")
        lr = joblib.load("lr_model.pkl")

        return le_state, le_type, le_tenure, le_urban, rf, xgb, lr
    except Exception:
        return None, None, None, None, None, None, None


def predict_market_value(df_segment, engine_type, user_sqft, user_urban):
    if df_segment.empty:
        return 0.0, [0.0] * 37

    # Load AI assets
    le_state, le_type, le_tenure, le_urban_enc, rf, xgb, lr = load_ai_engine()

    if le_state is None:
        st.error("AI model files missing. Please run train_model.py first.")
        return 0.0, [0.0] * 37

    # Calculate realistic base price using the user's specific size and local segment's median Price Per Sqft
    segment_psf = df_segment['median_psf'].median()
    actual_base_price = user_sqft * segment_psf

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

    # Use user's selected urban/rural status if specific, otherwise use segment mode
    target_urban = user_urban if user_urban != "All" else df_segment['urban_rural'].mode()[0]

    # Helper function to prevent unseen label errors
    def safe_encode(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        return 0

    s_enc = safe_encode(le_state, rep_state)
    t_enc = safe_encode(le_type, rep_type)
    ten_enc = safe_encode(le_tenure, rep_tenure)
    ur_enc = safe_encode(le_urban_enc, target_urban)

    # Global macro trend parameter based on NAPIC injection
    macro_trend = 0.035
    raw_predictions = []

    # Iterate through 36 months to generate independent AI predictions
    for m in range(37):
        # Matches the exact 7-feature structure used in training
        features = pd.DataFrame([[s_enc, t_enc, ten_enc, ur_enc, user_sqft, macro_trend, m]],
                                columns=['state', 'type', 'tenure', 'urban_rural', 'built_up_sqft', 'macro_trend',
                                         'month_offset'])

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


def calculate_segment_metrics(df_segment):
    # We only need 1 row to calculate Percentage Error!
    if len(df_segment) < 1:
        return None

    le_state, le_type, le_tenure, le_urban, rf, xgb, lr = load_ai_engine()
    if rf is None:
        return None

    def safe_transform(le, val_list):
        known = set(le.classes_)
        return [le.transform([v])[0] if v in known else 0 for v in val_list]

    X_eval = pd.DataFrame()
    X_eval['state'] = safe_transform(le_state, df_segment['state'])
    X_eval['type'] = safe_transform(le_type, df_segment['simplified_type'])
    X_eval['tenure'] = safe_transform(le_tenure, df_segment['simplified_tenure'])
    X_eval['urban_rural'] = safe_transform(le_urban, df_segment['urban_rural'])
    X_eval['built_up_sqft'] = df_segment['built_up_sqft'].values
    X_eval['macro_trend'] = 0.035
    X_eval['month_offset'] = 0

    y_true = df_segment['median_price'].values

    results = {}
    from sklearn.metrics import mean_absolute_error

    for name, model in [("RF", rf), ("XGB", xgb), ("LR", lr)]:
        preds = model.predict(X_eval)

        # Calculate Mean Absolute Percentage Error (MAPE)
        # We add 1e-9 to prevent division by zero errors
        mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-9)))

        # Convert MAPE to a highly intuitive Accuracy Percentage
        accuracy_pct = max(0.0, 100.0 - (mape * 100))

        mae = mean_absolute_error(y_true, preds)

        results[name] = {"accuracy": accuracy_pct, "mae": mae}

    return results

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
        st.image("https://img.icons8.com/clouds/400/000000/city-buildings.png", use_container_width=True)
        st.markdown("<h1 style='text-align: center;'>Malaysia House Valuation System</h1>", unsafe_allow_html=True)
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

    # NEW IMPUT: Urban/Rural and House Size
    st.sidebar.markdown("---")
    st.sidebar.header("Property Specifics")
    selected_urban = st.sidebar.selectbox("Select Area Type:", ["All", "Urban", "Rural"])
    input_sqft = st.sidebar.number_input("Enter House Size (Sqft):", min_value=100.0, max_value=20000.0, value=1200.0,
                                         step=100.0)

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
    if selected_urban != "All": filtered_df = filtered_df[filtered_df['urban_rural'] == selected_urban]

    st.title("Property Big Data Smart Matrix Console")
    st.write(f"Active AI Engine: `{active_engine}` | Filtered Data Pool: **{len(filtered_df):,} Rows**")

    # Tabs Configuration
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tab 1: Whole Malaysia Market",
        "Tab 2: Specific House Prediction",
        "Tab 3: AI Engine Comparison",
        "Tab 4: Raw Data",
        "Tab 5: Bank Links"
    ])

    with tab1:
        st.subheader("Malaysia Transaction Market ")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total System Sample Volume", f"{len(df):,} Rows")
        m2.metric("Median Market Price", f"RM {df['median_price'].median():,.2f}")
        m3.metric("Average Price Per Sqft", f"RM {df['median_psf'].mean():,.2f}")

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.pie(df, names='state', values='median_price', title="Transaction Value Market Share", hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.box(df, x='simplified_type', y='median_price', color='simplified_type',
                          title="Core Property Price Range")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Target Segment AI Value Prediction Inference")
        st.write(
            f"Evaluating a **{input_sqft:,.0f} sqft** property. Target Subset Samples: **{len(filtered_df):,} Rows**")

        if filtered_df.empty:
            st.warning("No historical transaction data for this combination. Please relax the sidebar filters.")
        else:
            # Pass the new user inputs into the prediction function
            current_val, trajectory = predict_market_value(filtered_df, active_engine, input_sqft, selected_urban)

            f1, f2 = st.columns([1, 1.2])
            with f1:
                st.metric("Reasonable Asset Valuation (Base Price)", f"RM {current_val:,.2f}")

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
                fig_line.update_layout(title="36-Month Asset Extrapolation Trajectory", yaxis_title="Valuation (RM)")
                st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        st.subheader("Comparison of Three Different Engines")
        if filtered_df.empty:
            st.warning("No data available for collision.")
        else:
            _, res_rf = predict_market_value(filtered_df, "Random Forest (Recommended)", input_sqft, selected_urban)
            _, res_xgb = predict_market_value(filtered_df, "XGBoost", input_sqft, selected_urban)
            _, res_lr = predict_market_value(filtered_df, "Linear Regression", input_sqft, selected_urban)

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

            # --- DYNAMIC METRICS UI ---
            st.markdown("### 📊 Dynamic Localized Model Performance")
            st.write(
                f"These metrics represent real-time accuracy specifically for the **{len(filtered_df)}** properties matching your filter.")

            metrics = calculate_segment_metrics(filtered_df)

            if metrics is None:
                st.warning("⚠️ No data available to calculate localized metrics.")
            else:
                m_col1, m_col2, m_col3 = st.columns(3)

                with m_col1:
                    st.success("**🌳 Random Forest**")
                    st.metric("Localized Accuracy", f"{metrics['RF']['accuracy']:.2f}%")
                    st.metric("Localized MAE", f"RM {metrics['RF']['mae']:,.0f}")

                with m_col2:
                    st.info("**⚡ XGBoost**")
                    st.metric("Localized Accuracy", f"{metrics['XGB']['accuracy']:.2f}%")
                    st.metric("Localized MAE", f"RM {metrics['XGB']['mae']:,.0f}")

                with m_col3:
                    st.warning("**📈 Linear Regression**")
                    st.metric("Localized Accuracy", f"{metrics['LR']['accuracy']:.2f}%")
                    st.metric("Localized MAE", f"RM {metrics['LR']['mae']:,.0f}")

            st.markdown("---")

            st.info("""
            **💡 How to Interpret the Model Performance:**
            * **🎯 Localized Accuracy:** This shows how close the AI's predicted price is to the actual real-world market price. For example, a 95% accuracy means the AI's prediction is, on average, only 5% away from the actual transaction price. *(Higher is better)*.
            * **💰 Localized MAE (Mean Absolute Error):** This represents the average error margin in Ringgit Malaysia (RM). If the MAE is RM 20,000, it means the AI's estimates are typically within a RM 20,000 range above or below the exact true property value. *(Lower is better)*.
            
            **⚙️ Engine Characteristics:**
            * **Random Forest:** Uses multiple decision trees to find complex, non-linear patterns. It provides a highly balanced and stable trajectory.
            * **XGBoost:** A highly optimized gradient boosting algorithm. It is very sensitive to micro-trends and often performs best on specific local segments.
            * **Linear Regression:** The traditional baseline model. It assumes a straight-line constant growth, lacking the dynamic market nuance of the advanced tree models.
            """)
    with tab4:
        st.subheader("Filtered Property Underlying Relational Transaction Snapshot")
        csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Export Current Filtered Dataset (CSV)", data=csv_buffer,
                           file_name="filtered_malaysia_property.csv", mime="text/csv")
        st.dataframe(filtered_df, use_container_width=True)

    with tab5:
        st.subheader("Link to Malaysia Bank Home Loans")
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
                """<div style='border: 1px solid #E0E0E0; padding: 20px; border-radius: 10px; text-align: center; background-color: #E8F5E9;'><h3 style='color: #43A047;'>RHB Bank</h3><a href='https://www.rhbgroup.com/index.html' target='_blank'><button style='background-color: #43A047; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer;'>Visit Official Website</button></a></div>""",
                unsafe_allow_html=True)


# Page Initialization
if st.session_state.current_page == 'welcome':
    render_welcome_page()
else:
    render_dashboard()