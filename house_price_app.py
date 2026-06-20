import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import joblib
import gzip
import os
import numpy as np

st.set_page_config(page_title=" House Price Analytics", layout="wide")

st.sidebar.title(" House Price Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv('clean_price_prediction_df.csv')  
    return df

df = load_data()

df['price_per_sqft'] = df['listedprice'] / df['area']
df['lot_to_area_ratio'] = df['lotarea'] / df['area']
df['bed_bath_ratio'] = df['bedroom'] / df['bathroom'].replace(0, np.nan)

page = st.sidebar.radio(
    'Pages',
    ['Main Dashboard', 'Price Factors Analysis', "Prediction"]
)

st.sidebar.title('Filters')

state_filter = st.sidebar.multiselect(
    'State', options=df['state'].unique(), default=df['state'].unique()
)

city_filter = st.sidebar.multiselect(
    'City', options=df['city'].unique(), default=df['city'].unique()
)

filtered_df = df[
    (df['state'].isin(state_filter)) &
    (df['city'].isin(city_filter))
]

if page == "Main Dashboard":
    st.title(" House Price Main Dashboard")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric(" Avg Listed Price", f"${filtered_df['listedprice'].mean():,.0f}")
    col2.metric(" Avg Area (sqft)", f"{filtered_df['area'].mean():,.0f}")
    col3.metric(" Avg Price/sqft", f"${filtered_df['price_per_sqft'].mean():,.0f}")

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(10))

elif page == "Price Factors Analysis":
    st.title(" Price Factors Analysis")
    tab1, tab2, tab3 = st.tabs(["Size & Rooms", "Location & Market", "Advanced Insights"])

    with tab1:
        st.subheader("Impact of Size & Rooms")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(filtered_df, x='area', y='listedprice', color='bedroom',opacity=0.6,
                           title='Area vs Price (colored by Bedrooms)',
                           trendline='ols',color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            bed_avg = filtered_df.groupby('bedroom')['listedprice'].mean().reset_index()
            fig = px.bar(bed_avg, x='bedroom', y='listedprice',title='Avg Price by Number of Bedrooms',
                color='listedprice',color_continuous_scale='Blues'
            )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


        bath_avg = filtered_df.groupby('bathroom')['listedprice'].mean().reset_index()
        fig = px.bar(bath_avg, x='bathroom', y='listedprice', 
                    title='Avg Price by Bathrooms', color='listedprice',color_continuous_scale='YlGnBu')
        fig.update_traces(width=0.8,textposition='outside')
        fig.update_layout(coloraxis_showscale=False,bargap=0.05,template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Location & Market Influence")
        col1, col2 = st.columns(2)

        with col1:
            state_price = filtered_df.groupby('state')['listedprice'].mean().sort_values(ascending=False).head(15)
            fig = px.bar(x=state_price.index, y=state_price.values, 
                        title='Avg Price by State (Top 15)', color_discrete_sequence=px.colors.sequential.Plasma,
                labels={'x': 'State', 'y': 'Avg Listed Price'},)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
            filtered_df.sample(min(1000, len(filtered_df))),
            x='longitude',y='latitude',color='listedprice',
            size='area',color_continuous_scale=px.colors.sequential.Mint,title='Geographic Distribution of Prices',)
            st.plotly_chart(fig, use_container_width=True)

        filtered_df['log_price'] = np.log1p(filtered_df['listedprice'])

        fig = px.histogram(filtered_df,x='log_price',nbins=50,marginal='box',
            title='Log Price Distribution',
            color_discrete_sequence=['#C62828']
        )

        fig.update_layout(title='House Price Distribution',bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Correlations & Other Factors")
        num_cols = ['bedroom', 'bathroom', 'area', 'lotarea', 'rentestimate', 'price_per_sqft', 'listedprice']
        import plotly.express as px

        corr_price = (
            filtered_df.select_dtypes(include='number')
            .corr()[['listedprice']]
            .sort_values('listedprice', ascending=False)
        )

        fig = px.imshow(corr_price,text_auto='.2f',color_continuous_scale='Reds',
            aspect='auto'
        )

        fig.update_layout(
            title='Correlation with House Price',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(filtered_df, x='rentestimate', y='listedprice', 
                        title='Rent Estimate vs Listed Price', trendline='ols')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction":
    st.title('House Price Prediction')

    df = pd.read_csv('clean_price_prediction_df.csv')
    #model = joblib.load('house_price_model.pkl')


    @st.cache_resource
    def load_model():
        model_path = "house_price_model.pkl.gz"

        if not os.path.exists(model_path):
            st.error(" Model file not found!")
            st.stop()

        try:
            with gzip.open(model_path, 'rb') as f:
                model = joblib.load(f)
            return model
        except Exception as e:
            st.error(f" Error loading model: {str(e)}")
            st.stop()


    model = load_model()

    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        state    = st.selectbox('State', sorted(df['state'].dropna().unique()))
        cities   = sorted(df.loc[df['state']==state,'city'].dropna().unique())
        city     = st.selectbox('City', cities)
        bedroom  = st.number_input('Bedrooms', 1, 10, 3)
        bathroom = st.number_input('Bathrooms', 1.0, 10.0, 2.0, step=0.5)
        area     = st.number_input('Area (sq ft)', 200.0, 20000.0, 1800.0)
    with col2:
        lotarea       = st.number_input('Lot Area (acres)', 0.0, 50.0, 0.3)
        rentestimate  = st.number_input('Rent Estimate', 0.0, 20000.0, 1800.0)
        latitude      = st.number_input('Latitude',  -90.0, 90.0,  float(df['latitude'].median()))
        longitude     = st.number_input('Longitude', -180.0, 180.0, float(df['longitude'].median()))

    if st.button('Predict price'):
        lot_ratio = lotarea / area if area else None
        new_row = pd.DataFrame([{
            'state': state, 'city': city,
            'bedroom': bedroom, 'bathroom': bathroom,
            'area': area, 'lotarea': lotarea,
            'rentestimate': rentestimate,
            'latitude': latitude, 'longitude': longitude,
            'lot_to_area_ratio': lot_ratio,
        }])
        pred = model.predict(new_row)[0]
        st.success(f'Predicted listed price: ${pred:,.0f}')
