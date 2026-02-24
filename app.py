import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        pipe = pickle.load(open('pipe.pkl','rb'))
        df = pickle.load(open('df.pkl','rb'))
        return pipe, df
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'pipe.pkl' and 'df.pkl' are in the same directory.")
        st.stop()

pipe, df = load_models()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3em;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        color: #6C757D;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .section-header {
        color: #495057;
        font-size: 1.5em;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #E9ECEF;
        padding-bottom: 5px;
    }
    .price-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.8em;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💻 Laptop Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get accurate price predictions for your dream laptop configuration</p>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Basic Specifications Section
    st.markdown('<h3 class="section-header">🏷️ Basic Specifications</h3>', unsafe_allow_html=True)
    
    spec_col1, spec_col2 = st.columns(2)
    
    with spec_col1:
        company = st.selectbox('Brand', df['Company'].unique(), help="Choose the laptop brand")
        type_name = st.selectbox('Type', df['TypeName'].unique(), help="Select laptop category")
        ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64], index=3, help="Select RAM capacity")
    
    with spec_col2:
        weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1, help="Enter laptop weight")
        touchscreen = st.selectbox('Touchscreen', ['No','Yes'], help="Does it have touchscreen?")
        ips = st.selectbox('IPS Display', ['No','Yes'], help="IPS display technology")
    
    # Display Section
    st.markdown('<h3 class="section-header">🖥️ Display Specifications</h3>', unsafe_allow_html=True)
    
    display_col1, display_col2 = st.columns(2)
    
    with display_col1:
        screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0, step=0.1, help="Screen diagonal size")
    
    with display_col2:
        resolution = st.selectbox('Screen Resolution', 
                                ['1920x1080','1366x768','1600x900','3840x2160',
                                 '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'],
                                help="Display resolution")
    
    # Hardware Section
    st.markdown('<h3 class="section-header">⚙️ Hardware Specifications</h3>', unsafe_allow_html=True)
    
    hardware_col1, hardware_col2 = st.columns(2)
    
    with hardware_col1:
        cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique(), help="Processor brand")
        hdd = st.selectbox('HDD Storage (GB)', [0,128,256,512,1024,2048], help="Hard disk storage")
        ssd = st.selectbox('SSD Storage (GB)', [0,8,128,256,512,1024], index=3, help="SSD storage capacity")
    
    with hardware_col2:
        gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique(), help="Graphics card brand")
        os = st.selectbox('Operating System', df['os'].unique(), help="Operating system")

with col2:
    st.markdown('<h3 class="section-header">📊 Configuration Summary</h3>', unsafe_allow_html=True)
    
    # Calculate PPI for display
    if resolution:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Display PPI:</strong> {ppi:.1f}<br>
            <strong>Total Storage:</strong> {hdd + ssd} GB<br>
            <strong>Screen Ratio:</strong> {X_res/Y_res:.2f}:1
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction button with better styling
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button('🔍 Predict Price', use_container_width=True, type="primary")
    
    if predict_button:
        try:
            # Convert categorical to numeric as per training
            touchscreen_val = 1 if touchscreen == 'Yes' else 0
            ips_val = 1 if ips == 'Yes' else 0
            
            # Build DataFrame for prediction
            query_df = pd.DataFrame([{
                'Company': company,
                'TypeName': type_name,
                'Ram': ram,
                'Weight': weight,
                'Touchscreen': touchscreen_val,
                'Ips': ips_val,
                'ppi': ppi,
                'Cpu brand': cpu,
                'HDD': hdd,
                'SSD': ssd,
                'Gpu brand': gpu,
                'os': os
            }])
            
            # Make prediction
            pred_price = int(np.exp(pipe.predict(query_df)[0]))
            
            # Display result with better formatting
            st.markdown(f"""
            <div class="price-result">
                💰 Predicted Price<br>
                <strong>₹{pred_price:,}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 📈 Price Insights")
            
            # Price range categorization
            if pred_price < 30000:
                category = "Budget-Friendly 💚"
                advice = "Great for basic tasks and students"
            elif pred_price < 60000:
                category = "Mid-Range 💙"
                advice = "Perfect for professionals and gaming"
            elif pred_price < 100000:
                category = "Premium 💜"
                advice = "High-performance for demanding tasks"
            else:
                category = "Luxury 🔥"
                advice = "Top-tier specs for professionals"
            
            st.success(f"**Category:** {category}")
            st.info(f"**Recommendation:** {advice}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check your m    odel files and input values.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6C757D; margin-top: 30px;'>
    <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
    <p><em>Predictions are estimates based on historical data</em></p>
</div>
""", unsafe_allow_html=True)