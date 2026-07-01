import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import (
    BertTokenizer,
    BertConfig,
    TFBertForSequenceClassification
)

# ==============================================================================
# 1. PAGE CONFIGURATION & ENTERPRISE CORE CSS
# ==============================================================================
st.set_page_config(
    page_title="Amazon Review Sentiment AI", 
    page_icon="📦",
    layout="centered"  
)

st.markdown("""
    <style>
    /* Global Canvas Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0F172A !important; /* Premium Dark Slate */
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Suppress Native Utility Overlays */
    header[data-testid="stHeader"], 
    .stAppHeader, 
    div[data-testid="stDecoration"] {
        visibility: hidden;
        display: none !important;
        height: 0px !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }
    
    /* Modern UI Card Components */
    .amazon-card {
        background: #1E293B; /* Deep Slate Base */
        padding: 3px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
        color: #F8FAFC;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .amazon-card:hover {
        border-color: #475569;
    }
    
    /* Typography Scaling Overrides */
    .amazon-card h1, .amazon-card h2, .amazon-card h3, .amazon-card h4 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        margin-top: 0px;
    }
    
    /* Ingestion Mode Radio Controls Custom Styling */
    div[data-testid="stRadio"] > label {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        margin-bottom: 10px;
    }
    
    .stButton {
    display: flex;
    justify-content: center;
            padding: 14px !important;
}        

    /* High-Performance Action Controls */
    .stButton>button {
        background: linear-gradient(135deg, #FF9900 0%, #E67E00 100%);
        color: #0F172A !important;
        border: none;
        border-radius: 15px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.2px;
        height: 44px;
        width: 450px;
        box-shadow: 0 4px 6px -1px rgba(255, 153, 0, 0.2);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #FFAA22 0%, #F58220 100%);
        box-shadow: 0 10px 15px -3px rgba(255, 153, 0, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    /* Text Input Enclosure Configurations */
    textarea {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 14px !important;
    }
    textarea:focus {
        border-color: #FF9900 !important;
        box-shadow: 0 0 0 2px rgba(255, 153, 0, 0.25) !important;
    }
    
    /* Precision Analytics Metrics Layout Display elements */
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-top: 4px;
    }
    
    /* Custom Distribution Bar Engine Styles */
    .progress-track { 
        background-color: #334155; 
        border-radius: 9999px; 
        overflow: hidden; 
        margin-bottom: 16px; 
        height: 10px; 
    }
    .progress-fill-pos { background: linear-gradient(90deg, #10B981, #059669); height: 100%; border-radius: 9999px; }
    .progress-fill-neg { background: linear-gradient(90deg, #EF4444, #DC2626); height: 100%; border-radius: 9999px; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED INFRASTRUCTURE & RESOURCE ENGINES
# ==============================================================================
MODEL_PATH = r"C:\Users\razan\Downloads\amazon_reviews"

@st.cache_resource
def load_sentiment_model(path):
    config = BertConfig.from_pretrained(path)
    model = TFBertForSequenceClassification.from_pretrained(path, config=config)
    tokenizer = BertTokenizer.from_pretrained(path)
    return model, tokenizer

with st.spinner("Initializing Deep Neural Weight Distributions..."):
    model, tokenizer = load_sentiment_model(MODEL_PATH)


# ==============================================================================
# 3. GLOBAL NAVIGATION APP-HEADER BAR
# ==============================================================================
st.markdown("""
    <div style="
        background-color: #0F172A; 
        padding: 20px 40px; 
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-top: -2rem; 
        margin-bottom: 3rem; 
        display: flex; 
        align-items: center; 
        justify-content: space-between; 
        border-bottom: 2px solid #FF9900;
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="color: #FFFFFF; font-size: 24px; font-weight: 700; letter-spacing: -0.8px;">
                amazon<span style="color: #FF9900; font-weight: 400;">review</span><span style="color: #64748B; font-size: 11px; margin-left: 14px; font-weight: 600; letter-spacing: 1.5px;">SENTIMENT AI</span>
            </span>
        </div>
        
    </div>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. CHUNKED SEQUENCE COMPUTATION RUNNER (OOM PREVENTION)
# ==============================================================================
def get_prediction_metrics(text_list, batch_size=32):
    total_samples = len(text_list)
    all_labels, all_confidences, all_neg_probs, all_pos_probs = [], [], [], []
    
    # Clean UI Native System Status Indicator Update
    progress_bar = st.progress(0, text="Evaluating input batches via pipeline layers...")
    
    for i in range(0, total_samples, batch_size):
        batch_texts = text_list[i : i + batch_size]
        percent_complete = min(i / total_samples, 1.0)
        progress_bar.progress(percent_complete, text=f"Processing matrices ({i:,} / {total_samples:,} sequences calculated)")
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="tf",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        
        outputs = model(**inputs)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()
        
        all_labels.extend(np.argmax(probabilities, axis=1))
        all_confidences.extend(np.max(probabilities, axis=1))
        all_neg_probs.extend(probabilities[:, 0])
        all_pos_probs.extend(probabilities[:, 1])
        
    progress_bar.empty()
    return np.array(all_labels), np.array(all_confidences), np.array(all_neg_probs), np.array(all_pos_probs)


# ==============================================================================
# 5. CONTROL SWITCHBOARD WORKSPACE SELECTION
# ==============================================================================
analysis_mode = st.radio(
    "Select an Analysis Tool:", 
    ["Single Review Analysis", "Batch Review Analysis"], 
    horizontal=True
)


# ------------------------------------------------------------------------------
# WORKSPACE SCENARIO 1: SINGLE DATA STRINGS
# ------------------------------------------------------------------------------
if analysis_mode == "Single Review Analysis":
    st.markdown('<div class="amazon-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom: 18px;'>Single Review Analysis</h3>", unsafe_allow_html=True)
    review = st.text_area(
        "Enter a Customer Review", 
        height=130, 
        placeholder="Type or paste customer feedback metrics here..."
    )
    predict_clicked = st.button("Analyze Text")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if not review.strip():
            st.error("⚠️ Command Interrupted: Please supply structural alphanumeric text data.")
        else:
            labels, confidences, neg_probs, pos_probs = get_prediction_metrics([review])
            predicted_label, confidence, prob_neg, prob_pos = labels[0], confidences[0], neg_probs[0], pos_probs[0]
            
            badge_color, badge_text = ("#0FA875", "POSITIVE SERVICE INDEX") if predicted_label == 1 else ("#EF4444", "CRITICAL / ATTENTION REQUIRED")
            
            st.markdown(f"""
                <div class="amazon-card">
                    <h3 style="margin-bottom: 24px; font-size: 16px; letter-spacing: 0.3px; color: #94A3B8 !important;">REAL-TIME METRIC ANALYSIS</h3>
                    <div style="display: flex; justify-content: space-between; gap: 20px;">
                        <div>
                            <p style="margin:0; color:#94A3B8; font-size:12px; font-weight:600; letter-spacing:0.5px;">PREDICTED INTENT LABEL</p>
                            <div class="metric-value" style="color:{badge_color};">{badge_text}</div>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin:0; color:#94A3B8; font-size:12px; font-weight:600; letter-spacing:0.5px;">CLASSIFICATION CONFIDENCE</p>
                            <div class="metric-value" style="color:#F8FAFC;">{confidence:.2%}</div>
                        </div>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #334155; margin: 24px 0;">
                    <p style="color:#F8FAFC; font-weight:600; margin-bottom:18px; font-size:14px;">Probability Distribution Classes</p>
                    <div style="margin-bottom: 18px;">
                        <div style="display:flex; justify-content:space-between; font-size: 13px; margin-bottom: 6px; color:#94A3B8;"><span>Positive Feature Weights</span><strong style="color:#F8FAFC;">{prob_pos:.2%}</strong></div>
                        <div class="progress-track"><div class="progress-fill-pos" style="width: {prob_pos*100}%;"></div></div>
                    </div>
                    <div>
                        <div style="display:flex; justify-content:space-between; font-size: 13px; margin-bottom: 6px; color:#94A3B8;"><span>Negative Feature Weights</span><strong style="color:#F8FAFC;">{prob_neg:.2%}</strong></div>
                        <div class="progress-track"><div class="progress-fill-neg" style="width: {prob_neg*100}%;"></div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# WORKSPACE SCENARIO 2: MASSIVE DATASETS (BULK CSV LEDGERS)
# ------------------------------------------------------------------------------
else:
    st.markdown('<div class="amazon-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom: 18px;'>Distributed Dataset Ingestion</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload review system ledger logs (Supports .CSV configurations only)", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_col = [col for col in df.columns if any(x in col.lower() for x in ['review', 'text', 'body', 'feedback', 'comments'])]
        
        if not text_col:
            st.error("❌ Vector Slicing Aborted: System failed to detect an explicit feedback column string name. Map target data rows to header value label 'Review'.")
        else:
            target_column = text_col[0]
            st.markdown(f"""
                <div style='color: #94A3B8; font-size:13px; margin-bottom: 20px; padding: 12px; background-color: #1E293B; border-radius: 6px; border-left: 3px solid #FF9900;'>
                📋 <b>Dataset Ready: </b> Using the <b>'{target_column}'</b> column for sentiment analysis. <b>{len(df):,}</b> reviews detected and ready for processing.    </div>
            """, unsafe_allow_html=True)
            
            if st.button("Process Vector Stream Classification"):
                with st.spinner("Running sentiment analysis..."):
                    texts = df[target_column].astype(str).tolist()
                    labels, confidences, neg_probs, pos_probs = get_prediction_metrics(texts)
                    
                    df['Predicted_Sentiment'] = ["Positive" if l == 1 else "Negative" for l in labels]
                    df['Confidence'] = confidences
                
                total_reviews = len(df)
                pos_count = np.sum(labels == 1)
                neg_count = total_reviews - pos_count
                pos_ratio = pos_count / total_reviews

                # Premium Analytics Summary Dashboard
                st.markdown("<h3 style='margin: 20px 0 15px 0;'>Customer Sentiment Overview</h3>", unsafe_allow_html=True)
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f'<div class="amazon-card" style="text-align:center;"><p style="color:#94A3B8; margin:0; font-size:11px; font-weight:600; letter-spacing:0.5px;">INGESTED VOLUME</p><div class="metric-value">{total_reviews:,}</div></div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f'<div class="amazon-card" style="text-align:center;"><p style="color:#10B981; margin:0; font-size:11px; font-weight:600; letter-spacing:0.5px;">OPTIMAL SENTIMENT RATIO</p><div class="metric-value" style="color:#10B981;">{pos_ratio:.2%}</div></div>', unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f'<div class="amazon-card" style="text-align:center;"><p style="color:#EF4444; margin:0; font-size:11px; font-weight:600; letter-spacing:0.5px;">RISK CRITICAL RATIO</p><div class="metric-value" style="color:#EF4444;">{(1-pos_ratio):.2%}</div></div>', unsafe_allow_html=True)

                # Distribution Visualization Panel
                st.markdown('<div class="amazon-card">', unsafe_allow_html=True)
                st.markdown("<h3>Sentiment Distribution</h3>", unsafe_allow_html=True)
                chart_data = pd.DataFrame({'Total Rows Analyzed': [pos_count, neg_count]}, index=['Positive', 'Negative'])
                st.bar_chart(chart_data)
                st.markdown('</div>', unsafe_allow_html=True)

                # Automated Strategy & Quality Control Logic Outputs
                st.markdown('<div class="amazon-card">', unsafe_allow_html=True)
                
               
                
                st.markdown("<p style='margin-top:20px; font-weight:600; color: #94A3B8; font-size:17px;'>Reviews Requiring Attention: </p>", unsafe_allow_html=True)
                neg_examples = df[df['Predicted_Sentiment'] == 'Negative'].sort_values(by='Confidence', ascending=False).head(3)
                
                if not neg_examples.empty:
                    for idx, row in neg_examples.iterrows():
                        st.markdown(f"""
                            <div style="background-color:#0F172A; padding:12px; border-radius:6px; margin-bottom:10px; border: 1px solid #334155; font-size: 13px; color: #CBD5E1;">
                                <i>"{str(row[target_column])[:160]}..."</i>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color:#64748B; font-size:13px;'><i>Zero negative threat values returned inside target execution window.</i></p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Unified Global Data Ledger Display
                st.markdown('<div class="amazon-card">', unsafe_allow_html=True)
                st.markdown("<h3>Sentiment Analysis Results</h3>", unsafe_allow_html=True)
                st.dataframe(df[[target_column, 'Predicted_Sentiment', 'Confidence']], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)