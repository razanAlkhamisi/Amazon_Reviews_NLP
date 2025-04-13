
import streamlit as st
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np

from transformers import BertConfig, TFBertForSequenceClassification

# Load configuration
config = BertConfig.from_pretrained("saved_model_directory/config.json")

# Load model with the config
model = TFBertForSequenceClassification.from_pretrained("saved_model_directory", config=config)


# تحميل الموديل والتوكنيزر
tokenizer = BertTokenizer.from_pretrained("saved_model_directory")

# إعداد Streamlit
st.title("تصنيف مشاعر المراجعات")
review = st.text_area("اكتب مراجعتك هنا:")

if st.button("تقييم"):
    inputs = tokenizer(review, return_tensors="tf", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1)
    predicted_label = tf.argmax(probs, axis=1).numpy()[0]

    label_map = {0: "سلبي", 1: "محايد", 2: "إيجابي"}
    st.write(f"🔍 التصنيف المتوقع: **{label_map[predicted_label]}**")
