import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Function to load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Function to perform translation
def translate(texts, src_lang, tgt_lang):
    model, tokenizer = load_model_and_tokenizer(src_lang, tgt_lang)
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts

# Show title and description
st.title("🌍 Multilingual Translation App")
st.write(
    "This app uses the MarianMT model from Hugging Face to translate text between multiple languages."
)

# Language selection
src_lang = st.selectbox("Select source language", ['en', 'fr', 'de', 'es', 'ru', 'it'])
tgt_lang = st.selectbox("Select target language", ['fr', 'en', 'de', 'es', 'ru', 'it'])

# Text input
text_input = st.text_area("Enter text to translate", "Hello, how are you?")

# Perform translation when the button is clicked
if st.button("Translate"):
    translated_texts = translate([text_input], src_lang, tgt_lang)
    st.write("Translated text:")
    for text in translated_texts:
        st.write(text)
