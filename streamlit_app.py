import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

@st.cache_resource
def load_model_and_tokenizer():
    model_name = 'facebook/m2m100_418M'
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

available_languages = {
    "Afrikaans": "af", "Amharic": "am", "Arabic": "ar", "Asturian": "ast",
    "Azerbaijani": "az", "Bashkir": "ba", "Belarusian": "be", "Bulgarian": "bg",
    "Bengali": "bn", "Breton": "br", "Bosnian": "bs", "Catalan": "ca",
    "Cebuano": "ceb", "Czech": "cs", "Welsh": "cy", "Danish": "da",
    "German": "de", "Greek": "el", "English": "en", "Esperanto": "eo",
    "Spanish": "es", "Estonian": "et", "Basque": "eu", "Persian": "fa",
    "Finnish": "fi", "French": "fr", "Western Frisian": "fy", "Irish": "ga",
    "Scottish Gaelic": "gd", "Galician": "gl", "Gujarati": "gu", "Hausa": "ha",
    "Hebrew": "he", "Hindi": "hi", "Croatian": "hr", "Haitian Creole": "ht",
    "Hungarian": "hu", "Armenian": "hy", "Indonesian": "id", "Igbo": "ig",
    "Iloko": "ilo", "Icelandic": "is", "Italian": "it", "Japanese": "ja",
    "Javanese": "jv", "Georgian": "ka", "Kazakh": "kk", "Central Khmer": "km",
    "Kannada": "kn", "Korean": "ko", "Luxembourgish": "lb", "Ganda": "lg",
    "Lingala": "ln", "Lao": "lo", "Lithuanian": "lt", "Latvian": "lv",
    "Malagasy": "mg", "Maori": "mi", "Macedonian": "mk", "Malayalam": "ml",
    "Mongolian": "mn", "Marathi": "mr", "Malay": "ms", "Maltese": "mt",
    "Burmese": "my", "Nepali": "ne", "Dutch": "nl", "Northern Sotho": "nso",
    "Norwegian": "no", "Occitan": "oc", "Oriya": "or", "Punjabi": "pa",
    "Polish": "pl", "Pashto": "ps", "Portuguese": "pt", "Quechua": "qu",
    "Romanian": "ro", "Russian": "ru", "Sindhi": "sd", "Sinhala": "si",
    "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Albanian": "sq",
    "Serbian": "sr", "Sundanese": "su", "Swedish": "sv", "Swahili": "sw",
    "Tamil": "ta", "Telugu": "te", "Tajik": "tg", "Thai": "th",
    "Tagalog": "tl", "Tswana": "tn", "Turkish": "tr", "Ukrainian": "uk",
    "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi", "Xhosa": "xh",
    "Yoruba": "yo", "Yiddish": "yi", "Chinese": "zh", "Zulu": "zu"
}

def translate(texts, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded_texts = tokenizer(texts, return_tensors="pt", padding=True)
    generated_tokens = model.generate(**encoded_texts, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_texts

st.title("M2M100 Language Translator")
st.write("Translate text from one language to another using the M2M100 model.")

src_lang = st.selectbox("Select source language", list(available_languages.keys()))
tgt_lang = st.selectbox("Select target language", list(available_languages.keys()))

text = st.text_area("Enter text to translate")

if st.button("Translate"):
    if text:
        translated_texts = translate([text], available_languages[src_lang], available_languages[tgt_lang])
        st.write("Translated Text:")
        for translated_text in translated_texts:
            st.write(translated_text)
    else:
        st.write("Please enter text to translate.")
