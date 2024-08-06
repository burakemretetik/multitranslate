!pip install transformers

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import logging
logging.set_verbosity_error()  # Transformatörlerden gelen uyarıları ve hataların baskılanması.

# Modelin ve tokenizer'ın yüklenmesi
def load_model_and_tokenizer():
    model_name = 'facebook/m2m100_418M'
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# M2M100 modelinin desteklediği diller
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

def display_languages():
    print("Available languages:")
    for lang, code in available_languages.items():
        print(f"{lang}: {code}")

# Çeviri fonksiyonu
def translate(texts, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded_texts = tokenizer(texts, return_tensors="pt", padding=True)
    generated_tokens = model.generate(**encoded_texts, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_texts

# Dil kodunun sorulması
def get_language_code(prompt):
    while True:
        code = input(prompt).strip()
        if code in available_languages.values():
            return code
        else:
            print("Invalid language code. Please try again.")

# Main fonksiyonu
def main():
    display_languages()

    src_lang = get_language_code("Enter the source language code: ")
    tgt_lang = get_language_code("Enter the target language code: ")

    print("The model is ready. Enter text to translate (type 'stop' to finish):")

    while True:
        text = input("> ").strip()

        if not text or text.lower() == 'stop':
            print("Session ended.")
            break

        translated_texts = translate([text], src_lang, tgt_lang)
        for translated_text in translated_texts:
            print(f"Translated: {translated_text}")

if __name__ == "__main__": #Çeviri için çalıştırın.
    main()
