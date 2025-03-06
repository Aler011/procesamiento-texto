from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
from collections import Counter
import Levenshtein
import nltk
import re
import spacy
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Ejemplo de tokenización
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer

# Descargar stopwords si no están disponibles
nltk.download('stopwords')
nltk.download('punkt')  # Descarga el paquete necesario para tokenización
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas
# Cargar modelo de spaCy
nlp = spacy.load("es_core_news_sm")

# Modelos de n-gramas preentrenados
def generate_ngrams(text, n=3):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return Counter(ngrams)

# Modelos de n-gramas preentrenados (ejemplo simplificado)
language_profiles = {
    'es': generate_ngrams("hola cómo estás amigo este es un ejemplo en español", n=3),
    'en': generate_ngrams("hello how are you friend this is an example in english", n=3),
    'fr': generate_ngrams("bonjour comment ça va ami ceci est un exemple en français", n=3),
}

# Función para calcular la proporción
def proportion_similarity(profile1, profile2):
    intersection = set(profile1.keys()) & set(profile2.keys())
    matches = sum(profile1[ngram] for ngram in intersection)
    total = sum(profile1.values())
    return matches / total if total > 0 else 0.0

# Función para detectar el idioma
def detect_language(text):
    text_profile = generate_ngrams(text, n=3)
    similarities = {lang: proportion_similarity(text_profile, profile) for lang, profile in language_profiles.items()}
    detected_language = max(similarities, key=similarities.get)
    return detected_language, similarities

def load_dictionary(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().splitlines()

def correct_word(word, dictionary):
    min_distance = float('inf')
    corrected_word = word
    for dict_word in dictionary:
        distance = Levenshtein.distance(word, dict_word)
        if distance < min_distance:
            min_distance = distance
            corrected_word = dict_word
    return corrected_word

def correct_text(text, dictionary):
    words = text.split()
    return " ".join([correct_word(word.lower(), dictionary) for word in words])

def spell_correct(language, text):
    dictionary_file = "Spanish-1000-common.txt" if language == "es" else "English-1000-common.txt"
    dictionary = load_dictionary(dictionary_file)
    return correct_text(text, dictionary)

def remove_stopwords(texto, idioma):
    tokens = re.findall(r'\b\w+\b', texto.lower())
    stop_words = set(stopwords.words('spanish' if idioma == 'es' else 'english'))
    return [word for word in tokens if word not in stop_words]

def lematizar_texto(texto):
    doc = nlp(texto)
    return [token.lemma_ for token in doc]

def procesar_tokenizacion(texto):
    return word_tokenize(texto)

@app.route('/procesar-texto', methods=['POST'])
def procesar_texto():
    data = request.get_json()
    if 'texto' not in data:
        return jsonify({'error': 'No se recibió el texto'}), 400
    
    texto_original = data['texto']
    idioma, similitudes = detect_language(texto_original)
    
    try:
        texto_corregido = spell_correct(idioma, texto_original)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    tokens_sin_stopwords = remove_stopwords(texto_corregido, idioma)
    texto_sin_stopwords = " ".join(tokens_sin_stopwords)

    #texto sin corregir
    #copia_texto_original = texto_original
    #tokenizer = RegexpTokenizer(r'\w+')
    #custom_word_tokens = tokenizer.tokenize(copia_texto_original)

    #texto corregido
    #correct_tokenizer = RegexpTokenizer(r'\w+')
    #correct_custom_word_tokens = correct_tokenizer.tokenize(texto_corregido)
    
    lematizado = lematizar_texto(texto_sin_stopwords)
    
    return jsonify({
        'idioma_detectado': idioma,
        'similitud_en': similitudes['en'],
        'similitud_es': similitudes['es'],
        'texto_original': texto_original,
        'texto_corregido': texto_corregido,
        #'texto_con_tokens': custom_word_tokens,
        #'texto_corregido_con_tokens': correct_custom_word_tokens,
        'texto_tokens_sin_stopwords': tokens_sin_stopwords,
        'texto_lematizado': lematizado
    })

@app.route('/detectar-idioma', methods=['POST'])
def detectar_idioma():
    data = request.get_json()
    if 'texto' not in data:
        return jsonify({'error': 'No se recibió el texto'}), 400
    
    texto = data['texto']
    idioma, similitudes = detect_language(texto)
    
    return jsonify({
        'idioma_detectado': idioma,
        'similitud_en': similitudes['en'],
        'similitud_es': similitudes['es'],
        })

@app.route('/corregir-texto', methods=['POST'])
def corregir_texto():
    data = request.get_json()
    if 'texto' not in data:
        return jsonify({'error': 'No se recibió el texto'}), 400
    
    texto = data['texto']
    idioma, similitudes = detect_language(texto)
    try:
        texto_corregido = spell_correct(idioma, texto)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify({'texto_corregido': texto_corregido})

@app.route('/tokenizar-con-stopwords', methods=['POST'])
def tokenizar_con_stopwords():
    data = request.get_json()

    texto = data['texto']
    idioma, similitudes = detect_language(texto)

    if 'texto' not in data:
        return jsonify({'error': 'No se recibió el texto'}), 400
    
    try:
        texto_corregido = spell_correct(idioma, texto)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    #texto sin corregir
    tokenizer = RegexpTokenizer(r'\w+')
    custom_word_tokens = tokenizer.tokenize(texto)

    #texto corregido
    correct_tokenizer = RegexpTokenizer(r'\w+')
    correct_custom_word_tokens = correct_tokenizer.tokenize(texto_corregido)
    
    return jsonify({'tokens': custom_word_tokens, 'tokens_corregidos': correct_custom_word_tokens})

@app.route('/tokenizar-sin-stopwords', methods=['POST'])
def tokenizar_sin_stopwords():
    data = request.get_json()

    texto = data['texto']
    idioma, similitudes = detect_language(texto)

    if 'texto' not in data:
        return jsonify({'error': 'No se recibió el texto'}), 400

    try:
        texto_corregido = spell_correct(idioma, texto)
        texto_sin_stopwords = remove_stopwords(texto_corregido, idioma)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify({'texto_sin_stopwords': texto_sin_stopwords})

if __name__ == '__main__':
    app.run(debug=True)
