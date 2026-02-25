import re
import nltk
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem

nltk.download('punkt')
nltk.download('stopwords')

mystem = Mystem()
stop_words = set(nltk.corpus.stopwords.words('russian'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', '', text)
    return text

def tokenize_and_lemmatize(text: str) -> list:
    # Mystem сразу возвращает список лемм
    lemmas = mystem.lemmatize(text)
    # Убираем пустые строки и стоп-слова
    lemmas = [w for w in lemmas if w.strip() and w not in stop_words and len(w) > 2]
    return lemmas

def preprocess(text: str) -> str:
    cleaned = clean_text(text)
    lemmas = tokenize_and_lemmatize(cleaned)
    return ' '.join(lemmas)