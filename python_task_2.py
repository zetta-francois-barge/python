import re
import fitz
import requests
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from translate import Translator


# Load PDF from local storage
def load_pdf_from_local(file_path):
    pdf_document = fitz.open(file_path)
    pdf_documents = pdf_document.load_page(3)
    text = pdf_documents.get_text()
    pdf_document.close()
    return text

# Load PDF from URL
def load_pdf_from_url(url):
    response = requests.get(url)
    pdf_data = response.content
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    pdf_documents = pdf_document.load_page(3)
    text = pdf_documents.get_text()
    pdf_document.close()
    return text

# Load PDF from database drive
def load_pdf_from_db(drive_id):
    path_to_drive = "https://docs.google.com/document/d/" + drive_id
    response = requests.get(path_to_drive)
    pdf_data = response.content
    pdf_document = fitz.open(pdf_data)
    pdf_documents = pdf_document.load_page(0)
    text = pdf_documents.get_text()
    return text


nltk.download('punkt')


# Preprocess the Sentences
def preprocess_text(sentences):
    tokens_list = []
    
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        sentence = sentence.replace(r"\n", " ")
        
        tokens = nltk.word_tokenize(sentence)
        tokens_list.append(tokens)
    return tokens_list

tokens_list = preprocess_text(sentences)
print(tokens_list)

keywords = ["put", "the", "keywords", "here"]

def vectorizer(sentences, keywords):
    
    # Vectorize the Text & keywords
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    keywords_vector = tfidf_vectorizer.transform(keywords)
    
    return keywords_vector, tfidf_matrix


def similarity(tfidf_matrix, keywords_vector, sentences, keywords):
    
    # Matrix of Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, keywords_vector)
    
    # Similarity between keywords and text
    similarity_df = pd.DataFrame(similarity_matrix, index=sentences, columns=keywords)
    
    return similarity_matrix, similarity_df



file_path = "Zetta Accounts for Mr Francois Barge - 06/20/2024.pdf"
# Do all function
def do_all_function(file_path, keywords):
    
    text = load_pdf_from_local(file_path)
    
    # Split the Text
    sentences = nltk.sent_tokenize(text)

    tokens_list = preprocess_text(sentences)
    print(tokens_list)
    keywords_vector, tfidf_matrix = vectorizer(sentences, keywords)
    
    similarity_matrix, similarity_df = similarity(tfidf_matrix, keywords_vector, sentences, keywords)
    print("Similarity Matrix:", similarity_matrix)
    print("Similarity DataFrame:\n", similarity_df)
    
    return tokens_list, similarity_matrix, similarity_df

tokens_list, similarity_matrix, similarity_df = do_all_function(file_path, keywords)

# Language Detection
def detect_language(sentences):
    languages = [detect(sentence) for sentence in sentences]
    return languages

languages = detect_language(sentences)
print(r"\nDetected Languages:", languages)

# Translation
def translate_text(sentences, dest_language='fr'):
    translator = Translator(to_lang=dest_language)
    translations = [translator.translate(sentence) for sentence in sentences]
    return translations

translated_sentences = translate_text(sentences, 'fr')
print(r"\nTranslated Sentences:", translated_sentences)
