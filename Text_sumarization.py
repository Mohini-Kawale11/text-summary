import os
import re
import math
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2

# Download necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# ---------------- PDF Text Extraction ----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = min(len(reader.pages), 100)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    return text

# ---------------- Sentence Splitting ----------------
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

# ---------------- Word Frequency ----------------
def word_frequency(sentences):
    freq = {}
    all_words = []
    for sent in sentences:
        for w in nltk.word_tokenize(sent.lower()):
            if w.isalpha() and w not in stop_words:
                all_words.append(ps.stem(w))
    for w in all_words:
        freq[w] = freq.get(w,0)+1
    # normalize
    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] = freq[w]/max_freq
    return freq

# ---------------- Sentence Scoring ----------------
def score_sentences(sentences, freq):
    scores = {}
    for sent in sentences:
        score = 0
        words = [ps.stem(w.lower()) for w in nltk.word_tokenize(sent) if w.isalpha()]
        for w in words:
            score += freq.get(w, 0)
        scores[sent] = score
    return scores

# ---------------- Summarization ----------------
def summarize(text, word_limit=100):
    sentences = split_into_sentences(text)
    freq = word_frequency(sentences)
    scores = score_sentences(sentences, freq)
    sorted_sents = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    
    summary = ""
    total_words = 0
    for sent, score in sorted_sents:
        sent_words = len(sent.split())
        if total_words + sent_words <= word_limit:
            summary += " " + sent
            total_words += sent_words
        if total_words >= word_limit:
            break
    return summary.strip()

# ---------------- Question Answering ----------------
def answer_question(text, question):
    doc = nlp(text)
    sentences = list(doc.sents)
    q_doc = nlp(question)
    keywords = [token.lemma_ for token in q_doc if token.is_alpha and not token.is_stop]
    
    best_sent = None
    best_count = 0
    for sent in sentences:
        count = sum(1 for kw in keywords if kw in sent.lemma_)
        if count > best_count:
            best_count = count
            best_sent = sent
    return best_sent.text if best_sent else "No relevant answer found."

# ---------------- Main ----------------
def main():
    pdf_path = input("Enter the path to the PDF file: ")
    word_limit = int(input("Enter the desired word limit for the summary: "))
    
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("The PDF appears to be empty or could not extract text.")
        return
    
    summary = summarize(text, word_limit)
    print("\nSummary:\n", summary)
    
    question = input("\nEnter your question based on the PDF content: ")
    answer = answer_question(text, question)
    print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
