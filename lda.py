import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import warnings

warnings.filterwarnings("ignore")

# Download NLTK stopwords
nltk.download("stopwords")
nltk.download("punkt")

# Load English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    doc = nlp(text)  # Process with spaCy NLP pipeline
    tokens = [token.lemma_ for token in doc if token.text not in stopwords.words("english") and token.is_alpha]
    return tokens

# Function to train LDA and compute coherence score
def train_lda_and_compute_coherence(texts, num_topics=5):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    
    # Compute coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence="c_v")
    coherence_score = coherence_model_lda.get_coherence()
    
    return lda_model, coherence_score, dictionary, corpus

# Ensure script runs correctly on Windows
if __name__ == "__main__":
    # Load dataset (Ensure it has a 'text' column containing student essays)
    df = pd.read_csv("student_essays.csv")  # Change to actual filename

    # Separate AI-assisted vs. Non-AI essays
    df_no_ai = df[df["condition"] == "no_ai"]  # Essays written without Grammarly AI
    df_ai = df[df["condition"] == "ai"]  # Essays written with Grammarly AI

    # Apply preprocessing
    df_no_ai["processed_text"] = df_no_ai["text"].apply(preprocess_text)
    df_ai["processed_text"] = df_ai["text"].apply(preprocess_text)

    # Train LDA and get coherence scores
    lda_no_ai, coherence_no_ai, dict_no_ai, corpus_no_ai = train_lda_and_compute_coherence(df_no_ai["processed_text"].tolist())
    lda_ai, coherence_ai, dict_ai, corpus_ai = train_lda_and_compute_coherence(df_ai["processed_text"].tolist())

    # Print coherence scores
    print(f"Coherence Score (Without Grammarly AI): {coherence_no_ai:.4f}")
    print(f"Coherence Score (With Grammarly AI): {coherence_ai:.4f}")

    # LDA Topic Visualization (Optional)
    lda_display_ai = gensimvis.prepare(lda_ai, corpus_ai, dict_ai)
    lda_display_no_ai = gensimvis.prepare(lda_no_ai, corpus_no_ai, dict_no_ai)

    print("Visualizing LDA Topics for AI-assisted Writing")
    pyLDAvis.display(lda_display_ai)

    print("Visualizing LDA Topics for Non-AI Writing")
    pyLDAvis.display(lda_display_no_ai)
