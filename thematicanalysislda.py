import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample Verbal Speech Data which we'll replace with the user transcribed text from interviews.
text_data = [
    "The new AI model is improving human interactions through voice recognition.",
    "Healthcare innovations with AI are helping doctors diagnose diseases faster.",
    "Education is evolving with AI-powered learning assistants and personalized content.",
    "AI is transforming businesses with automation and data-driven decision-making.",
    "Ethical concerns in AI development are growing, requiring strict regulations."
]

# **Step 1: Preprocess Text Data**
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert to lowercase
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords.words('english')]
    return " ".join(tokens)

# Apply preprocessing
cleaned_texts = [preprocess_text(text) for text in text_data]

# **Step 2: Convert Text to Vector Representation**
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(cleaned_texts)

# **Step 3: Perform LDA Topic Modeling**
num_topics = 3  # Adjust the number of themes want
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(X)

# **Step 4: Display Topics**
def print_topics(model, vectorizer, top_n=10):
    terms = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(model.components_):
        print(f"\nTheme {idx + 1}:")
        print(", ".join([terms[i] for i in topic.argsort()[-top_n:]]))

print_topics(lda_model, vectorizer)

# **Step 5: Visualize Topics**
def plot_top_words(model, feature_names, n_top_words=10):
    fig, axes = plt.subplots(1, num_topics, figsize=(12, 6), sharex=True)
    for topic_idx, topic in enumerate(model.components_):
        top_features = topic.argsort()[-n_top_words:]
        feature_names = [feature_names[i] for i in top_features]
        weights = topic[top_features]
        
        ax = axes[topic_idx]
        ax.barh(feature_names, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

plot_top_words(lda_model, vectorizer.get_feature_names_out())
