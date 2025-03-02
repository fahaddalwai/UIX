import pandas as pd
import nltk

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')  # This is needed for sentence tokenization

# Load dataset
df = pd.read_csv("student_essays.csv")  # Ensure it has 'text' and 'condition' columns

# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(text):
    words = nltk.word_tokenize(str(text).lower())  # Tokenize and convert to lowercase
    total_words = len(words)  # Total word count (tokens)
    unique_words = len(set(words))  # Unique word count (types)
    return unique_words / total_words if total_words > 0 else 0  # Avoid division by zero

# Apply function to calculate TTR for each essay
df["TTR"] = df["text"].apply(calculate_ttr)

# Separate AI-assisted vs. Non-AI essays
df_ai = df[df["condition"] == "ai"]
df_no_ai = df[df["condition"] == "no_ai"]

# Calculate average TTR for both groups
avg_ttr_ai = df_ai["TTR"].mean()
avg_ttr_no_ai = df_no_ai["TTR"].mean()

# Print results
print(f"Average TTR (With Grammarly AI): {avg_ttr_ai:.4f}")
print(f"Average TTR (Without Grammarly AI): {avg_ttr_no_ai:.4f}")

# Save results to a new CSV
df.to_csv("student_essays_with_ttr.csv", index=False)
print("Updated dataset with TTR saved as student_essays_with_ttr.csv")
