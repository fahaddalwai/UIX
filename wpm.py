import pandas as pd

# Load dataset (Ensure it has 'text' and 'time_taken' columns)
df = pd.read_csv("student_essays.csv")  # Change this to your actual filename

# Ensure time is in minutes
df["time_taken"] = df["time_taken"].astype(float) / 60  # Convert from seconds to minutes if needed

# Function to compute words per minute (WPM)
def calculate_wpm(text, time_taken):
    word_count = len(str(text).split())  # Count number of words in the essay
    return word_count / time_taken if time_taken > 0 else 0  # Avoid division by zero

# Apply function to calculate WPM for each essay
df["WPM"] = df.apply(lambda row: calculate_wpm(row["text"], row["time_taken"]), axis=1)

# Separate AI-assisted vs. Non-AI writing
df_ai = df[df["condition"] == "ai"]
df_no_ai = df[df["condition"] == "no_ai"]

# Calculate average WPM for both conditions
avg_wpm_ai = df_ai["WPM"].mean()
avg_wpm_no_ai = df_no_ai["WPM"].mean()

# Print results
print(f"Average WPM (With Grammarly AI): {avg_wpm_ai:.2f}")
print(f"Average WPM (Without Grammarly AI): {avg_wpm_no_ai:.2f}")

# Save results to a new CSV
df.to_csv("student_essays_with_wpm.csv", index=False)
print("Updated dataset with WPM saved as student_essays_with_wpm.csv")
