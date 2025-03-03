# **UIX 535 - Code Scripts**

## **Directory Structure**
```
./
├── lda.py              # LDA Topic Modeling & Coherence Analysis
├── ldttr.py           # Lexical Diversity (Type-Token Ratio) Calculation
├── student_essays.csv # Dataset of student essays
└── wpm.py             # Words Per Minute (WPM) Calculation
```

## **Usage**
### **1️⃣ Install Dependencies**
```bash
pip install pandas nltk spacy gensim sklearn pyLDAvis
python -m spacy download en_core_web_sm
```

### **2️⃣ Run Scripts**
- **LDA Topic Modeling & Coherence Score:**
  ```bash
  python lda.py
  ```
- **Lexical Diversity (Type-Token Ratio - TTR):**
  ```bash
  python ldttr.py
  ```
- **Words Per Minute (WPM) Calculation:**
  ```bash
  python wpm.py
  ```

### **3️⃣ Dataset Format (`student_essays.csv`)**

| id | text | time_taken | condition |
|----|------|------------|------------|
| 1  | "Writing an essay without AI took more effort." | 900 | no_ai |
| 2  | "Grammarly AI helped refine my writing." | 720 | ai |
