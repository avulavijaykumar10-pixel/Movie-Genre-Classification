# =========================================
# MOVIE GENRE CLASSIFICATION PROJECT
# Naive Bayes + Visualizations + UI Input
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# 1. LOAD DATASET
# -------------------------
df = pd.read_csv("IMDB_Dataset.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Shape:", df.shape)

# -------------------------
# 2. GENRE DISTRIBUTION
# -------------------------
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Genre", order=df["Genre"].value_counts().index)
plt.title("Genre Distribution")
plt.xticks(rotation=45)
plt.show()

# -------------------------
# 3. SPLIT DATA
# -------------------------
X = df["Description"]
y = df["Genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 4. TF-IDF VECTORIZATION
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# 5. TRAIN MODEL
# -------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------------
# 6. PREDICTION
# -------------------------
y_pred = model.predict(X_test_vec)

# -------------------------
# 7. ACCURACY
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# -------------------------
# 8. CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# 9. ACCURACY GRAPH
# -------------------------
plt.figure(figsize=(5,4))
plt.bar(["Naive Bayes"], [accuracy], color="green")
plt.ylim(0,1)
plt.title("Model Accuracy")
plt.show()

# -------------------------
# 10. TOP WORDS GRAPH
# -------------------------
all_words = " ".join(df["Description"]).lower().split()
common_words = Counter(all_words).most_common(15)

words = [w[0] for w in common_words]
counts = [w[1] for w in common_words]

plt.figure(figsize=(10,5))
plt.bar(words, counts, color="orange")
plt.title("Top 15 Most Common Words")
plt.xticks(rotation=45)
plt.show()

# -------------------------
# 11. PREDICTION DISTRIBUTION
# -------------------------
pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

pred_df["Predicted"].value_counts().plot(kind="bar", color="purple")
plt.title("Predicted Genre Distribution")
plt.show()

# -------------------------
# 12. SAMPLE INPUT PREDICTION
# -------------------------
sample = input("\nEnter a movie description: ")
sample_vec = vectorizer.transform([sample])
print("\nPredicted Genre:", model.predict(sample_vec)[0])

# -------------------------
# 13. CONTINUOUS USER INPUT
# -------------------------
while True:
    text = input("\nEnter movie description (or type 'exit'): ")
    
    if text.lower() == "exit":
        print("Program Ended")
        break
    
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    
    print("Predicted Genre:", prediction[0])