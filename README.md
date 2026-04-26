# 🎬 Movie Genre Classification using Naive Bayes

This project is a Machine Learning-based system that predicts the genre of a movie based on its plot description using Natural Language Processing (NLP). It uses the Naive Bayes algorithm along with TF-IDF vectorization to classify text into different movie genres such as Action, Comedy, Romance, Horror, and Thriller.

---

## 📌 Project Objective
The main objective of this project is to automatically classify movie genres from textual descriptions, reducing the need for manual tagging and improving content organization systems.

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- TF-IDF Vectorizer
- Naive Bayes Classifier

---

## 📊 Dataset Description
The dataset consists of movie plot descriptions and their corresponding genres. Each entry includes:
- Movie Description (Text Input)
- Genre Label (Target Output)

---

## 🧠 Workflow
1. Load dataset
2. Preprocess text data (cleaning & transformation)
3. Convert text into numerical features using TF-IDF
4. Split dataset into training and testing sets
5. Train Naive Bayes model
6. Evaluate model performance
7. Visualize results using graphs
8. Predict genre for new input text

---

## 📈 Visualizations Included
- Genre distribution bar chart
- Confusion matrix heatmap
- Word frequency analysis
- Model accuracy representation

---

## 🎯 Results
The model achieves moderate accuracy depending on dataset size. It performs well on simple and structured movie descriptions but may struggle with complex or limited data scenarios.

---

## 🚀 Future Enhancements
- Use larger and balanced datasets
- Implement advanced models like SVM or Logistic Regression
- Apply deep learning models such as LSTM or BERT
- Deploy as a web application for real-time predictions

---

## ▶️ How to Run
1. Install required libraries using pip
2. Load the dataset file (IMDB_Dataset.csv)
3. Run the Python script or Jupyter Notebook
4. Enter custom movie descriptions to get predictions

---

## 📄 License
This project is developed for educational purposes only.

---

## ⭐ Acknowledgement
This project was built as a Machine Learning mini project to demonstrate text classification using NLP techniques.
