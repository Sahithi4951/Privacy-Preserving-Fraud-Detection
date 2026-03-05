import pandas as pd
import numpy as np
import string
import re
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("C:/Users/bashe/Downloads/archive (4)/spam.csv", encoding='latin1')

# Renaming the columnsṇ
data = data.rename(columns = { "v1" : "label" , "v2" : "message"})

# Removing unnecessary columns
data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Encoding the text into numeric
lab = LabelEncoder()
data["label"] = lab.fit_transform(data["label"])

# cleaning the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  # stemming once
    return ' '.join(words)
cleaned_text = []
for i,row in data.iterrows():
    text = row['message']
    clean = clean_text(text)
    cleaned_text.append(clean)
data['cleaned_text'] = cleaned_text

# vectorizing the texts
vectorizer = TfidfVectorizer(max_features = 3000)
X = data['cleaned_text']
y = data['label']

# training and test the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight={0:1, 1:10}
)

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

threshold = 0.4

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    custom_pred = (probs > threshold).astype(int)

    return {
        "Confusion-Matrix" : confusion_matrix(y_test, custom_pred), 
        "Recall": recall_score(y_test, custom_pred),
        "Precision": precision_score(y_test, custom_pred),
        "F1": f1_score(y_test, custom_pred),
        "ROC-AUC": roc_auc_score(y_test, probs),
        "PR-AUC" : average_precision_score(y_test,probs)
    }

print("Threshold:",threshold)
print("Random Forest:", evaluate_model(rf, X_train_tfidf, y_train, X_test_tfidf, y_test))
print("Logistic Regression:", evaluate_model(lr,X_train_tfidf, y_train, X_test_tfidf, y_test))
print(data.shape)