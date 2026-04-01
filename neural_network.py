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
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score, confusion_matrix
import torch 
import torch.nn as nn

data = pd.read_csv(r"C:\Users\bashe\Downloads\archive (5)\creditcard.csv", encoding='latin1')

data = data.rename(columns = { "v1" : "label" , "v2" : "message"})

data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

lab = LabelEncoder()
data["label"] = lab.fit_transform(data["label"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words] 
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
X_train_arr = X_train_tfidf.toarray()
X_test_arr = X_test_tfidf.toarray()
X_train_tensor = torch.FloatTensor(X_train_arr)
X_test_tensor = torch.FloatTensor(X_test_arr)
y_train_tensor = torch.FloatTensor(y_train)
N = len(y_train_tensor)
reshaped_y = y_train_tensor.reshape((N,1))
y_test_arr = np.array(y_test)
y_test_tensor = torch.FloatTensor(y_test_arr)

model = nn.Sequential(
    nn.Linear(3000,16),
    nn.ReLU(),
    nn.Linear(16,1)
)

num_positives = sum(y_train == 1)
num_negatives = sum(y_train == 0)

loss_fn = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([num_negatives/num_positives]))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Results:",)

for epoch in range(100):

    test_logits = model(X_train_tensor)
    loss = loss_fn(test_logits , reshaped_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
model.eval()
with torch.no_grad():
    logits_test = model(X_test_tensor)
    probs = torch.sigmoid(logits_test)
    threshold = 0.2
    custom_pred = (probs > threshold).int().numpy()

    print("Threshold:",threshold)
    print(f"Confusion-Matrix : {confusion_matrix(y_test, custom_pred)}\nRecall: {recall_score(y_test_tensor, custom_pred)}\nPrecision: {precision_score(y_test_tensor, custom_pred)}\nF1: {f1_score(y_test_tensor, custom_pred)}\nROC-AUC: {roc_auc_score(y_test_tensor, probs)}\nPR-AUC: {average_precision_score(y_test_tensor,probs)}")
