import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("C:\\Users\\vaish\\Downloads\\training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Map target labels to binary values
df['target'] = df['target'].map({0: 0, 4: 1})

# Split data into features and target
X = df['text']
y = df['target']

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Validate the model
y_val_pred = lr_model.predict(X_val_tfidf)
print(classification_report(y_val, y_val_pred))

# Test the model
y_test_pred = lr_model.predict(X_test_tfidf)
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print("Accuracy Score:", accuracy_score(y_test, y_test_pred))

# Save the model and vectorizer
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
