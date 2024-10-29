from flask import Flask, request, render_template
import joblib
import spacy

# Load the trained model and vectorizer
lr_model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load SpaCy model (though we're not using it for preprocessing in this case)
nlp = spacy.load('en_core_web_sm')

# Initialize Flask app
app = Flask(__name__)

def preprocess_text(text):
    # Return the text as-is, without additional preprocessing
    return text

def predict_sentiment(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Vectorize the input text
    vectorized_text = vectorizer.transform([preprocessed_text])
    
    # Predict sentiment
    prediction = lr_model.predict(vectorized_text)
    return int(prediction[0])

# Define the main route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text input from the form
        text = request.form['text']
        
        # Check if the text input is empty
        if not text.strip():
            return render_template('index.html', prediction_text="Please enter text to analyze sentiment.")
        
        # Get the prediction
        sentiment = predict_sentiment(text)
        
        # Return the result
        if sentiment == 1:
            return render_template('index.html', prediction_text='The sentiment is positive.')
        else:
            return render_template('index.html', prediction_text='The sentiment is negative.')

if __name__ == "__main__":
    app.run(debug=True)
