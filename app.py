from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load all components
models = {
    'random_forest': joblib.load('models/random_forest_model.pkl'),
    'logistic_reg': joblib.load('models/logistic_regression_model.pkl'),
    'gradient_boost': joblib.load('models/gradient_boosting_model.pkl'),
    'decision_tree': joblib.load('models/decision_tree_model.pkl')
}
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])  # Add GET if you want both
def predict():
    if request.method == 'POST':
      
        # Get user input
        news_text = request.form['news_text']
        
        # Vectorize input
        vec_text = vectorizer.transform([news_text])
        
        # Get predictions from all models
        predictions = {
            'Random Forest': models['random_forest'].predict(vec_text)[0],
            'Logistic Regression': models['logistic_reg'].predict(vec_text)[0],
            'Gradient Boosting': models['gradient_boost'].predict(vec_text)[0],
            'Decision Tree': models['decision_tree'].predict(vec_text)[0]
        }
        
        # Convert to human-readable labels
        results = {name: "Real" if pred == 1 else "Fake" 
                  for name, pred in predictions.items()}
        
        # Calculate consensus
        votes = list(predictions.values())
        final_verdict = "Real" if sum(votes) > len(votes)/2 else "Fake"
        
        return render_template('index.html', 
                            prediction_text=f"Final Verdict: {final_verdict}",
                            results=results,
                            user_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)