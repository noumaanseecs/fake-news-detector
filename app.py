from flask import Flask, render_template, request, jsonify
from ml_handler import classifier

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        title = request.form.get('title', '')
        text = request.form.get('text', '')

        full_text = f"{title}. {text}"
        result = classifier.predict(full_text)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'prediction': 'Real News', 'confidence': 50})


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'app': 'Fake News Detector'})

# NO if __name__ == '__main__' section needed for PythonAnywhere