from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS from flask_cors
from model import Model

try:
    path = './files/skipgram_model.pth'
    model_instance = Model(path)
    print("Model has been loaded successfully")
except KeyError:
    print(f'Error:{KeyError}')

app = Flask(__name__)
# Allow requests from 'http://localhost:3000' to the '/make_recommendations' route
CORS(app, origins=["*"])

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/skipgram', methods=['POST'])
def skipGram():
    try:
        data = request.get_json()
        word = data.get('word', '')
        k = data.get('k', 5)

        top_k_words = model_instance.predict_similar_words(word, k)

        # Print or log the content of top_k_words
        print("Top K Words:", top_k_words)

        response = {'Answer': top_k_words}
        return jsonify(response)
    except KeyError as e:
        return jsonify({'error': f"KeyError: {str(e)}"}), 404
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
