from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import argparse

app = Flask(__name__)

logging.basicConfig(format='%(asctime)s-%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
            
class SimilarityPredictor:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
        logger.info(f"Loaded model from {model_path}")

    def predict(self, text1, text2):
        embs1 = self.model.encode(text1, convert_to_tensor=True)
        embs2 = self.model.encode(text2, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embs1, embs2)
        return float(np.clip(cos_sim.item(),0,1))
    
@app.route('/similarity', methods=['POST'])
def predict_similarity():
    try:
        data = request.get_json()
        if not data or 'text1' not in data or 'text2' not in data:
            return jsonify({"error": "Missing fields in request"}), 400
        
        similarity_score = predictor.predict(data['text1'],data['text2'])
        return jsonify({"similarity score": round(similarity_score,4)})
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity predictor')
    parser.add_argument('--model_path', type=str, default='models', help='Path to the model file')
    parser.add_argument('--port', type=int, default=5000, help='Port to run API')
    args = parser.parse_args()
    
    global predictor
    predictor = SimilarityPredictor(args.model_path)

    app.run(host='0.0.0.0', port=args.port, debug=True)