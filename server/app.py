from flask import Flask, app, request, jsonify
import time
from core import get_prediction

app = Flask(__name__)
@app.route('/predict/', methods=['GET','POST'])
def send_image():
    path='storage/cache/LSD1.jpg'
    prediction, confidence = get_prediction(path)
    payload = {
        'Status': 'Success',
        'Message': 'Image sent',
        'Prediction': prediction,
        'Confidence': confidence,
    }
    return jsonify(payload)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)