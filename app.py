from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
print("starting")
mobile = tf.keras.applications.MobileNetV2()
x = mobile.layers[-2].output
x = tf.keras.layers.Dropout(0.25)(x)
predictions = tf.keras.layers.Dense(23, activation='softmax')(x)
model = tf.keras.models.Model(inputs=mobile.input, outputs=predictions)
model.compile(tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])
print("model loaded and compiled")
model.load_weights('model_weights.h5')

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image = request.files['image']
    image_path = 'path_to_save_image.jpg'
    image.save(image_path)

    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Make the prediction
    predictions = model.predict(image_array)
    predicted_label = np.argmax(predictions, axis=1)[0].item()
    confidence = predictions[0][predicted_label] * 100

    return jsonify({'predicted_label': predicted_label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)