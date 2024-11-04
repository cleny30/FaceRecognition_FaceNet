from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import zipfile
import tempfile
import cv2
import numpy as np
from .facenet import load_model, prewhiten
import pickle
import shutil
import tensorflow.compat.v1 as tf
from .classifier import training  # Import the training class

from .data_preprocess import run_preprocessing, clear_directory  # Import the functions
import logging

app = Flask(__name__)
CORS(app)

# Define the directories for extraction and preprocessing
EXTRACTION_PATH = '.app/train_img'
OUTPUT_PATH = '.app/aligned_img'
CROPPED_FACE = '.app/cropped_faces'
CLASS_PATH = './app/class'

# Configure TensorFlow
tf.disable_v2_behavior()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# Load model and parameters
modeldir = './app/model/20180402-114759.pb'
classifier_filename = './app/class/classifier.pkl'

# Load face recognition model
load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_faces(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(gray, 1.3, 5)
  return [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

def recognize_face(face_image):
  classifier_loaded = os.path.exists(classifier_filename) 
  if classifier_loaded:
     with open(classifier_filename, 'rb') as infile:
      model, class_names = pickle.load(infile, encoding='latin1')
  # Preprocess the image
  image = cv2.resize(face_image, (160, 160))
  image = prewhiten(image)
  image_reshape = image.reshape(-1, 160, 160, 3)

  # Calculate embeddings
  feed_dict = {images_placeholder: image_reshape, phase_train_placeholder: False}
  emb_array = sess.run(embeddings, feed_dict=feed_dict)

  # Predict the name
  predictions = model.predict_proba(emb_array)
  best_class_indices = np.argmax(predictions, axis=1)
  best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

  if best_class_probabilities[0] > 0.9:
      return {
          "detected": class_names[best_class_indices[0]],
          "accuracy": float(best_class_probabilities[0])
      }
  else:
      return {"message": "Recognition confidence is too low"}

@app.route('/upload', methods=['POST'])
def upload_zip():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and file.filename.endswith('.zip'):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip_path = os.path.join(temp_dir, file.filename)
                file.save(temp_zip_path)

                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACTION_PATH)

            # Run the data preprocessing
            nrof_images_total, nrof_successfully_aligned = run_preprocessing(EXTRACTION_PATH, OUTPUT_PATH)
            clear_directory(EXTRACTION_PATH)

            # Run the training process
            print("Training Start")
            obj = training(OUTPUT_PATH, modeldir, classifier_filename)
            get_file = obj.main_train()
            print('Saved classifier model to file "%s"' % get_file)

            return jsonify({
                'message': 'File successfully uploaded, extracted, processed, and trained',
                'total_images': nrof_images_total,
                'successfully_aligned_images': nrof_successfully_aligned,
                'classifier_file': get_file
            }), 200

        return jsonify({'error': 'Invalid file type, only .zip allowed'}), 400

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Register the /detect endpoint only if the classifier is loaded

@app.route('/detect', methods=['POST'])
def upload_image():
    classifier_loaded = os.path.exists(classifier_filename) 
    if classifier_loaded:
      if 'file' not in request.files:
          return jsonify({"error": "No file part"}), 400

      file = request.files['file']
      if file.filename == '':
          return jsonify({"error": "No selected file"}), 400

      file_bytes = np.frombuffer(file.read(), np.uint8)
      image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
      faces = detect_and_crop_faces(image)

      os.makedirs(CROPPED_FACE, exist_ok=True)
      results = []
      for i, face in enumerate(faces):
          face_filename = os.path.join(CROPPED_FACE, f"face_{i}.jpg")
          cv2.imwrite(face_filename, face)

          result = recognize_face(face)
          result["filename"] = face_filename
          results.append(result)

      # Clean up the cropped faces directory
      shutil.rmtree(CROPPED_FACE)

      return jsonify(results), 200
    
    return jsonify("No classifier"), 404

if __name__ == '__main__':
  os.makedirs(EXTRACTION_PATH, exist_ok=True)
  os.makedirs(OUTPUT_PATH, exist_ok=True)
  os.makedirs(CLASS_PATH, exist_ok=True)
  app.run(host='0.0.0.0', port=80)