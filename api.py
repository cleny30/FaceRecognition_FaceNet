from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import zipfile
import tempfile
from data_preprocess import run_preprocessing, clear_directory  # Import the functions

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

# Define the directories for extraction and preprocessing
EXTRACTION_PATH = 'train_img'
OUTPUT_PATH = 'aligned_img'

@app.route('/upload', methods=['POST'])
def upload_zip():
  if 'file' not in request.files:
      return jsonify({'error': 'No file part'}), 400

  file = request.files['file']

  if file.filename == '':
      return jsonify({'error': 'No selected file'}), 400

  if file and file.filename.endswith('.zip'):
      # Create a temporary directory
      with tempfile.TemporaryDirectory() as temp_dir:
          temp_zip_path = os.path.join(temp_dir, file.filename)
          file.save(temp_zip_path)

          # Extract the ZIP file
          with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
              zip_ref.extractall(EXTRACTION_PATH)

      # Run the data preprocessing
      nrof_images_total, nrof_successfully_aligned = run_preprocessing(EXTRACTION_PATH, OUTPUT_PATH)

      # Clear the train_img directory
      clear_directory(EXTRACTION_PATH)

      # Return the results of the preprocessing
      return jsonify({
          'message': 'File successfully uploaded, extracted, and processed',
          'total_images': nrof_images_total,
          'successfully_aligned_images': nrof_successfully_aligned
      }), 200

  return jsonify({'error': 'Invalid file type, only .zip allowed'}), 400

if __name__ == '__main__':
  # Ensure the extraction and output paths exist
  if not os.path.exists(EXTRACTION_PATH):
      os.makedirs(EXTRACTION_PATH)
  if not os.path.exists(OUTPUT_PATH):
      os.makedirs(OUTPUT_PATH)

  app.run(debug=True)