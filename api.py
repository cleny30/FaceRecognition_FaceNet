from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import zipfile
import tempfile

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

# Define the directory where the images will be extracted
EXTRACTION_PATH = 'train_img'

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

      return jsonify({'message': 'File successfully uploaded and extracted'}), 200

  return jsonify({'error': 'Invalid file type, only .zip allowed'}), 400

if __name__ == '__main__':
  # Ensure the extraction path exists
  if not os.path.exists(EXTRACTION_PATH):
      os.makedirs(EXTRACTION_PATH)

  app.run(debug=True)