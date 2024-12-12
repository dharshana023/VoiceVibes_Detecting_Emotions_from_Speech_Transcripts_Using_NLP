from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from emoo import process_audio_for_emotion_detection, convert_mp3_to_wav

# Initialize Flask app
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Function to check if uploaded file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render the main page for file upload
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle audio file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert to wav if necessary
        if filename.endswith('.mp3'):
            file_path = convert_mp3_to_wav(file_path)

        # Process the audio file and get results
        results = process_audio_for_emotion_detection(file_path)

        # Redirect to result page with the analysis results
        return render_template('result.html', results=results)

    return redirect(request.url)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
