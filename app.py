from flask import Flask, render_template, request, redirect
import os
import sys
import dill
import librosa
import pandas as pd
import numpy as np
from src.utils import load_object
from collections import Counter


application = Flask(__name__)
app = application

file_path_arr = []

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('home.html')
    
@app.route('/predict', methods = ['GET','POST'])
def predict_file():
    if 'file' not in request.files:
        # return redirect(request.url)
        return render_template('home.html')
    
    file = request.files['file']
    
    if file.filename == '':
        # return redirect(request.url)
        return render_template('home.html')
    

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_path_arr.append(file_path)
    
    audio_file_path = file_path_arr[-1]
    # audio_file_path = r"C:\Users\Bhavya Prakash\Downloads\Folder 1\Folder 1\Funk.mp3"
    y, sr = librosa.load(audio_file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_df = pd.DataFrame(mfccs.T, columns = ['MFCCS2','MFCCS1','MFCCS3','MFCCS4','MFCCS5','MFCCS6','MFCCS7','MFCCS8','MFCCS9','MFCCS10','MFCCS11','MFCCS12','MFCCS13','MFCCS14','MFCCS15','MFCCS16','MFCCS17','MFCCS18','MFCCS19','MFCCS20'])
    
    model = load_object(file_path = 'artifacts/model.pkl')
    preprocessor = load_object(file_path = 'artifacts/preprocessor.pkl')
    scaled_data = preprocessor.transform(mfccs_df)
    prediction = model.predict(scaled_data)
    most_common_genre = Counter(prediction).most_common(1)[0][0]
    if most_common_genre == 1:
        return (render_template('home.html',results = 'Funk', song_name=file.filename))
        
    elif most_common_genre == 2:
        return (render_template('home.html',results = 'Rock', song_name=file.filename))
        
    elif most_common_genre == 3:
        return (render_template('home.html',results = 'Hip_Hop', song_name=file.filename))
        
    elif most_common_genre == 4:
        return (render_template('home.html',results = 'Pop', song_name=file.filename))
        
    elif most_common_genre == 5:
        return (render_template('home.html',results = 'Jazz', song_name=file.filename))
        
    elif most_common_genre == 6:
        return (render_template('home.html',results = 'Romance', song_name=file.filename))

if __name__ == '__main__':
    app.run(debug=True)