import pickle
import os
import json
from args import parser
from tensorflow.keras.models import model_from_json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

VOLUME_PATH = 'C:/Users/harsh/Audio Denoising/demo_data/test'
Predictmodel = pickle.load(open('predictmodel.pkl','rb'))
app = Flask(__name__)
# load json and create model
json_file = open('C:/Users/harsh/Audio Denoising/weights'+'/'+'model_best.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('C:/Users/harsh/Audio Denoising/weights'+'/'+'model_best'+'.h5')


@app.route('/')

def home():
    return render_template('home.html')

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['fileToUpload']
        file_name = 'noisy_voice_long_t2.wav'

        # generate absolute file path
        file_path = os.path.join(VOLUME_PATH,file_name)

        # write file to storage
        f.save(file_path)
        
        args = parser.parse_args()

        weights_path = args.weights_folder
        name_model = args.name_model
        audio_dir_prediction = args.audio_dir_prediction
        dir_save_prediction = args.dir_save_prediction
        audio_input_prediction = args.audio_input_prediction
        audio_output_prediction = args.audio_output_prediction
        sample_rate = args.sample_rate 
        min_duration = args.min_duration
        frame_length = args.frame_length
        hop_length_frame = args.hop_length_frame      
        n_fft = args.n_fft
        hop_length_fft = args.hop_length_fft
        Predictmodel(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
        audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft)

        return render_template("home.html", name = f.filename)
     


if __name__=="__main__":
    app.run(debug=True)