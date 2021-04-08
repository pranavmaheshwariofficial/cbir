import os
from facemask import CNN
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import facemask 
"""
import joblib
joblib.load("kFoldModel1.pkl")
"""
app = Flask(__name__,template_folder='./')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        name, ext = os.path.splitext(uploaded_file.filename)
        file_name = name+datetime.now().strftime("%d%m%Y%H%M%S")+ext
        uploaded_file.save(os.path.join('Dataset/test',file_name))
        dataObj = facemask.Data()
        dataObj.buildDataLoader(False)
        trainTest = facemask.TrainTest(dataObj)
        trainTest.predictProbabilities(file_name)
    return render_template('result.html', result=file_name)

if __name__ == '__main__':
    CNN.__module__ = 'upload'
    app.run()