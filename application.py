import numpy as np
import os
import joblib
from flask import session
from flask import Flask, render_template, request, flash, redirect
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array 



app = Flask(__name__)

def predict(values, dic):
    return 0

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/index.html")
def home1():
    return render_template('index.html')

@app.route("/goal.html")
def adminh():
    return render_template("goal.html") 

@app.route("/goal", methods=['GET', 'POST'])
def goalPage():
    time = int(request.form['time'])
    side = int(request.form['side'])
    bodypart = int(request.form['bodypart'])
    location = int(request.form['location'])
    situation = int(request.form['situation'])
    assist_method = int(request.form['assist_method'])
    fast_break = int(request.form['fast_break'])

    vector = np.vectorize(float)
    check = np.array([time, side, bodypart, location, situation, assist_method, fast_break]).reshape(1, -1)

    model_path = "./Model/prediction_model.sav"
    check = vector(check)
    clf = joblib.load(model_path)

    B_pred = clf.predict(check)
    if B_pred == 1:
        result = "GOAL!"
        print("GOAL!")
    else:
        result = "NO GOAL"
        print("NO GOAL")
    
    return render_template('goal.html', data=result)

@app.route("/analysis.html")
def anaylis():
    return render_template("analysis.html") 


@app.route("/about.html")
def about():
    return render_template("about.html") 

if __name__ == '__main__':
	app.run(debug = True)
     
