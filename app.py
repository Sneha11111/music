from flask import Flask, render_template, request, year ,average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp,Area,Item
import numpy as np
import pandas as pd
import pickle

#loading models
dtr = pickle.load (open('dtr.pkl' , 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
     return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
     if request.method=='POST':
          Year = request.form['Year']
          average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']	
          pesticides_tonnes = request.form['pesticides_tonnes']
          avg_temp = request.form['avg_temp']	
          Area = request.form['Area']	
          Item = request.form['Item']
if __name__=='__main__' :
     app.run(debug=True)

features = np.array([[ year ,average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp,Area,Item]])
    
transformed_features = preprocessor.transform(features)
predicted_value = dtr.predict(transformed_features).reshape(1,-1)

render_template('index.html',predicted_value=predicted_value)
   
