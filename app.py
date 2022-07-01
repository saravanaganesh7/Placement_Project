 # Importing essential libraries
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn


# Load the LogisticRegression model
model = pickle.load(open('place_prediction.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        gender = request.form.get('gen')
        ssc_p = float(request.form['ssc_p'])
        ssc_b = request.form.get('cp')
        hsc_p = float(request.form['Hsc_p'])
        hsc_b = request.form.get('hp')
        hsc_s = request.form.get('sp')
        degree_p = float(request.form['d_p'])
        degree_t= request.form.get('pa')
        workex = request.form.get('wx')
        etest_p = float(request.form['tha'])
        specialisation = request.form.get('spec')
        mba_p = float(request.form['m_P'])
        
        
              
        
        data = np.array([[gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p]])
        
        #data.tofile('sample3.csv',sep=',')
        
        my_prediction = model.predict(data)
        
        a = np.array(my_prediction)
        
        a.tofile('sample1.csv',sep=',')
        return render_template('result.html', prediction=my_prediction)
        

if __name__ == '__main__':
     app.run(debug=True)
