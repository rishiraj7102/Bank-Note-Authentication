from flask import Flask,request,render_template
import numpy as np
import pickle

filename="bankauthorization.pkl"
model=pickle.load(open(filename,'rb'))

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    variance=request.form['variance']
    skewnessr=request.form['skewness']
    curtosis=request.form['curtosis']
    entropy=request.form['entropy']
    # data=[np.array(entropy)]
    data=[np.array([variance,skewnessr,curtosis,entropy])]
    prediction=model.predict(data)
    if prediction==1:
        output='Note is Valid'
    elif prediction==0:
        output="Note invalid"
    return render_template('index.html',mypred=output)


if __name__=='__main__':
    app.run(debug=True)