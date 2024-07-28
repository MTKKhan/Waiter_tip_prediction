# import dependencies.
import pickle
from flask import Flask,render_template,request
import numpy as np
app = Flask(__name__)

# loading all the models with different variation from pikle file.
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('rfr_model.pkl', 'rb') as f:
    rfr_model = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('stack_model.pkl', 'rb') as f:
    stack_model = pickle.load(f)

# starting main page.
@app.route('/')
def main():
    return render_template('index.html')

# getting input from user and predicting output using 
# model and variation specified.
@app.route('/get_tip',methods=['POST'])
def predict():
    bill = float(request.form['bill'])
    sex = int(request.form['sex'])
    smoker = int(request.form['smoker'])
    time = int(request.form['time'])
    size = int(request.form['size'])
    day = [False,False,False]
    get_day = int(request.form['day'])

    if (get_day == 1):
        day[0] = True
    elif (get_day == 2):
        day[1] = True
    elif (get_day == 3):
        day[2] = True
    
    values = [bill,sex,smoker,time,size,day[0],day[1],day[2]]

    final = np.array([values])

    # model and variations.
    model = str(request.form['model'])
    variation = int(request.form['variation'])

    # predict_tip = []
    if model == 'linear regression':
        predict_tip = lr_model[variation].predict(final)
    elif model == 'random forest':
        predict_tip = rfr_model[variation].predict(final)
    elif model == 'svm':
        predict_tip = svm_model[variation].predict(final)
    elif model == 'knn':
        predict_tip = knn_model[variation].predict(final)
    elif model == 'stacking model':
        predict_tip = stack_model[variation].predict(final)

    return render_template('get_tip.html',predict=round(predict_tip[0],2))
   
if __name__ == '__main__':
    # running app on port 8000
    app.run(debug=True,port=8000)


