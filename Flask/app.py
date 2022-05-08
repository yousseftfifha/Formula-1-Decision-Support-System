import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# model = pickle.load(open('modelz.pkl', 'rb'))
PerfomancePredictormodel = pickle.load(open('performancePredictor.pkl', 'rb'))
PerfomancePredictorInRacemodel = pickle.load(open('performancePredictorInRace.pkl', 'rb'))
ConstructorperformancePredictorInRacemodel = pickle.load(open('ConstructorperformancePredictorInRace.pkl', 'rb'))
Knnmodel = pickle.load(open('Knn.pkl', 'rb'))
appended_data = pickle.load(open('appended_data1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = PerfomancePredictormodel.predict(final_features)*100

    # output = round(prediction[0],3)
    # output = [float(np.round(x)) for x in prediction]


    return render_template('index.html', DriverQualifyingP=' Predicted Performance 🥁🥁 : {} '.format(int(prediction))+'%')
@app.route('/predictConstructor',methods=['POST'])
def predictConstructor():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = ConstructorperformancePredictorInRacemodel.predict(final_features)*100

    # output = round(prediction[0],3)
    # output = [float(np.round(x)) for x in prediction]


    return render_template('index.html', ConstructorRaceP=' Predicted Constructor Performance 🥁🥁 : {} '.format(int(prediction))+'%')

@app.route('/predictRace',methods=['POST'])
def predictRace():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = PerfomancePredictorInRacemodel.predict(final_features)*100

    # output = round(prediction[0],3)
    # output = [float(np.round(x)) for x in prediction]


    return render_template('index.html', DriverRaceP=' Predicted Performance 🥁🥁 : {} '.format(int(prediction))+'%')


@app.route('/classifier',methods=['POST'])
def classifier():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = Knnmodel.predict(final_features)

    # output = round(prediction[0],3)
    # output = [float(np.round(x)) for x in prediction]


    return render_template('index.html', DriverClass=' Predicted Class 🥁🥁 : {} '.format(int(prediction)))

@app.route('/html_table', methods=['POST'])
def html_table():
    myvar = request.form["circuit_id"]
    df=appended_data[appended_data['circuit_id']== myvar]
    df.rename(columns = {'circuit_id':'circuit'}, inplace = True)
    return render_template('index.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)


# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = PerfomancePredictormodel.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)