from flask import Flask, render_template, url_for, request, jsonify, Markup           
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np

app = Flask(__name__, template_folder='web')

# Load the vecotrize vocabulary specific to the category
# rb: read bytes
# wb: write bytes
with open(r"./models/toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"./models/severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"./models/obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"./models/insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"./models/threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"./models/identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled models
with open(r"./models/toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"./models/severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"./models/obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"./models/insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"./models/threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"./models/identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)

# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('index.html')

# Predicting function
@app.route("/predict", methods=['POST'])
def predict():
    """
    Function that inputs the written text and then is predicted by each of the trained moodel's toxic feature.

    Returns: the rendered object from Flask API that takes in the differente prediction to the html file.
    """
    
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]

    # In hundreds
    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:,1] * 100

    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:,1] * 100

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:,1] * 100

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:,1] * 100

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:,1] * 100

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:,1] * 100

    # Round it
    out_tox = round(pred_tox[0], 2) 
    out_sev = round(pred_sev[0], 2) 
    out_obs = round(pred_obs[0], 2) 
    out_ins = round(pred_ins[0], 2) 
    out_thr = round(pred_thr[0], 2) 
    out_ide = round(pred_ide[0], 2) 

    print('Done') # Helper message



    # bar_labels=['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate']
    # bar_values=[20,40,60,80,100]

    return render_template('index.html', 
                            pred_tox = 'Toxic Level Detected: {} %'.format(out_tox),
                            pred_sev = 'Severe Toxic Level Detected: {} %'.format(out_sev), 
                            pred_obs = 'Obscene Level Detected: {} %'.format(out_obs),
                            pred_ins = 'Insult Level Detected: {} %'.format(out_ins),
                            pred_thr = 'Threat Level Detected: {} %'.format(out_thr),
                            pred_ide = 'Identity Hate Level Detected: {} %'.format(out_ide)                        
                            )
    # data = [out_tox, out_sev, out_obs, out_ins, out_thr, out_ide]
    # return render_template('index.html',
    #                         #  max =100, 
    #                         #  labels = bar_labels,
    #                         #   values =bar_values,
    #                           pred_tox = out_tox,
    #                           pred_sev = out_sev,
    #                           pred_obs = out_obs,
    #                           pred_ins = out_ins,
    #                           pred_thr = out_thr,
    #                           pred_ide = out_ide)
    # return render_template('index.html', data_predict = data)
    # bar_labels=['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate']
    # data = [out_tox, out_sev, out_obs, out_ins, out_thr, out_ide]
    # bar_values=data
    # return render_template('index.html', title='Bitcoin Monthly Price in USD', max=17000, labels=bar_labels, values=bar_values)
    
# Server reloads itself if code changes so no need to keep restarting:
app.run(debug=True)
