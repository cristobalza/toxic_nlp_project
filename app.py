from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np

app = Flask(__name__, template_folder='web')

# Load the TF-IDF vocabulary specific to the category
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

# Load the pickled RDF models
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

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]

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

    out_tox = round(pred_tox[0], 2) 
    out_sev = round(pred_sev[0], 2) 
    out_obs = round(pred_obs[0], 2) 
    out_ins = round(pred_ins[0], 2) 
    out_thr = round(pred_thr[0], 2) 
    out_ide = round(pred_ide[0], 2) 

    print(out_tox)

    return render_template('index.html', 
                            pred_tox = 'Toxic Level Detected: {} %'.format(out_tox),
                            pred_sev = 'Severe Toxic Level Detected: {} %'.format(out_sev), 
                            pred_obs = 'Obscene Level Detected: {} %'.format(out_obs),
                            pred_ins = 'Insult Level Detected: {} %'.format(out_ins),
                            pred_thr = 'Threat Level Detected: {} %'.format(out_thr),
                            pred_ide = 'Identity Hate Level Detected: {} %'.format(out_ide)                        
                            )
     
# Server reloads itself if code changes so no need to keep restarting:
app.run(debug=True)
