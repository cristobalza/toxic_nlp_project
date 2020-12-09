import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import time
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue

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

# Title
st.title('Toxicity of your words')

st.warning("This app may show innacurate results, so please don't take this results for granted. This app is still under construction and by no means it is ready to be used in the real world. Thanks for your understanding ")

status = st.radio("Do you understand the above message? If so, please answer to continue with the app:", ("Yes", "No"))

if status == 'Yes':
    # Text input 
    time.sleep(5)
    text = st.text_area('Enter your Text', 'Type here' )

    if st.button('Predict'):
        # Take a string input from user
        user_input = text
        data = [user_input]

        # In hundreds
        vect = tox.transform(data)
        pred_tox = tox_model.predict_proba(vect)[:,1] 

        vect = sev.transform(data)
        pred_sev = sev_model.predict_proba(vect)[:,1] 

        vect = obs.transform(data)
        pred_obs = obs_model.predict_proba(vect)[:,1] 

        vect = thr.transform(data)
        pred_thr = thr_model.predict_proba(vect)[:,1] 

        vect = ins.transform(data)
        pred_ins = ins_model.predict_proba(vect)[:,1] 

        vect = ide.transform(data)
        pred_ide = ide_model.predict_proba(vect)[:,1] 

        # Round it
        out_tox = round(pred_tox[0], 2) * 100
        out_sev = round(pred_sev[0], 2) * 100
        out_obs = round(pred_obs[0], 2) * 100
        out_ins = round(pred_ins[0], 2) * 100
        out_thr = round(pred_thr[0], 2) * 100
        out_ide = round(pred_ide[0], 2) * 100

        bar_labels=['toxic',
        'severe_toxic',
        'obscene', 
        'insult',
        'threat',
            'identity_hate']
        data = [out_tox, out_sev, out_obs, out_ins, out_thr, out_ide]
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Make fake dataset
        height = data
        bars = bar_labels
        y_pos = np.arange(len(bars))
        
        # Create horizontal bars
        plt.barh(y_pos, height)
        
        # Create names on the y-axis
        plt.yticks(y_pos, bars)
        plt.xlim(0,100)
        plt.xlabel("% Toxicity Level Detected")

        st.pyplot()
        results = [ 'Toxic Level Detected: {} %'.format(out_tox),
                    'Severe Toxic Level Detected: {} %'.format(out_sev), 
                    'Obscene Level Detected: {} %'.format(out_obs),
                    'Insult Level Detected: {} %'.format(out_ins),
                    'Threat Level Detected: {} %'.format(out_thr),
                    'Identity Hate Level Detected: {} %'.format(out_ide)]

        for i in range(len(results)):
            st.success(results[i])
        
        st.balloons()
else :
    st.error(":(")
