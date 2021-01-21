import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.style as style
style.use('fivethirtyeight')
import seaborn as sns
import time

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

code_repo = '[Github repo](https://github.com/cristobalza/toxic_nlp_project)'
ibm_docu = '[Documentation](https://developer.ibm.com/technologies/artificial-intelligence/models/max-toxic-comment-classifier/)'
kaggle_link = '[Kaggle Competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)'
# Title
st.title('Toxic Word Detector App')

st.markdown('### Introduction ')
st.markdown('\n\n')
st.markdown("When we are engaging  in text conversations in online platforms, in most cases, we know the intention of our words. However, sometimes we don't know how *bad* our words can be interpreted.")
st.markdown("The purpose of this app is not silence people that have discriminatory tendencies. Instead, I think it is better to let know people why the things they say can hurt other people's feelings.")
st.markdown("This app uses Machine Learning to predict your text against different categories of language toxicity such as `toxic, severe toxic, obscene, insulting, threating, identity hateful` categories. To see more about the model and work pipeline used in this project, you can visit my "+code_repo+".", unsafe_allow_html=True)
st.markdown("It is important for the user/reader to know that the data used to train the models for this project was from a "+kaggle_link+" and the documentation of the toxic categories can be found here "+ibm_docu+".",  unsafe_allow_html=True)



st.info("This app may show innacurate results, so please don't take this results for granted. This app is still under construction and by no means it is ready to be used in the real world situations such as social media purposes and others. \n \n Thanks for your understanding ")

# status = st.radio("Do you understand the above message? If so, please answer to continue with the app:", ("Yes", "No"))

# if status == 'Yes':
if st.checkbox("Do you understand the above message? If so, please answer to continue with the app:"):
    # Text input 
    # time.sleep(5)

    st.markdown('### Instructions')

    st.markdown('- Step 1: Input desired text data')
    st.markdown('- Step 2: Press `Predict` button and see how toxic was your text.')

    text = st.text_area('Enter your Text', 'Type here' )

    if st.button('Predict'):
        # Take a string input from user
        user_input = text
        data = [user_input]

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

        # Round it and multiply by 100
        out_tox = round(pred_tox[0], 2) * 100
        out_sev = round(pred_sev[0], 2) * 100
        out_obs = round(pred_obs[0], 2) * 100
        out_ins = round(pred_ins[0], 2) * 100
        out_thr = round(pred_thr[0], 2) * 100
        out_ide = round(pred_ide[0], 2) * 100

        bar_labels=['toxic','severe_toxic','obscene', 'insult','threat', 'identity_hate']
        data = [out_tox, out_sev, out_obs, out_ins, out_thr, out_ide]
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # # Graph
        toxic_cols_perc=['toxic %','severe_toxic %','obscene %', 'insult %','threat %', 'identity_hate %'][::-1]
        df = pd.DataFrame(data = [data[::-1]], columns = bar_labels)
        # height = data
        # bars = bar_labels
        # y_pos = np.arange(len(bars))
        
        # # Create horizontal bars
        # plt.barh(y_pos, height)
        
        # # Format graph
        # plt.yticks(y_pos, bars)
        sns.barplot(data=df, orient = 'h', palette='inferno')
        plt.xlim(0,100)
        plt.xlabel("% Toxicity Level Detected")
        st.pyplot()


        st.table(df)

        #End
        st.balloons()
# else :
#     st.error(":(")
link = '[cristobalza.com](http://cristobalza.com)'
st.markdown('See my website and other projects @ '+link, unsafe_allow_html=True)
