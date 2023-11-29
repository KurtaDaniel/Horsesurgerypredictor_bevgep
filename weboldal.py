import streamlit as st

import pandas as pd

#import os
#path = os.getcwd()
#st.write(path)

st.markdown("<h1 style='text-align: center; '>Horse surgery predictor (RandomForest)</h1>", unsafe_allow_html=True)

import joblib
model_RandomForestClassifier = joblib.load("random_forest.joblib")

from sklearn.preprocessing import LabelEncoder

st.markdown("""
    <style>
    .stRadio [role=radiogroup]{
        align-items: center;
        
    }
    </style>
""",unsafe_allow_html=True)

st.markdown("""
    <style>
    .stSlider{
        height:30px;
    }
    </style>
""",unsafe_allow_html=True)

st.markdown("""
    <style>
    .stButton{
        font-weight: bold;
    }
    </style>
""",unsafe_allow_html=True)




#Rectal Temp
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Rectal Temp</h2>", unsafe_allow_html=True)
rectal_temp = st.slider("Rectal temp", 0.0, 100.0,label_visibility="hidden")

#packed_cell_volume
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Packed cell volume</h2>", unsafe_allow_html=True)
packed_cell_volume = st.slider("packed_cell_volume", 0, 100,label_visibility="hidden")

#total_protein
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Total protein</h2>", unsafe_allow_html=True)
total_protein = st.slider("total_protein", 0.0, 100.0,label_visibility="hidden")

#lesion_1
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Lesion_1</h2>", unsafe_allow_html=True)
lesion_1 = st.slider("lesion_1", 0, 15000,label_visibility="hidden")

#lesion_2
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Lesion_2</h2>", unsafe_allow_html=True)
lesion_2 = st.slider("lesion_2", 0, 15000,label_visibility="hidden")

#lesion_3
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Lesion_3</h2>", unsafe_allow_html=True)
lesion_3 = st.slider("lesion_3", 0, 15000,label_visibility="hidden")

#pulse
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Pulse</h2>", unsafe_allow_html=True)
pulse = st.slider("Pulse", 0, 200,label_visibility="hidden")


#respiratory_rate
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Respiratory rate</h2>", unsafe_allow_html=True)
respiratory_rate = st.slider("respiratory_rate", 0, 100,label_visibility="hidden")


#Age
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Age</h2>", unsafe_allow_html=True)
label_encoder_x = LabelEncoder()
age = st.radio(
    "Age",
    ["adult","young"],
    label_visibility="hidden")
label_encoder_x.fit(["adult","young"]) #0 1
age = label_encoder_x.transform([age])



#temp_of_extremities
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Temp of extremities</h2>", unsafe_allow_html=True)
temp_of_extremities = st.radio(
    "temp_of_extremities",
    ["cool","normal","warm","cold"],
    label_visibility="hidden")
label_encoder_x.fit(["cool","normal","warm","cold"]) #1 2 3 0
temp_of_extremities = label_encoder_x.transform([temp_of_extremities])

#peripheral_pulse
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Peripheral pulse</h2>", unsafe_allow_html=True)
peripheral_pulse = st.radio(
    "peripheral_pulse",
    ["normal","reduced","absent","increased"],
    label_visibility="hidden")
label_encoder_x.fit(["normal","reduced","absent","increased"]) #
peripheral_pulse = label_encoder_x.transform([peripheral_pulse])

#mucous_membrane
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Mucous membrane</h2>", unsafe_allow_html=True)
mucous_membrane = st.radio(
    "mucous_membrane",
    ["normal_pink","pale_pink","pale_cyanotic","bright_pink","bright_red","dark_cyanotic"],
    label_visibility="hidden")
label_encoder_x.fit(["normal_pink","pale_pink","pale_cyanotic","bright_pink","bright_red","dark_cyanotic"])
mucous_membrane = label_encoder_x.transform([mucous_membrane])

#capillary_refill_time
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Capillary refill time</h2>", unsafe_allow_html=True)
capillary_refill_time = st.radio(
    "capillary_refill_time",
    ["less_3_sec","more_3_sec","3"],
    label_visibility="hidden")
label_encoder_x.fit(["less_3_sec","more_3_sec","3"])
capillary_refill_time = label_encoder_x.transform([capillary_refill_time])

#pain
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Pain</h2>", unsafe_allow_html=True)
pain = st.radio(
    "pain",
    ["mild_pain","depressed","extreme_pain","severe_pain","alert"],
    label_visibility="hidden")
label_encoder_x.fit(["mild_pain","depressed","extreme_pain","severe_pain","alert"])
pain = label_encoder_x.transform([pain])

#peristalsis
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Peristalsis</h2>", unsafe_allow_html=True)
peristalsis = st.radio(
    "peristalsis",
    ["hypomotile","absent","hypermotile","normal"],
    label_visibility="hidden")
label_encoder_x.fit(["hypomotile","absent","hypermotile","normal"])
peristalsis = label_encoder_x.transform([peristalsis])

#abdominal_distention
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Abdominal distention</h2>", unsafe_allow_html=True)
abdominal_distention = st.radio(
    "abdominal_distention",
    ["none","slight","moderate","severe"],
    label_visibility="hidden")
label_encoder_x.fit(["none","slight","moderate","severe"])
abdominal_distention = label_encoder_x.transform([abdominal_distention])

#nasogastric_tube
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Nasogastric tube</h2>", unsafe_allow_html=True)
nasogastric_tube = st.radio(
    "nasogastric_tube",
    ["slight","none","significant"],
    label_visibility="hidden")
label_encoder_x.fit(["slight","none","significant"])
nasogastric_tube = label_encoder_x.transform([nasogastric_tube])

#nasogastric_reflux
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Nasogastric reflux</h2>", unsafe_allow_html=True)
nasogastric_reflux = st.radio(
    "nasogastric_reflux",
    ["none","more_1_liter","less_1_liter"],
    label_visibility="hidden")
label_encoder_x.fit(["none","more_1_liter","less_1_liter"])
nasogastric_reflux = label_encoder_x.transform([nasogastric_reflux])

#rectal_exam_feces
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Rectal exam feces</h2>", unsafe_allow_html=True)
rectal_exam_feces = st.radio(
    "rectal_exam_feces",
    ["absent","normal","decreased","increased"],
    label_visibility="hidden")
label_encoder_x.fit(["absent","normal","decreased","increased"])
rectal_exam_feces = label_encoder_x.transform([rectal_exam_feces])

#abdomen
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Abdomen</h2>", unsafe_allow_html=True)
abdomen = st.radio(
    "abdomen",
    ["distend_large","distend_small","normal","other","firm"],
    label_visibility="hidden")
label_encoder_x.fit(["distend_large","distend_small","normal","other","firm"])
abdomen = label_encoder_x.transform([abdomen])

#outcome
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Outcome</h2>", unsafe_allow_html=True)
outcome = st.radio(
    "outcome",
    ["lived","died","euthanized"],
    label_visibility="hidden")
label_encoder_x.fit(["lived","died","euthanized"])
outcome = label_encoder_x.transform([outcome])


#surgical_lesion
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Surgical lesion</h2>", unsafe_allow_html=True)
surgical_lesion = st.radio(
    "surgical_lesion",
    ["yes","no"],
    label_visibility="hidden")
label_encoder_x.fit(["yes","no"])
surgical_lesion = label_encoder_x.transform([surgical_lesion])

#cp_data
st.markdown("<h2 style='text-align: center; height:0px; font-size: 20px; '>Cp data</h2>", unsafe_allow_html=True)
cp_data = st.radio(
    "cp_data",
    ["no","yes"],
    label_visibility="hidden")
label_encoder_x.fit(["no","yes"])
cp_data = label_encoder_x.transform([cp_data])



data = [{'age': age, 'rectal_temp': rectal_temp, 'pulse': pulse, 'respiratory_rate': respiratory_rate,
    'temp_of_extremities': temp_of_extremities, 'peripheral_pulse': peripheral_pulse, 'mucous_membrane': mucous_membrane,
    'capillary_refill_time': capillary_refill_time, 'pain': pain, 'peristalsis': peristalsis, 'abdominal_distention': abdominal_distention,
    'nasogastric_tube': nasogastric_tube, 'nasogastric_reflux': nasogastric_reflux, 'rectal_exam_feces': rectal_exam_feces,
    'abdomen': abdomen, 'packed_cell_volume': packed_cell_volume, 'total_protein': total_protein, 'outcome': outcome,
    'surgical_lesion': surgical_lesion, 'lesion_1': lesion_1, 'lesion_2': lesion_2, 'lesion_3': lesion_3, 'cp_data': cp_data}] 




  
def kiir():
    dff = pd.DataFrame(data) 
    #st.write(model_RandomForestClassifier.predict(dff))
    if model_RandomForestClassifier.predict(dff)==0:
        #st.write("No")
        st.markdown("<h3 style='text-align: center; height:0px; font-size: 20px;font-weight: normal; '>No surgery was needed</h3>", unsafe_allow_html=True)
    else:
        #st.write("Yes")
        st.markdown("<h3 style='text-align: center; height:0px; font-size: 20px;font-weight: normal; '>Surgery was needed</h3>", unsafe_allow_html=True)



writeOut = False


if st.button('TN',use_container_width=True):#good
    data = [{'age': 0, 'rectal_temp': 38.5, 'pulse': 66, 'respiratory_rate': 28,
        'temp_of_extremities': 1, 'peripheral_pulse': 3, 'mucous_membrane': 3,
        'capillary_refill_time': 2, 'pain': 2, 'peristalsis': 0, 'abdominal_distention': 2,
        'nasogastric_tube': 2, 'nasogastric_reflux': 2, 'rectal_exam_feces': 1,
        'abdomen': 0, 'packed_cell_volume': 45, 'total_protein': 8.4, 'outcome': 0,
        'surgical_lesion': 0, 'lesion_1': 11300, 'lesion_2': 0, 'lesion_3': 0, 'cp_data': 0}] 
    writeOut = True

if st.button('TP',use_container_width=True):#good
    data = [{'age': 0, 'rectal_temp': 37.4, 'pulse': 50, 'respiratory_rate': 32,
        'temp_of_extremities': 1, 'peripheral_pulse': 3, 'mucous_membrane': 3,
        'capillary_refill_time': 1, 'pain': 4, 'peristalsis': 0, 'abdominal_distention': 1,
        'nasogastric_tube': 2, 'nasogastric_reflux': 2, 'rectal_exam_feces': 3,
        'abdomen': 0, 'packed_cell_volume': 45, 'total_protein': 7.9, 'outcome': 2,
        'surgical_lesion': 1, 'lesion_1': 2208, 'lesion_2': 0, 'lesion_3': 0, 'cp_data': 1}] 
    writeOut = True

if st.button('FN',use_container_width=True):
    data = [{'age': 0, 'rectal_temp': 39.2, 'pulse': 88, 'respiratory_rate': 20,
        'temp_of_extremities': 1, 'peripheral_pulse': 2, 'mucous_membrane': 4,
        'capillary_refill_time': 1, 'pain': 3, 'peristalsis': 0, 'abdominal_distention': 3,
        'nasogastric_tube': 2, 'nasogastric_reflux': 2, 'rectal_exam_feces': 0,
        'abdomen': 4, 'packed_cell_volume': 50, 'total_protein': 85, 'outcome': 1,
        'surgical_lesion': 0, 'lesion_1': 2208, 'lesion_2': 0, 'lesion_3': 0, 'cp_data': 0}] 
    writeOut = True

if st.button('FP',use_container_width=True):#good
    data = [{'age': 0, 'rectal_temp': 38.5, 'pulse': 96.0, 'respiratory_rate': 36.0,
        'temp_of_extremities': 1, 'peripheral_pulse': 3, 'mucous_membrane': 3,
        'capillary_refill_time': 2, 'pain': 1, 'peristalsis': 0, 'abdominal_distention': 3,
        'nasogastric_tube': 0, 'nasogastric_reflux': 0, 'rectal_exam_feces': 0,
        'abdomen': 0, 'packed_cell_volume': 70.0, 'total_protein': 8.5, 'outcome': 0,
        'surgical_lesion': 1, 'lesion_1': 1400, 'lesion_2': 0, 'lesion_3': 0, 'cp_data': 1}] 
    writeOut = True

if st.button('Predict using your own data',use_container_width=True):
    data = [{'age': int(age), 'rectal_temp': rectal_temp, 'pulse': pulse, 'respiratory_rate': respiratory_rate,
       'temp_of_extremities': int(temp_of_extremities), 'peripheral_pulse': int(peripheral_pulse), 'mucous_membrane': int(mucous_membrane),
       'capillary_refill_time': int(capillary_refill_time), 'pain': int(pain), 'peristalsis': int(peristalsis), 'abdominal_distention': int(abdominal_distention),
       'nasogastric_tube': int(nasogastric_tube), 'nasogastric_reflux': int(nasogastric_reflux), 'rectal_exam_feces': int(rectal_exam_feces),
       'abdomen': int(abdomen), 'packed_cell_volume': packed_cell_volume, 'total_protein': total_protein, 'outcome': int(outcome),
       'surgical_lesion': int(surgical_lesion), 'lesion_1': lesion_1, 'lesion_2': lesion_2, 'lesion_3': lesion_3, 'cp_data': int(cp_data)}] 
    writeOut = True

if writeOut:
    kiir()

    
#0 no 1 yes
