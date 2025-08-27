import streamlit as st
import numpy as np
import pickle
import pandas as pd


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

kidney_model = load_pickle('kidney_model.pkl')
kidney_scaler = load_pickle('kidney_scaler.pkl')
kidney_encoders= load_pickle('kidney_label_encoders.pkl')

liver_model = load_pickle('liver_model.pkl')
liver_scaler = load_pickle('liver_scaler.pkl')
gender_mapping= load_pickle('gender_mapping.pkl')
threshold = load_pickle('threshold.pkl')

parkinsons_model = load_pickle('parkinsons_rf_model.pkl')
parkinsons_scaler = load_pickle('parkinsons_scaler.pkl')

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-Disease Prediction System", layout="wide")
st.title("🧠🩺 Disease Prediction App")
st.markdown("This app predicts the presence of **Kidney**, **Liver**, or **Parkinson's** disease using trained ML models.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Kidney Disease", "🧬 Liver Disease", "🧠 Parkinson’s Disease"])

# ------------------------ KIDNEY ------------------------
with tab1:

    st.title("Kidney Disease Prediction")
    
    # Numeric inputs
    age = st.number_input("Age", min_value=1, max_value=120, step=1, key="age")
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, key="bp")
    sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.05, step=0.01, format="%.2f", key="sg")
    al = st.number_input("Albumin", min_value=0, max_value=5, step=1, key="al")
    su = st.number_input("Sugar", min_value=0, max_value=5, step=1, key="su")
    bgr = st.number_input("Blood Glucose Random", min_value=0, max_value=500, step=1, key="bgr")
    bu = st.number_input("Blood Urea", min_value=0, max_value=300, step=1, key="bu")
    sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, step=0.1, key="sc")
    sod = st.number_input("Sodium", min_value=0.0, max_value=200.0, step=0.1, key="sod")
    pot = st.number_input("Potassium", min_value=0.0, max_value=20.0, step=0.1, key="pot")
    hemo = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, step=0.1, key="hemo")
    pcv = st.number_input("Packed Cell Volume", min_value=0, max_value=60, step=1, key="pcv")
    wc = st.number_input("White Blood Cell Count", min_value=0, max_value=25000, step=100, key="wc")
    rc = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, step=0.1, key="rc")

    # Categorical inputs
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"], key="rbc")
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"], key="pc")
    pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"], key="pcc")
    ba = st.selectbox("Bacteria", ["present", "notpresent"], key="ba")
    htn = st.selectbox("Hypertension", ["yes", "no"], key="htn")
    dm = st.selectbox("Diabetes Mellitus", ["yes", "no"], key="dm")
    cad = st.selectbox("Coronary Artery Disease", ["yes", "no"], key="cad")
    appet = st.selectbox("Appetite", ["good", "poor"], key="appet")
    pe = st.selectbox("Pedal Edema", ["yes", "no"], key="pe")
    ane = st.selectbox("Anemia", ["yes", "no"], key="ane")

    # Convert categorical to numeric
    def encode(value, mapping):
        return mapping[value]

    rbc = encode(rbc, {"normal": 0, "abnormal": 1})
    pc = encode(pc, {"normal": 0, "abnormal": 1})
    pcc = encode(pcc, {"notpresent": 0, "present": 1})
    ba = encode(ba, {"notpresent": 0, "present": 1})
    htn = encode(htn, {"no": 0, "yes": 1})
    dm = encode(dm, {"no": 0, "yes": 1})
    cad = encode(cad, {"no": 0, "yes": 1})
    appet = encode(appet, {"good": 0, "poor": 1})
    pe = encode(pe, {"no": 0, "yes": 1})
    ane = encode(ane, {"no": 0, "yes": 1})

    # Arrange input in the correct order
    features = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,
                        sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])

    # Scale input
    features_scaled = kidney_scaler.transform(features)

    # Prediction
    if st.button("🔍 Predict"):
        prediction = kidney_model.predict(features_scaled)[0]
        if prediction == 1:
            st.error("✅ The model predicts: **No CKD**")
        else:
            st.success("⚠️ The model predicts: **Chronic Kidney Disease (CKD)** ")
# ------------------------ LIVER ------------------------
with tab2:
    st.header("Liver Disease Prediction")

    liver_features = [
        'Age','Gender','Total_Bilirubin','Direct_Bilirubin',
        'Alkaline_Phosphotase','Alamine_Aminotransferase',
        'Aspartate_Aminotransferase','Total_Protiens',
        'Albumin','Albumin_and_Globulin_Ratio'
    ]

    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin", min_value=0.0, step=0.1)
    db = st.number_input("Direct Bilirubin", min_value=0.0, step=0.1)
    alkp = st.number_input("Alkaline Phosphotase", min_value=0, step=1)
    alat = st.number_input("Alamine Aminotransferase", min_value=0, step=1)
    asat = st.number_input("Aspartate Aminotransferase", min_value=0, step=1)
    tp = st.number_input("Total Proteins", min_value=0.0, step=0.1)
    alb = st.number_input("Albumin", min_value=0.0, step=0.1)
    agr = st.number_input("Albumin and Globulin Ratio", min_value=0.0, step=0.1)

    # Use gender mapping pickle
    gender_val = gender_mapping[gender]

    liver_input = [age, gender_val, tb, db, alkp, alat, asat, tp, alb, agr]

    if st.button("Predict Liver Disease"):
        try:
            # Scale input
            liver_input_scaled = liver_scaler.transform([liver_input])
            liver_prob = liver_model.predict_proba(liver_input_scaled)[:,1][0]

            liver_pred = 1 if liver_prob >= threshold else 0

            # Display result
            if liver_pred == 1:
                st.warning(f"⚠️ Likely Liver Disease Detected")
            else:
                st.success(f"✅ No Liver Disease Detected")

        except Exception as e:
            st.error(f"Invalid input: {e}")

# ------------------------ PARKINSONS ------------------------
with tab3:
    st.header("Parkinson’s Disease Prediction")

    parkinsons_features = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
        'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP',
        'MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5',
        'MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1',
        'spread2','D2','PPE']

    parkinsons_input = []
    for feature in parkinsons_features:
        val = st.text_input(f"{feature}", key=feature)
        parkinsons_input.append(val)

    if st.button("Predict Parkinson’s Disease"):
        try:
            parkinsons_input = [float(x) for x in parkinsons_input]
            parkinsons_input = parkinsons_scaler.transform([parkinsons_input])
            result = parkinsons_model.predict(parkinsons_input)
            st.success("Likely Parkinson’s Disease" if result[0]==1 else "No Parkinson’s Disease")
        except Exception as e:
            st.error(f"Invalid input: {e}")