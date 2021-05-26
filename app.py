from flask import Flask,render_template,redirect,request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

file_curr = open("SVC_random_best",'rb')
model = pickle.load(file_curr)
file_curr.close()
cols = ['CS_SatisfactionWithPlanPhysicians_Coordination of care', 'CS_GettingCare_Getting care easily', 'CS_GettingCare_Getting care quickly', 'CS_SatisfactionWithPlanServices_Handling claims', 'CS_SatisfactionWithPlanServices_Rating of health plan', 'CS_SatisfactionWithPlanPhysicians_Rating of primary-care doctor', 'CS_SatisfactionWithPlanPhysicians_Rating of specialists', 'PRV_OtherPreventiveServices_Adult BMI assessment', 'PRV_ChildrenAndAdolescentWellCare_BMI percentile assessment', 'PRV_CancerScreening_Breast cancer screening', 'PRV_CancerScreening_Cervical cancer screening', 'PRV_ChildrenAndAdolescentWellCare_Childhood immunizations status- combination 10', 'PRV_OtherPreventiveServices_Chlamydia screening', 'PRV_CancerScreening_Colorectal cancer screening', 'PRV_OtherPreventiveServices_Flu shots for adults', "PRV_Women'sReproductiveHealth_Postpartum care", "PRV_Women'sReproductiveHealth_Prenatal checkups", 'TRT_OtherTreatmentMeasures_Appropriate antibiotic use, adults with acute bronchitis', 'TRT_OtherTreatmentMeasures_Appropriate antibiotic use, children with colds', 'TRT_OtherTreatmentMeasures_Appropriate testing and care, children with sore throat', 'TRT_OtherTreatmentMeasures_Appropriate use of imaging studies for low back pain', 'TRT_Asthma_Asthma control', 'TRT_Asthma_Asthma drug management', 'TRT_Diabetes_Blood pressure control (140/90)', 'TRT_OtherTreatmentMeasures_Bronchodilator after hospitalization for acute COPD', 'TRT_MentalAndBehavioralHealth_Cholesterol and blood sugar testing for youth on antipsychotic medications', 'TRT_HeartDisease_Controlling high blood pressure', 'TRT_Diabetes_Eye exams', 'TRT_MentalAndBehavioralHealth_First-line psychosocial care for youth on antipsychotic medications', 'TRT_MentalAndBehavioralHealth_Follow-up after hospitalization for mental illness', 'TRT_Diabetes_Glucose control', 'TRT_OtherTreatmentMeasures_Steroid after hospitalization for acute COPD']
label_dict = [2.5,3.0,3.5,4.0,4.5,5.0]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        int_features = [x for x in request.form.values()]
        final = np.array(int_features)
        data_unseen = pd.DataFrame([final],columns = cols)
        from sklearn.preprocessing import StandardScaler
        se = StandardScaler()
        se.fit_transform(data_unseen)
        prediction = model.predict(data_unseen)
        return render_template("home.html",pred = "The rank predicted by our model is {}".format(label_dict[int(prediction)]))
    else:
        return redirect('/')

