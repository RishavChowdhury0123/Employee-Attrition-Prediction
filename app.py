import pandas as pd
import numpy as np
import pickle
import streamlit as st

st.set_page_config(page_title='Employee Attrition Predictor', layout='wide')

def encode(x):
    if x=='Yes':
        return 1
    else: 
        return 0

@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def load_data():
    path= r'C:\Users\DELL\Python files\Employee Attrition\df.pkl'
    with open(path, 'rb') as ref:
        df= pickle.load(ref)
    path= r'C:\Users\DELL\Python files\Employee Attrition\model.pkl'
    with open(path, 'rb') as ref:
        pipe= pickle.load(ref)
    
    return df, pipe

# To transform numbers to abbreviated format
def format_numbers(number, pos=None, fmt= '.0f'):
    fmt= '%'+fmt
    thousands, lacs, crores= 1_000, 1_00_000, 1_00_00_000
    if number/crores >=1:
        return (fmt+' Cr.') %(number/crores)
    elif number/lacs >=1:
        return (fmt+' Lacs.') %(number/lacs)
    elif number/thousands >=1:
        return (fmt+' K') %(number/thousands)
    else:
        return fmt %(number)

# Function for encoding multiple features
class CustomEncoder:

    def __init__(self, columns):
        self.columns= columns
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        from sklearn.preprocessing import LabelEncoder
        out= X.copy()
        if self.columns is not None:
            out[self.columns]= out[self.columns].apply(lambda x: LabelEncoder().fit_transform(x))
        else:
            out= out.apply(lambda x: LabelEncoder().fit_transform(x))
        return out
    
    def fit_transform(self, X, y=None):
        out= X.copy()
        return self.fit(out).transform(out)

df,pipe=load_data()

def main():
    st.title('Employee Attrition Prediction')

    cols= st.columns(4)
    age= cols[0].slider('Selact Age',min_value=18,max_value=60 , key='age')
    business_travel= cols[1].selectbox('Business Travel', bus_trav.keys(), key='business')
    department= cols[2].selectbox('Department',df.department.unique(), key='department')
    office_dist= cols[3].slider('Office distance from home(in Kms.)', min_value=1, max_value=30, key='office_dist')
    

    cols= st.columns(5)
    education= cols[0].slider('Educatiion (1 to 5)', min_value= 1, max_value=5, key= 'education')
    gender= cols[1].radio('Gender', ['Male','Female'], key='gender')
    income= cols[2].number_input('Income', min_value= 10000, max_value= 200000, key='income')
    salary_hike= cols[3].slider('Salary Hike(%)', min_value=1, max_value=25, key='salary_hike')
    years_with_curr= cols[4].slider('Years with current manager', min_value= 0, max_value= 20, key='manager')
    
    btn= st.button('Predict Attrition', key='button')
    if btn:
        X= df.copy()
        vals= [age, bus_trav.get(business_travel), department, office_dist, education, gender, income, salary_hike, years_with_curr]

 
        X= X.append(dict(zip(X.columns, vals)), ignore_index=True)
        X= CustomEncoder(X.select_dtypes('O').columns).fit_transform(X)

        pred= pipe.predict_proba(X.iloc[-2:,:])[-1,1]
        st.markdown('There is {:.0%} chance of attrition.'.format(pred))


bus_trav= {'Travel Rarely': 'Travel_Rarely', 'Travel Frequently':'Travel_Frequently','Non Travel':'Non-Travel'}

if __name__=='__main__':
    main()
