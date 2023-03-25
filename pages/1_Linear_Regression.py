import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes


st.set_page_config(page_title='Linear Regression',layout='wide')


def build_model(df):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1] 
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=(100-split_size)/100)

    st.markdown('**1.2 Data Splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Testing set')
    st.info(X_test.shape)

    st.markdown('**Variables details**')
    st.write('The X Variable')
    st.info(X.columns)
    st.write('The Y Variable')
    st.info(Y.name)

    lr = LinearRegression(
        fit_intercept=fit_intercept, 
        copy_X=copy_X, 
        n_jobs=parameter_n_jobs, 
        normalize=normalize,
    )
    
    lr.fit(X_train, Y_train)

    st.subheader('2. Model Performace')
    st.markdown('**Training set**')
    Y_pred_train = lr.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.write('Accuracy:')
    st.info(lr.score(X_test,Y_test))

    st.markdown('**2.2. Test set**')
    Y_pred_test = lr.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(lr.get_params())

    
st.write("""
# Linear Regression VLab

In this virtual lab, the *LinearRegressionRegressor()* function is used in this simulation for build a regression model using the **Linear Regression** algorithm.

Try adjusting the hyperparameters!

""")


with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://github.com/scikit-learn/scikit-learn/blob/9aaed498795f68e5956ea762fef9c440ca9eb239/sklearn/datasets/data/diabetes_data_raw.csv.gz)
""")


with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('3. General Parameters'):
    fit_intercept = st.sidebar.select_slider('fit_intercept', options=[True, False])
    copy_X = st.sidebar.select_slider('copy_X', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
    normalize = st.sidebar.select_slider('Normalize', options=[False, True])


st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )
        st.markdown('The Diabetes dataset is used as the example.')
        st.write(df.head(5))
        #print(type(df['X'][0]))
        build_model(df)

st.markdown(
"""
<style>
    [data-testid="stSidebarNav"] {
        background-image: url(https://kjsit.somaiya.edu.in/assets/kjsieit/images/Logo/kjsieit-logo.svg);
        background-repeat: no-repeat;
        padding-top: 120px;
        background-position: 20px 20px;
    }

</style>
"""
, unsafe_allow_html=True)