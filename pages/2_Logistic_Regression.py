import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris


st.set_page_config(page_title='Logistic Regression',layout='wide')


def build_model(df):
    X = df.iloc[:,:4]
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
    lr = LogisticRegression(
        class_weight= class_weight,
        dual= dual,
        fit_intercept= fit_intercept,
        l1_ratio= l1_ratio,
        max_iter= max_iter,
        multi_class= multi_class,
        n_jobs= n_jobs,
        penalty= penalty,
        random_state= 0,
        solver= solver,
        warm_start= warm_start
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
# Logistic Regression VLab

In this virtual lab, the *LogisticRegression()* function is used in this simulation for build a regression model using the **Logistic Regression** algorithm.

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
    class_weight = st.sidebar.select_slider('class_weight', options=['auto', 'balanced', None])
    dual = st.sidebar.select_slider('dual', options=[False, True])
    fit_intercept = st.sidebar.select_slider('fit_intercept', options=[False, True])
    l1_ratio = st.sidebar.slider('Seed number (l1_ratio)', 0, 1, None, None)
    max_iter = st.sidebar.slider('Number of iterations (max_iter)', 0, 1000, 100, 100)
    multi_class = st.sidebar.select_slider('multi_class', options=['auto', 'ovr', 'multinomial'])
    n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
    penalty = st.sidebar.select_slider('penalty', options=['l2', 'l1', 'elasticnet'])
    random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    solver = st.sidebar.select_slider('solver', options=['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
    warm_start = st.sidebar.select_slider('warm_start', options=[False, True])


st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        Y = pd.Series(iris.target, name='response')
        df = pd.concat( [X,Y], axis=1 )
        st.markdown('The Iris dataset is used as the example.')
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