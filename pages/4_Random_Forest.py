import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt


st.set_page_config(page_title='Random Forest',layout='wide')


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

    

    rf = RandomForestRegressor(
        n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs
    )
    
# max_depthint, default=None min_weight_fraction_leaffloat, default=0.0 max_leaf_nodesint, default=None min_impurity_decreasefloat, default=0.0 verboseint, default=0 warm_startbool, default=False class_weight{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None ccp_alphanon-negative float, default=0.0 max_samplesint or float, default=None
    
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performace')
    st.markdown('**Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.write('Accuracy:')
    st.info(rf.score(X_test,Y_test))

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

    st.subheader('5. Graph')
    fig, ax = plt.subplots()
    ax.scatter(df['age'], df['response'])
    ax.set_xlabel('age')
    ax.set_ylabel('Response')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(df['bmi'], df['response'], c='red')
    ax.set_xlabel('bmi')
    ax.set_ylabel('Response')
    st.pyplot(fig)

    
st.write("""
# Random Forest VLab

In this virtual lab, the *RandomForestRegressor()* function is used in this simulation for build a regression model using the **Random Forest** algorithm.

Try adjusting the hyperparameters!

""")


with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")


with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        boston = load_diabetes()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

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




