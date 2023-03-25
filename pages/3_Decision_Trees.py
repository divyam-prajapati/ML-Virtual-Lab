import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris


st.set_page_config(page_title='Decision Tree',layout='wide')


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
    dc = DecisionTreeClassifier(
        #ccp_alpha=ccp_alpha,
        class_weight=class_weight,
        criterion=criterion,
        #max_depth=max_depth,
        max_features=max_features,
        #max_leaf_nodes=max_leaf_nodes,
        #min_impurity_decrease=min_impurity_decrease,
        #min_impurity_split=min_impurity_split,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        #min_weight_fraction_leaf=min_weight_fraction_leaf,
        random_state=random_state,
        splitter=splitter,
    )
    
    clf = dc.fit(X_train, Y_train)
    plot_tree(clf)

    st.subheader('2. Model Performace')
    st.markdown('**Training set**')
    Y_pred_train = dc.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.write('Accuracy:')
    st.info(dc.score(X_test,Y_test))

    st.markdown('**2.2. Test set**')
    Y_pred_test = dc.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(dc.get_params())

    
st.write("""
# Decesion Tree VLab

In this virtual lab, the *DecisionTreeClassifier()* function is used in this simulation for build a classification model using the **Decision Tree** algorithm.

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
    criterion=st.sidebar.select_slider('criterion', options=['gini', 'entropy', 'log_loss'])
    #max_depth = st.sidebar.slider('max_depth', 1, 10, None, None)
    max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2', None])
    #max_leaf_nodes = st.sidebar.slider('max_leaf_nodes', 1, 10, 2, None)
    min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 10, 1, 1)
    min_samples_split = st.sidebar.slider('min_samples_split', 1, 10, 2, 2)
    random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, None)
    splitter = st.sidebar.select_slider('splitter', options=['best', 'random'])

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
        Y = pd.Series(iris.target, name='target')
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