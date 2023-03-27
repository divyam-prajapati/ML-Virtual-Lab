import streamlit as st

st.set_page_config(
    page_title="VLAB",
    page_icon="📈",
    layout='wide',
)

st.write("""# Machine Learning Virtual Lab by KJSIT""")

st.markdown(
    """
    ##### In this virtual lab we will go through Linear Regression, Logistic Regression, Decision Trees and Random Forest.
     

    ##### How to use  
    **Step 1: Select a algorithm from the sidebar. **
    
    **Step 2: Upload your own dataset or use one given by default.** 
    
    **Step 3: Try adjusting the hyperparameters in the sidebar.**

    **Step 4: Fill out the feedback form. **
"""
)

st.markdown("""
    <div id='footer'>
        Developed by Department of Computer Engineering
        <br>
        Made in    
        <a href="//streamlit.io" target="_blank"><img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" alt="Streamlit" /></a> 
        by <B>TAHA LOKAT</B> and <B>DIVYAM PRAJAPATI</B>
        <br>
        Guided By: <B>Prof. Kavita Bathe</B> 
    </div>
""", unsafe_allow_html=True)

st.markdown(""" <style>
header {background: rgba(0,0,0,0)!important;}
#MainMenu {visibility: hidden;}
h1 {margin: 0rem; text-align: center; margin-bottom: 2rem;}
footer {visibility: hidden!important;}
.logo {width: 25px; height: 25px;} 
.block-container {padding: 0 8rem!important; margin: 0rem; display: flex; justify-content: center; align-items: center;}
#footer {bottom: 1rem; position: fixed; left: 50%; margin-left: -210px; width: 420px; color: rgba(22,36,67, 0.8); text-align: center!important;}
#footer a {cursor: pointer; text-decoration: none; text-style: none; padding: 0.1rem;}
#footer a img {width: 25px; height: 13.61px;}
.stMarkdown ol, p {margin: 1rem 2rem; text-align: justify;}
.stMarkdown h4 {margin: 0rem 12rem; text-align: center;}
[data-testid="stSidebarNav"] {
    background-image: url(https://kjsit.somaiya.edu.in/assets/kjsieit/images/Logo/kjsieit-logo.svg);
    background-repeat: no-repeat;
    padding-top: 120px;
    background-position: 20px 20px;
}
</style> """, unsafe_allow_html=True)



#  .appview-container section {width: 200px!important;}
#header {display: flex; flex: 1; flex-direction: row; justify-content: space-between; align-items: center; margin: 1rem 5rem;}
#header #kjsit {height: 50%; margin: 0rem;}
#header h1 {margin: 0rem;font-size: 2.5rem;}
# st.markdown("""
#     <div id='header'>
#         <h1>Machine Learning Virtual Lab by KJSIT</h1>
#     </div>
# """, unsafe_allow_html=True)
#title {color: rgba(22,36,67, 1); text-align; center;}
#header {display: flex; flex: 1; flex-direction: row; justify-content: center; align-items: center;}
# 
# .stMarkdown h4, h5 {margin: 0rem 10rem; text-align: justify;}
# .stMarkdown ol, p {margin: 1rem 12rem; text-align: justify;}
# .stMarkdown h4 {margin: 0rem 12rem; text-align: center;}