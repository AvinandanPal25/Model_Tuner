# Importing Packages:---
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, mean_squared_log_error, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
#-------------------------------------------------------------------------------------------------------
st.set_page_config(page_title='Hypertunning_Model_Parameters', page_icon=None, layout='centered', initial_sidebar_state='collapsed')

# Creating containers for main Display
header = st.beta_container()
about = st.beta_container()
dataset = st.beta_container()

# Creating Columns in Main Display, to display the Performance of the Model
trXshape,trYshape, tXshape, tYshape = st.beta_columns((10,9,10,9))
outcome, divider, evaluate = st.beta_columns((5.3,0.7,11))
#-------------------------------------------------------------------------------------------------------------

with header:
    st.markdown("<h1 style='text-align: center; color:#582F9A ;'>Welcome to my Machine Learning MODEL_TUNER App!!!</h1>", unsafe_allow_html=True)
    #st.title("Welcome to my ML App!!!")
with about:
    st.header("**This Application is useful to find out the BEST Score of a Selected Model on a Particular Dataset. This can be achieved by tunning certain parameters specific to the Model.**")
            

col1, col2= st.sidebar.beta_columns((1,2))
col2.markdown("#### **---@Avinandan_Pal**")

st.sidebar.markdown("# **PREDICTOR PALLETE**")

# Creating containers for the Sidebar
selection1 = st.sidebar.beta_container()

#---------------------------------------------------------------------------------------------------------------
with selection1: #Dataset and Model Selection, and specifying Test Size and Random State 

    df = st.selectbox('Select your Dataset',options=['Select among these-----','Bank_Churn','Social_Network_Ads','House_Price'],index=0)
    model = st.radio('Select your Model', options = ['Gradient Boosting','XGboost'],index=0)
    st.write("----------")
    train_split_size = st.slider("Choose your preferred Train Size:", min_value=0.7, max_value=0.9, value=0.8, step=0.05)
    random_state = st.number_input("Choose Random State:", min_value=0, max_value=None, value=2, step=1)

    #The following 4 lines are for avioding an error, without these and checkbox unselected exp is not defined
    if df == 'Select among these-----':
        exp=True
    else:
        exp=False

    Param_select = st.checkbox("Select to TUNE Parameters" )
    if Param_select:
        exp=True

# Creating Expander to tune paramerters
param_expander = st.sidebar.beta_expander('', expanded=exp)


if df == 'Select among these-----':
    st.subheader( "**---To Proceed, reach out to the PREDICTOR PALLETE---**")
    st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#-------------------------------------------------------------------------------------------------------------
# Defining Required faunctions:---

@st.cache
def get_df(df_name):
    df=pd.read_csv(df_name)
    return df


def GB_params_C(): #user input parameters for Gradient Boost Classifier Model
    loss = param_expander.radio('LOSS function:---', options = ['deviance','exponential'], index=0) 
    learning_rate = param_expander.selectbox("Choose Learning Rate:---", options=[0.001,0.005,0.007,0.01,0.03,0.05,0.1,0.3,0.5,0.8,1], index=5)
    subsample = param_expander.slider("Choose Fraction of Sample for Fitting:---", min_value=0.01, max_value=1.0, value=0.5,step=0.1)
    n_estimators = param_expander.number_input("Choose number of Estimators:---", min_value=5, max_value=1000, value=100, step=100)
    max_depth = param_expander.slider("Maximum Depth of the trees should be:---", min_value=1, max_value=32, value=3,step=1)
    min_samples_leaf = param_expander.slider("Min no. of sample at the Leaf Node:---", min_value=1, max_value=None, value=1,step=1)
    n_iter = param_expander.number_input("Type/Select no. of iteration before Early Stopping:---", min_value=0,max_value=20, value=5,step=1)
    complete_btn = False
    if Param_select:
        st.sidebar.write("#### **IF YOU ARE SATISFIED WITH THE CURRENT COMBINATION, GET THE VALUES BY SAYING---**")
        complete_btn = st.sidebar.button("I'll take this")
    return(loss, learning_rate, subsample, n_estimators,max_depth, min_samples_leaf,n_iter, complete_btn)


def GB_params_R(): #user input parameters for Gradient Boost Regressor Model
    loss =param_expander.radio('LOSS function:---', options = ['ls','lad','huber','quantile'], index=0) 
    learning_rate = param_expander.selectbox("Choose Learning Rate:---", options=[0.001,0.005,0.007,0.01,0.03,0.05,0.1,0.3,0.5,0.8,1], index=5)
    subsample =param_expander.slider("Choose Fraction of Sample for Fitting:---", min_value=0.1, max_value=1.0, value=0.5,step=0.1)
    n_estimators = param_expander.number_input("Choose number of Estimators:---", min_value=5, max_value=1000, value=100, step=100)
    max_depth = param_expander.slider("Maximum Depth of the trees should be:---", min_value=1, max_value=40, value=3,step=1)
    min_samples_leaf = param_expander.slider("Min no. of sample at the Leaf Node:---", min_value=1, max_value=None, value=1,step=1)
    n_iter = param_expander.number_input("Type/Select no. of iteration before Early Stopping:---", min_value=0,max_value=20, value=5,step=1)
    complete_btn = False
    if Param_select:
        st.sidebar.write("#### **IF YOU ARE SATISFIED WITH THE CURRENT COMBINATION, GET THE VALUES BY SAYING---**")
        complete_btn = st.sidebar.button("I'll take this")
    return(loss, learning_rate, subsample, n_estimators, max_depth, min_samples_leaf,n_iter, complete_btn)


def XGB_params(): #user input parameters for XGboost Classifier and Regressor Model
    eta = param_expander.selectbox("Choose eta value(Learning Rate):---", options=[0.001,0.005,0.01,0.03,0.05,0.1,0.3,0.5,0.8,1], index=6)
    gamma = param_expander.number_input("Type/Select value of gamma:---", min_value=0, max_value=None, value=0,step=1)
    subsample = param_expander.slider("Choose Fraction of Sample for Fitting:---", min_value=0.1, max_value=1.0, value=1.0,step=0.1)
    tree_method = param_expander.radio('TREE Method:---', options = ['auto','exact','approx','hist'], index=0)     
    n_estimators = param_expander.number_input("Choose number of Estimators:---", min_value=10, max_value=1000, value=100, step=100)
    max_depth = param_expander.slider("Maximum Depth of the trees should be:---", min_value=1, max_value=32, value=6,step=1)
    colsample_bytree = param_expander.slider("Choose Fraction of Columns for each tree:---",  min_value=0.1, max_value=1.0, value=1.0,step=0.1)
    reg_lambda = param_expander.slider("Type/Select lambda value:---", min_value=0, max_value=None, value=1, step=1)
    complete_btn = False
    if Param_select:
        st.sidebar.write("#### **IF YOU ARE SATISFIED WITH THE CURRENT COMBINATION, GET THE VALUES BY SAYING---**")
        complete_btn = st.sidebar.button("I'll take this")
    return(eta, gamma, subsample, tree_method, n_estimators, max_depth, colsample_bytree,reg_lambda, complete_btn)


def data_prepare(DF): #To Display a sample of the Selected Data to the User and also spliting it into INPUT and TARGET 
    col_no = DF.shape[1]
    X=DF.iloc[:,1:col_no-1]
    y=DF.iloc[:,-1]
    data_index=DF.iloc[:,0]
    DF=DF.iloc[:,1:]
    #Standard Scaling the input data
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    return(DF,X,y,data_index)


def data_show(DF):
    with dataset:
        st.write("<=======================================================================>")
        st.markdown("* ### Here are 5 sample rows of your selected dataframe, the last column being the **Target** Column:---")
        st.dataframe(DF.sample(5))
        st.write("<=======================================================================>")


# Function to build Model       
#GB Model:--'loss':loss,'learning_rate':learning_rate,'subsample':subsample,'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,'n_iter_no_change':n_iter
def GB_Clf(random_state,param):
    clf = GradientBoostingClassifier(random_state=random_state, loss=param['loss'], learning_rate=param['learning_rate'], subsample=param['subsample'], n_estimators=param['n_estimators'], max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], n_iter_no_change=param['n_iter_no_change'])
    return(clf)
def GB_Reg(random_state,param):
    reg = GradientBoostingRegressor(random_state=random_state, loss=param['loss'], learning_rate=param['learning_rate'], subsample=param['subsample'], n_estimators=param['n_estimators'], max_depth=param['max_depth'], min_samples_leaf=param['min_samples_leaf'], n_iter_no_change=param['n_iter_no_change'])
    return(reg)


#XGB Model:--'eta':eta,'gamma':gamma,'subsample':subsample,'n_estimators':n_estimators,'max_depth':max_depth,'max_leaves':max_leaves,'reg_lambda':reg_lambda
def XGB_Clf(random_state,param):
    clf = XGBClassifier(random_state=random_state, eta=param['eta'], gamma=param['gamma'], subsample=param['subsample'], n_estimators=param['n_estimators'], tree_method=param['tree_method'], max_depth=param['max_depth'], colsample_bytree=param['colsample_bytree'], reg_lambda=param['reg_lambda'])
    return(clf)
def XGB_Reg(random_state,param):
    reg = XGBRegressor(random_state=random_state, eta=param['eta'], gamma=param['gamma'], subsample=param['subsample'], n_estimators=param['n_estimators'], tree_method=param['tree_method'], max_depth=param['max_depth'], colsample_bytree=param['colsample_bytree'], reg_lambda=param['reg_lambda'])
    return(reg)


def data_shape_display(X_train,X_test,y_train,y_test):
    with trXshape:
        st.markdown("### **X_Train shape:{}**".format(X_train.shape))
        st.write("")
    with tXshape:
        st.markdown("### **X_Test shape:{}**".format(X_test.shape))
        st.write("")
    with trYshape:
        st.markdown("### **y_Train shape:{}**".format(y_train.shape))
        st.write("")
    with tYshape:
        st.markdown("### **y_Test shape:{}**".format(y_test.shape))
        st.write("")


   
# Function to Fit Model
def model_fit(model, X_train, X_test, y_train):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    return(y_pred)


@st.cache
# Evaluation of the Classification Model
def clf_Evaluate(y_test, y_pred):
    score = round(accuracy_score(y_test,y_pred),3)
    matrix = confusion_matrix(y_test,y_pred)
    f1 = round(f1_score(y_test,y_pred),3)
    recall = round(recall_score(y_test,y_pred),3)
    precision = round(precision_score(y_test,y_pred),3)
    return(score, matrix, f1, recall, precision)

def reg_Evaluate(y_test, y_pred):
    score = round(r2_score(y_test,y_pred),3)
    mse = round(mean_squared_error(y_test,y_pred),5)
    mae = round(mean_absolute_error(y_test,y_pred),8)
    return(score, mae, mse)


#Building the model according to the tunned parameters     

def model_perform(model_name,Score_name,score,factor1,ev1,factor2,ev2,factor3,ev3):
    with outcome:
        st.markdown("# ----Model")
        st.write("~~~~~~~~~")
        st.info("**<Model Performance on Test Data>**")
        st.success("**{} Score:  {}**".format(Score_name,score))
        st.write("~~~~~~~~~")
            
    with divider:
        st.write("# ")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("|")
        st.write("|")
        st.write(".")
        st.write("|")
        st.write("|")

    with evaluate:
        st.markdown("# Performance----")
        st.write("~~~~~~~~~")
        st.info("**<Evaluation of the Model>**")
        if model_name=='Regression':
            st.success("** {} : {}**".format(factor1,ev1))
            st.success("** {} : {}**".format(factor2,ev2))
            st.write("~~~~~~~~~")
        else:
            st.success("** {} : {}**".format(factor2,ev2))
            st.success("** {} : {}**".format(factor3,ev3))
            st.write("~~~~~~~~~")
            st.write(factor1, ev1)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#-------------------------M---A---I---N----------F---U---N---C---T---I---O---N-------------------------------------------

# Reading corresponding Dataset:---
if df == 'Bank_Churn':
    DF = get_df('Bank_Churn.csv')
    DF,X,y,data_index = data_prepare(DF)
    data_show(DF)
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_split_size,random_state=3)
    
if df == 'Social_Network_Ads':
    DF = get_df('Social_Network_Ads.csv')
    DF,X,y,data_index = data_prepare(DF)
    data_show(DF)
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_split_size,random_state=3)
      
if df == 'House_Price':
    DF = get_df('House_Price.csv')
    DF,X,y,data_index = data_prepare(DF)
    data_show(DF)
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_split_size,random_state=3)

#-------------------------------------------------------------------------------------------------------------------------

if model == 'Gradient Boosting' and df == 'House_Price':
    with param_expander:
        loss, learning_rate, subsample, n_estimators,max_depth,min_samples_leaf,n_iter, complete_btn = GB_params_R()
        gb_parameter = {'loss':loss,'learning_rate':learning_rate,'subsample':subsample,'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,'n_iter_no_change':n_iter}   
    if complete_btn:
        dataset.warning("**---Your Selected Combination of Parameters---**")
        dataset.write(gb_parameter)

if model == 'Gradient Boosting' and (df == 'Bank_Churn' or df == 'Social_Network_Ads'):
    with param_expander:
        loss, learning_rate, subsample, n_estimators,max_depth,min_samples_leaf,n_iter, complete_btn = GB_params_C()
        gb_parameter = {'loss':loss,'learning_rate':learning_rate,'subsample':subsample,'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,'n_iter_no_change':n_iter}
    if complete_btn:
        dataset.warning("**---Your Selected Combination of Parameters---**")
        dataset.write(gb_parameter)

if model == 'XGboost':
    with param_expander:
        eta, gamma, subsample, tree_method, n_estimators, max_depth, colsample_bytree,reg_lambda, complete_btn = XGB_params()
        xgb_parameter = {'eta':eta,'gamma':gamma,'subsample':subsample,'n_estimators':n_estimators, 'tree_method':tree_method, 'max_depth':max_depth,'colsample_bytree':colsample_bytree,'reg_lambda':reg_lambda}
    if complete_btn:
        dataset.warning("**---Your Selected Combination of Parameters---**")
        dataset.write(xgb_parameter)

#-------------------------------------------------------------------------------------------------------------------------

# Model Building:---
if df == 'Bank_Churn' or df == 'Social_Network_Ads':
    model_name='Classification'
    if model == 'Gradient Boosting':
        clf = GB_Clf(random_state,gb_parameter)
        y_pred = model_fit(clf,X_train,X_test,y_train)
        accuracy, cn_matrix, F1_score, Recall_score, Precision_score = clf_Evaluate(y_test,y_pred)
        if Param_select==False:
            data_shape_display(X_train,X_test,y_train,y_test)
            st.subheader("**Above are the default Performances of the model. Tune the Parameters to improve the result---**")
        Score_name="Accuracy"
        Score=accuracy
        factor1="Confusion Matrix"
        ev1=cn_matrix
        factor2="Recall Score"
        ev2=Recall_score
        factor3="Precision Score"
        ev3=Precision_score
        if complete_btn==False:
            model_perform(model_name,Score_name,Score,factor1,ev1,factor2,ev2,factor3,ev3)               
    else:
        clf = XGB_Clf(random_state,xgb_parameter)
        y_pred = model_fit(clf,X_train,X_test,y_train)
        accuracy, cn_matrix, F1_score, Recall_score, Precision_score = clf_Evaluate(y_test,y_pred)      
        if Param_select==False:
            data_shape_display(X_train,X_test,y_train,y_test)
            st.markdown(" ### **Above is the default Performance of the model. Tune the Parameters to improve the result---**")
        Score_name="Accuracy"
        Score=accuracy
        factor1="Confusion Matrix" 
        ev1=cn_matrix
        factor2="Recall Score"
        ev2=Recall_score
        factor3="Precision Score"
        ev3=Precision_score
        if complete_btn==False:
            model_perform(model_name,Score_name,Score,factor1,ev1,factor2,ev2,factor3,ev3)               


elif df =='House_Price': 
    model_name='Regression'
    if model == 'Gradient Boosting':
        reg = GB_Reg(random_state,gb_parameter)
        y_pred = model_fit(reg,X_train,X_test,y_train)
        r2, mae, mse= reg_Evaluate(y_test, y_pred)      
        if Param_select==False:
            data_shape_display(X_train,X_test,y_train,y_test)
            st.markdown(" ### **Above is the default Performance of the model. Tune the Parameters to improve the result---**")
        Score_name="R-Squared(R2)"
        Score=r2
        factor1="Mean Absolute Error"
        ev1=mae 
        factor2="Mean Squared Error"
        ev2=mse    
        if complete_btn==False:
            model_perform(model_name,Score_name,Score,factor1,ev1,factor2,ev2,factor2,ev2)               
    else:
        reg = XGB_Reg(random_state,xgb_parameter)
        y_pred = model_fit(reg,X_train,X_test,y_train)
        r2, mae, mse= reg_Evaluate(y_test, y_pred)      
        if Param_select==False:
            data_shape_display(X_train,X_test,y_train,y_test)
            st.markdown(" ### **Above is the default Performance of the model. Tune the Parameters to improve the result---**")
        Score_name="R-Squared(R2)"
        Score=r2
        factor1="Mean Absolute Error"
        ev1=mae 
        factor2="Mean Squared Error"
        ev2=mse
        if complete_btn==False:
            model_perform(model_name,Score_name,Score,factor1,ev1,factor2,ev2,factor2,ev2)               

else:
    pass      




    
   
    


