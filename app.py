# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:48:25 2023

@author: manaser
"""

import pandas as pd
import streamlit as st
import sksurv # pip install scikit-survival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder 
import numpy as np
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import psutil
#import tempfile
#import os

# Define a function to get the current CPU and memory usage of the system
def get_system_usage():
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory().percent
    return cpu_percent, mem_percent

# Define a function to check if the app can serve a new user based on the current resource usage
def can_serve_user():
    cpu_percent, mem_percent = get_system_usage()
    # Check if the current CPU and memory usage are below the threshold
    if cpu_percent < 80 and mem_percent < 80:
        return True
    else:
        return False

#st.set_page_config(page_title="Survival Analysis", page_icon=":guardsman:", layout="wide")
st.set_page_config(page_title="Survival Analysis", page_icon=":guardsman:")

st.markdown("""
<style>
body {
    background-color: #1E1E1E;
    color: white;
}
</style>
    """, unsafe_allow_html=True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
#Main Streamlit App Code
def main():
    # Check if the app can serve a new user
    if can_serve_user():
        #st.write("Welcome to my app!")           
        # Set page layout
        st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)
        
        #Set Title page
        st.title('Survival Analysis')
        
        if 'available_data' not in st.session_state:
            st.session_state['available_data'] = False
        
        # Load data
        data_file = st.file_uploader('Upload data file', type=['xlsx', 'csv'])
        

        if data_file is not None:
            if '.xlsx' in data_file.name or 'csv' in data_file.name:
                try:
                    if '.xlsx' in data_file.name:
                        data = pd.read_excel(data_file, engine='openpyxl')
                        st.session_state['available_data'] = True
                    elif '.csv' in data_file.name:
                        data = pd.read_csv(data_file)
                        st.session_state['available_data'] = True
                    #st.write(data)
                except Exception as e:
                    st.write(f"Error: {e}")
                    st.session_state['available_data'] = False
        else:
            st.write('Invalid file format. Please upload an Excel file (.xlsx) or a CSV file (.csv).')
            st.session_state['available_data'] = False
   
        if st.session_state['available_data']:

            # Set default column names
            censoring_col = None
            duration_col = None
            included_cols = []
            
            # Set sidebar options
            st.sidebar.header("Survival Analysis Inputs")
            censoring_col = st.sidebar.selectbox("Select censoring column (e.g. event)",
                                                 [""] + list(data.columns))
            duration_col = st.sidebar.selectbox("Select duration column (e.g. time to event)",
                                                [""] + list(data.columns))
        
            # Exclude censoring_col and duration_col from included_cols until they are selected
            if censoring_col is not None and duration_col is not None:
                included_cols = [col for col in data.columns if col not in [censoring_col, duration_col]]
        
            # Show original data
            st.header("Original Data")
            st.write(data)

            # Filter data based on selected columns
            if censoring_col and duration_col:
                
                # Run Kaplan-Meier analysis
                kmf = KaplanMeierFitter()
                kmf.fit(data[duration_col], data[censoring_col], label='Original Data')
        
                # Generate survival curve
                fig, ax = plt.subplots()
                st.subheader("Survival Curve")
                st.write("The survival curve below shows the probability of survival over time - entire data.")
                kmf.plot_survival_function(at_risk_counts=True, ax=ax, show_censors=True)

                plt.title('Kaplan-Meier Curve - Original Data')
                # plt.xlabel('Time')
                plt.ylabel('Survival Probability')
                #st.line_chart(kmf.survival_function_)
                st.write(fig)
        
                # # Generate median survival time
                median_survival_time = kmf.median_survival_time_
                st.write(f"Median survival time: {median_survival_time:.2f} {duration_col}")
        
                # Generate survival rate at specific time points
                st.subheader("Survival Rate")
                st.write("The table below shows the survival rate at specific time points.")
        
                time_points = st.slider("Select time points to show survival rate",
                                min_value=float(data[duration_col].min()),
                                max_value=float(data[duration_col].max()),
                                step=1.0,
                                value=[float(data[duration_col].min()), float(data[duration_col].max())])
        
                # Find nearest neighbors for each time point in time_points
                survival_rates = []
                for t in time_points:
                    nearest_time = kmf.survival_function_.index.to_series().sub(t).abs().idxmin()
                    survival_rate = kmf.survival_function_.loc[nearest_time]
                    survival_rates.append(survival_rate)
                
                # Display survival rates in a table
                results_df = pd.DataFrame({"Time Point": time_points, "Survival Rate": survival_rates})
                st.write(results_df)
            
                # Survival Analysis
                st.header("Survival Analysis")
                # Show selected columns
                included_cols = st.multiselect("Select the columns to include in the analysis", list(included_cols), default=[])
                
                if included_cols:
                    data = data[[censoring_col, duration_col] + included_cols]
                    data = data.fillna(value=np.nan)
                    data = data.dropna(subset=[censoring_col, duration_col] + included_cols)
            
                    # Show filtered data
                    st.subheader("Filtered Data")
                    st.write(data)
                    
                    # One hot encoding for categorical columns
                    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
                    data_onehot = data[[censoring_col, duration_col] + included_cols]
                    
                    st.write('data_onehot before applying transform:', data_onehot)
                    
                    for variable in cat_cols:
                        data_onehot[variable] = data_onehot[variable].astype('category') # changes object dtype to category dtype so it works with OneHot function 
                    data_onehot = OneHotEncoder().fit_transform(data_onehot)
                                   
                    # Data split
                    split_percent = st.sidebar.slider('percentage of train/test:', 0, 100, 80)
                    # Bootstrapping
                    bootstrapping = st.sidebar.radio('Do you want to implement bootstrapping?', ('Yes', 'No'), index=1)
                    if bootstrapping == 'Yes':
                        num_samples = st.sidebar.slider('Bootstrapping number of samples:', 1, 1000, 100)
                        
                    #copyright
                    st.sidebar.text('Mohamed Naser ©2023')
                                   
                    st.subheader('Cox proportional hazards model results:')
                    
                    X = data_onehot.drop([censoring_col, duration_col], axis=1)
                    y = sksurv.util.Surv.from_dataframe(censoring_col, duration_col, data_onehot) # get y variable in usable format (events, time)
                    
                    st.write('X:', X)
                    st.write('y:', y)
                    st.write('data_onehot:', data_onehot)
                    
                    try:
                        estimator = CoxPHSurvivalAnalysis()
                        estimator.fit(X, y)
                        score = estimator.score(X, y)
                        st.write(pd.Series(estimator.coef_, index=X.columns))
                        st.write('Entire dataset C-index = ', score)
                    except Exception as e:
                        st.write(f"Error: {e}")    
                                                    
                    st.subheader("Survival Curve")
                    st.write("The survival curve below shows the probability of survival over time - filtered data.")

                    fig, ax = plt.subplots()
                    kmf = KaplanMeierFitter()
                    kmf.fit(data_onehot[duration_col], data_onehot[censoring_col], label='Filtered Data')
                    kmf.plot_survival_function(at_risk_counts=True, ax=ax, show_censors=True)
                    plt.title('Kaplan-Meier Curve - Filtered Data')
                    plt.ylabel('Survival Probability')

                    st.write(fig)

                    if bootstrapping == 'Yes':
                        c_index_list = []
                        try:
                            for n in range(num_samples):
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percent/100, 
                                                                            random_state=n, shuffle=True, stratify = y[censoring_col])
                                estimator = CoxPHSurvivalAnalysis()
                                estimator.fit(X_train, y_train)
                                score = estimator.score(X_test, y_test)
                                c_index_list.append(score)
                                
                            c_index_mean = np.mean(c_index_list)
                            c_index_sd = np.std(c_index_list)
                            
                            #st.write("test for normality", shapiro(c_index_list)) # test for normality 
                            st.subheader('CoxPHSurvivalAnalysis: Bootstrap')
                            st.write('Number of sample:', num_samples, 'Train/Test split:', split_percent/100)
                            st.write("bootstrap mean score = ", c_index_mean, "bootstrap standard deviation = ", c_index_sd)
                        except Exception as e:
                            st.write(f"Error: {e}")
                            st.write('Try to choose a different train/test percentage')  
                    
                    else:
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percent/100, 
                                                                    random_state=0, shuffle=True, stratify = y[censoring_col])
                            
                            estimator = CoxPHSurvivalAnalysis()
                            estimator.fit(X_train, y_train)
                            train_score = estimator.score(X_train, y_train)
                            test_score = estimator.score(X_test, y_test)

                            st.subheader('CoxPHSurvivalAnalysis')
                            st.write('Train/Test Split:', split_percent/100)
                            st.write('Train C-index:', train_score)
                            st.write('Test C-index:', test_score)

                        except Exception as e:
                            st.subheader('CoxPHSurvivalAnalysis')
                            st.write(f"Test set is empty: {e}")
                            st.write('Using the entire data set instead!')
  
                            estimator = CoxPHSurvivalAnalysis()
                            estimator.fit(X, y)
                            score = estimator.score(X, y)
                            st.write('Entire data C-index:', score)

    else:
        st.write("Sorry, the app is currently overloaded. Please try again later.")
        
if __name__ == "__main__":
    main()