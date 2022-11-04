import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from utils import transform_column_to_onehot, split_data_and_train_model, transform_data_to_predict, score_model, train_with_kFold
from utils import split_column_by_type, label_output_feature, get_confusion_matrix
from utils import get_f1, get_logloss, get_precision, get_recall

st.set_page_config(page_title='Linear Regression')

def resetSessionState():
    st.session_state['trained'] = False 
    st.session_state['score'] = [0, 0]
    return

if __name__=='__main__':
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False
        st.session_state['score'] = [0, 0]
        st.session_state['model'] = None
        st.session_state['cfs_matrix'] = None
        
    st.title("Classify anything with Logistic Regression")
    
    uploaded_file = st.file_uploader("Let's upload your dataset")
    if uploaded_file is not None:
        if (uploaded_file.name[-3:] != 'csv'):
            st.warning("Please up load file csv")
            st.stop()
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df)
            column_labels = df.columns.to_numpy()
            # categorical_column_labels, numerical_column_labels = split_column_by_type(df, column_labels)
            
            st.header("Select output feature")
            st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            selected_output_feature = st.radio("", column_labels, label_visibility='collapsed', on_change=resetSessionState)
            
            st.write(f"Output feature: {selected_output_feature}")
            if (selected_output_feature):
                st.write(f"This feature contains {len(df[selected_output_feature].unique())} classes")
                st.write(f'First 5 classes in this feature: {df[selected_output_feature].unique()[:5]}')

            st.header("Select training features")
            cols = st.columns(4)
            selected_columns = []
            ii = 0
            for (i, column_label) in enumerate(column_labels):
                if (column_label != selected_output_feature):
                    t = cols[(i - ii) % 4].checkbox(column_label, on_change=resetSessionState)
                    if t:
                        selected_columns.append(column_label)
                else:
                    ii = 1
        
            if (len(selected_columns) == 0):
                st.info("Select training feature to continue")
                st.stop()
        
            # Initalize
            y, label_encoder = label_output_feature(df[selected_output_feature].to_numpy())
            X = np.array([])
            dict_encoder = {}
            precision_score = 0
            recall_score = 0
            F1_score = 0
            log_loss = 0
            cfs_matrix = np.zeros((len(label_encoder.classes_), len(label_encoder.classes_)))
            
            X, dict_encoder= transform_column_to_onehot(df=df, selected_columns=selected_columns, dict_encoder=dict_encoder)
            
            st.header("Train test split: ")
            kFold_mode = st.checkbox("K Fold", on_change=resetSessionState)
                    
            if kFold_mode:
                n_splits = int(st.number_input("Enter n split: ", min_value=2, step=1))
                if st.button("Train model!"):
                    st.session_state['trained'] = True
                    model, train_score, test_score = train_with_kFold(X, y, n_splits)
                    st.session_state['score'] = [train_score, test_score]
                    st.session_state['model'] = model
                    cfs_matrix = get_confusion_matrix(model, X, y)
                    st.session_state['cfs_matrix'] = cfs_matrix
            else:
                train_split_ratio = float(st.slider('Select train ratio in range', 0, 90, 80, on_change=resetSessionState))
                if st.button("Train model!"):
                    st.session_state['trained'] = True
                    model, X_train, X_test, y_train, y_test = split_data_and_train_model(X=X, y=y, train_split_ratio=train_split_ratio)
                    st.session_state['model'] = model
                    test_score = score_model(model, X_test, y_test)
                    train_score = score_model(model, X_train, y_train)
                    st.session_state['score'] = [train_score, test_score]
                    cfs_matrix = get_confusion_matrix(model, X, y)
                    st.session_state['cfs_matrix'] = cfs_matrix
                    
            if not st.session_state['trained']:
                st.info("Train model to continue")
                st.stop()
                
            precision_score, recall_score, F1_score = get_precision(st.session_state['model'], X, y), \
                                                        get_recall(st.session_state['model'], X, y), \
                                                        get_f1(st.session_state['model'], X, y)
            
            col1, col2 = st.columns(2)
            fig, ax = plt.subplots()
            if kFold_mode:
                if not st.session_state['trained']:
                    st.session_state['score'] = [[0], [0]]
                r = np.arange(len(st.session_state['score'][0]))
                ax.bar(r - 0.2, st.session_state['score'][0], color ='b', width = 0.4, label='Train')
                ax.bar(r + 0.2, st.session_state['score'][1], color ='g', width = 0.4, label='Test')
                ax.set_xticks(r, r)
                ax.set_xlabel("Fold")
                ax.set_ylabel("MSE")
                ax.set_title("Train test score")
                ax.legend()
                col1.pyplot(fig)
            else:
                ax.bar(["Train", "Test"], st.session_state['score'], color ='maroon', width = 0.4)
                ax.set_xlabel("MSE")
                ax.set_ylabel("Set")
                ax.set_title("Train test score")
                col1.pyplot(fig)                
            col1.write(f"Precision: {precision_score}")
            col1.write(f"Recall: {recall_score}")
            col1.write(f"F1: {F1_score}")
            # col1.write(f"Log loss: {log_loss}")
                
            # fig2 = sns.heatmap(cfs_matrix)
            # col2.pyplot(fig2)  
            df_cfs = pd.DataFrame(st.session_state['cfs_matrix'], columns=label_encoder.classes_, index=label_encoder.classes_)
            col2.write(df_cfs)
                
            st.header("Make Prediction")
            dict_data_to_prediction = {}
            for (index, data_column) in enumerate(selected_columns):
                if (df.dtypes[data_column] == 'object'):
                    unique_val = df[data_column].unique()
                    cat_val = st.selectbox(data_column, (unique_val))
                    cat_val_onehot = dict_encoder[data_column].transform([[cat_val]]).toarray().squeeze()
                    dict_data_to_prediction[data_column] = cat_val_onehot
                else:
                    dict_data_to_prediction[data_column] = st.number_input(data_column)
            
            if st.button("Predict!"):
                data = transform_data_to_predict(selected_columns, dict_data_to_prediction)
                try:
                    model = st.session_state['model'] 
                    prediction = model.predict(data)
                    st.write(f"{selected_output_feature}: {label_encoder.inverse_transform([prediction.item(0)]).item(0)}")
                except Exception as e:
                    st.write(e)
            
        except Exception as e:
            st.write(e)
