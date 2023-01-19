'''
An KNN Classification algorithm that predicts a user's disease based off their selected symptoms

improve last part of first function to make it go faster
fix spacing in sym_lst and dataframe
'nan' is in sym_lst
'''

import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

K = 9


def get_prediction_df(initial_df):
    """
    Improves the initial dataframe for prediction purposes

    ":param initial_df: a dataframe created from raw data
    :return: an improved dataframe
    """
    # Create a list containing the unique symptoms in the data
    full_sym_lst = [list(initial_df[col]) for col in initial_df.columns[1:]]
    sym_lst = list(Counter([str(sym).strip() for lst in full_sym_lst for sym in lst]).keys())
    sym_lst.remove('nan')

    tmp = pd.melt(initial_df.reset_index(), id_vars=['index'], value_vars=list(initial_df.columns[1:]))
    tmp['add1'] = 1

    dis_sym_df = pd.pivot_table(tmp,
                                values='add1',
                                index='index',
                                columns='value')
    dis_sym_df.insert(0, 'Disease', initial_df['Disease'])
    dis_sym_df = dis_sym_df.fillna(0)

    dis_sym_df.columns = [col.strip() for col in dis_sym_df.columns]

    return dis_sym_df, sym_lst


def predict(model, X_train, X_test, y_train, y_test):
    mdl = model.fit(X_train, y_train)
    prediction = mdl.predict(X_test)

    score = metrics.accuracy_score(y_test, prediction)

    return mdl


def predict_disease(initial_df):

    # Manipulate the DataFrame to split
    dis_sym_df, sym_lst = get_prediction_df(initial_df)

    # Divide the DataFrame into targets and features
    y = dis_sym_df['Disease']
    X = dis_sym_df.drop(['Disease'], axis=1)

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True, random_state=42,
                                                        stratify=y)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc_modl = predict(rfc, X_train, X_test, y_train, y_test)

    # Naive Bayes Model
    nb = MultinomialNB()
    nb_modl = predict(nb, X_train, X_test, y_train, y_test)

    # KNN Classifier
    knn = KNeighborsClassifier(K)
    knn_modl = predict(knn, X_train, X_test, y_train, y_test)

    # Logistic Regression Model
    lr = LogisticRegression()
    lr_modl = predict(lr, X_train, X_test, y_train, y_test)

    return rfc_modl, nb_modl, knn_modl, lr_modl


def report_precautions(pred_dis, prec_df):
    return_str = ''
    for idx, row in prec_df.iterrows():
        if row['Disease'] == pred_dis:
            return_str += row['Precaution_1'] + ', ' + row['Precaution_2']
            if str(row['Precaution_3']) != 'nan':
                return_str += ', ' + row['Precaution_3']
            if str(row['Precaution_4']) != 'nan':
                return_str += ', ' + row['Precaution_4']

    return return_str



