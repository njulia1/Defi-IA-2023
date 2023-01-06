import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from xgboost import cv
from sklearn import model_selection as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy
from time import ctime
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to data')
    parser.add_argument('--data_test', help='path to test data')
    parser.add_argument('--features', help='path to hotel features')
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data_bis = data
    data = data.drop(['hotel_id','avatar_name','order_requests','Unnamed: 0'], axis = 1)
    data_bis = data_bis.drop(['Unnamed: 0'], axis = 1)

    dict_city = data.groupby("city")["price"].mean().to_dict()
    dict_language = data.groupby("language")["price"].mean().to_dict()
    dict_brand = data.groupby("brand")["price"].mean().to_dict()
    dict_group = data.groupby("group")["price"].mean().to_dict()

    encoder = TargetEncoder()
    data['city'] = encoder.fit_transform(data['city'],data['price'])
    data['language'] = encoder.fit_transform(data['language'],data['price'])
    data['brand'] = encoder.fit_transform(data['brand'],data['price'])
    data['group'] = encoder.fit_transform(data['group'],data['price'])

    df_train = data.drop('price', axis = 1)
    df_train['sample'] = 0  

    ## Préparation du jeu de données test
    test=pd.read_csv(args.data_test,sep=",",header=0)
    test=test.drop(['order_requests'],axis=1)
    bis=test

    test=test.drop(['index'],axis=1)
    features=pd.read_csv(args.features,sep=",",header=0)
    features_test=features.loc[features["hotel_id"].loc[test['hotel_id']]]
    features_test=features_test.reset_index()
    df=pd.concat([features_test,test[['stock','date','language',"mobile"]]],axis=1)
    df=df.drop(['hotel_id'],axis=1)
    df=df.drop(['index'],axis=1)
    df['city']=df['city'].map(dict_city)
    df['group']=df['group'].map(dict_group)
    df['brand']=df['brand'].map(dict_brand)
    df['language']=df['language'].map(dict_language)
    df['sample'] = 1

    df_test = df

    all_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    print("loading")

    train = df_train
    test = df_test

    orig_train = train.copy()

    train = pd.concat(( orig_train, test ))
    train.reset_index( inplace = True, drop = True )

    x = train.drop( ['sample'], axis = 1 )
    y = train.sample
    y = train['sample']

    print("cross-validating...")

    n_estimators = 100
    clf = RF( n_estimators = n_estimators, n_jobs = -1 )

    predictions = np.zeros(y.shape)

    cv = CV.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 5678)

    for f, ( train_i, test_i ) in enumerate(cv.split(x,y)):

        print("# fold {}, {}".format( f + 1, ctime()))

        x_train = x.iloc[train_i]
        x_test = x.iloc[test_i]
        y_train = y.iloc[train_i]
        y_test = y.iloc[test_i]
        clf.fit( x_train, y_train )

        p = clf.predict_proba(x_test)[:,1]

        auc = AUC( y_test, p )
        print("# AUC: {:.2%}\n".format( auc ))

        predictions[ test_i ] = p

    train['p'] = predictions

    i = predictions.argsort()
    train_sorted = train.iloc[i]

    train_sorted_train = train_sorted[train_sorted['sample'] == 0]
    tab_final = train_sorted_train[train_sorted_train['p'] != 0]
    tab = data_bis.iloc[tab_final.index]
    tab.to_csv("data/data_improved.csv",index = False)
    print('Dataset improved saved!')