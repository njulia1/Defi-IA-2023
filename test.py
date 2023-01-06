import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle

from sklearn.model_selection import train_test_split


def submission_test(model, path_to_test, path_to_features, dict_city, dict_language, dict_brand, dict_group):
    print('Loading test dataset')
    test = pd.read_csv(path_to_test,sep=",",header=0)
    test = test.drop(['order_requests'],axis=1)
    bis = test
    test = test.drop(['index'],axis=1)
    test.head()

    features=pd.read_csv(path_to_features,sep=",",header=0)
    features_test=features.loc[features["hotel_id"].loc[test['hotel_id']]]
    features_test=features_test.reset_index()

    df=pd.concat([features_test,test[['stock','date','language',"mobile"]]],axis=1)
    df.index=df['hotel_id']
    df=df.drop(['hotel_id'],axis=1)
    df=df.drop(['index'],axis=1)
    
    df['city']=df['city'].map(dict_city)
    df['group']=df['group'].map(dict_group)
    df['brand']=df['brand'].map(dict_brand)
    df['language']=df['language'].map(dict_language)

    print('Computing predictions')
    predictions = model.predict(df)

    print('Saving of submissions dataset')
    submission=pd.DataFrame()
    submission['index'] = bis['index']
    submission['price'] = predictions
    submission['price'] = np.exp(submission['price'])
    submission.to_csv('submission.csv', index = False)

if __name__ == '__main__':

    with open(os.path.join('dicts', 'city.pkl'), 'rb') as f1:
        dict_city = pickle.load(f1)

    with open(os.path.join('dicts', 'language.pkl'), 'rb') as f1:
        dict_language = pickle.load(f1)

    with open(os.path.join('dicts', 'brand.pkl'), 'rb') as f1:
        dict_brand = pickle.load(f1)

    with open(os.path.join('dicts', 'group.pkl'), 'rb') as f1:
        dict_group = pickle.load(f1)

    with open(os.path.join('model', 'model.pkl'), "rb") as f1:
        model = pickle.load(f1)

    submission_test(model, os.path.join('data', 'test_set.csv'), os.path.join('data', 'features_hotels.csv'), dict_city, dict_language, dict_brand, dict_group)