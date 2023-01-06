import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder
import pickle
import argparse

def data_loading(path_to_data):
    return pd.read_csv(path_to_data)

def data_preprocessing(path_to_data):
    print("Data loading")
    data = data_loading(path_to_data)
    data = data.drop(columns = 'hotel_id')
    data = data.drop(columns = 'avatar_name')
    data = data.drop(['order_requests'],axis = 1)

    print("Dictionnary saving for Gradio")
    dict_city = data.groupby("city")["price"].mean().to_dict()
    dict_language = data.groupby("language")["price"].mean().to_dict()
    dict_brand = data.groupby("brand")["price"].mean().to_dict()
    dict_group = data.groupby("group")["price"].mean().to_dict()
    
    with open(os.path.join('dicts', 'city.pkl'), 'wb') as f1:
       pickle.dump(dict_city, f1)

    with open(os.path.join('dicts', 'language.pkl'), 'wb') as f1:
        pickle.dump(dict_language, f1)

    with open(os.path.join('dicts', 'brand.pkl'), 'wb') as f1:
        pickle.dump(dict_brand, f1)

    with open(os.path.join('dicts', 'group.pkl'), 'wb') as f1:
       pickle.dump(dict_group, f1)


    print('Data encoding')
    encoder = TargetEncoder()
    data['city'] = encoder.fit_transform(data['city'],data['price'])
    data['language'] = encoder.fit_transform(data['language'],data['price'])
    data['brand'] = encoder.fit_transform(data['brand'],data['price'])
    data['group'] = encoder.fit_transform(data['group'],data['price'])

    print('Splitting dataset into train and test set')
    price = data["price"]
    price = np.log(price)
    X_train, X_test, price_train, price_test = train_test_split(data,price,test_size=0.25,random_state=11)
    X_train = X_train.drop(['price'],axis=1)
    X_test = X_test.drop(['price'],axis=1)

    return X_train, X_test, price_train, price_test

def model_training(X_train, X_test, price_train, price_test, train = 2):
    print("Model training")
    if train == '1':
        print("Random Forest algorithm")
        param=[{"max_depth":[20,21,22]}]
        regrfopt = GridSearchCV(RandomForestRegressor(n_estimators=200), param, cv=10,n_jobs=-1)
        rf_opt = regrfopt.fit(X_train, price_train)
        Ypred_gbreg = rf_opt.predict(X_test)
        print("MSE : ",mean_squared_error(Ypred_gbreg,price_test))
        print("R2 : ",r2_score(price_test,Ypred_gbreg))
        print("Model saving")

        with open(os.path.join('model', 'model.pkl'), 'wb') as f1:
            pickle.dump(rf_opt, f1)
    else:
        print("Boosting algorithm")
        param = [{"max_depth":[20], "learning_rate":[0.1]}]
        reggradboost = GridSearchCV(GradientBoostingRegressor(n_estimators=100), param, cv=10,n_jobs=-1)
        gradboost_Opt = reggradboost.fit(X_train, price_train)
        Ypred_gbreg = gradboost_Opt.predict(X_test)
        print("MSE : ",mean_squared_error(Ypred_gbreg,price_test))
        print("R2 : ",r2_score(price_test,Ypred_gbreg))
        print("Model saving")

        with open(os.path.join('model', 'model.pkl'), 'wb') as f1:
            pickle.dump(gradboost_Opt, f1)

    print("Model saved!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to data')
    parser.add_argument('--model', help='model to use, 1 for random forest, 2 for boosting')
    args = parser.parse_args()

    path_to_data = args.data
    X_train, X_test, price_train, price_test = data_preprocessing(path_to_data)
    model_training(X_train, X_test, price_train, price_test, args.model)
    print('Model have been saved')