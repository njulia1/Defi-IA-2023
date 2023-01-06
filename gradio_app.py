import pickle
import gradio
import os
import pandas as pd
import numpy as np

def predict_price(city, language, date, brand, group, parking, pool, mobile, children_policy):
    
    hotel_features = pd.DataFrame(columns = ["group","brand","city", "parking", "pool","children_policy","stock","date","language","mobile"])

    #en réalité l'utilisateur n'a pas la possibilité de choisir le stock de chambre dns l'hotel, on le fixe donc à 20 ici.
    stock = 20 

    # 1 = l'hotel à 1 parking, 0 sinon
    if parking: 
        park = 1
    else:
        park = 0

    # 1 = l'hotel a une piscine, 0 sinon
    if pool:
        pool = 1
    else:
        pool = 0

    # 1 si la recherche a été faite sur le tel 0 si sur l'ordi
    if mobile:
        mobile = 1
    else:
        mobile = 0

    # le client souhaite un hôtel sans enfant ou non.
    if children_policy == "no_policy":
        children_policy = 0
    elif children_policy == "no_less_12":
        children_policy = 1
    else:
        children_policy = 2

    with open(os.path.join('dicts','city.pkl'), 'rb') as f1:
        dict_city = pickle.load(f1)

    with open(os.path.join('dicts','language.pkl'), 'rb') as f1:
        dict_language = pickle.load(f1)

    with open(os.path.join('dicts','brand.pkl'), 'rb') as f1:
        dict_brand = pickle.load(f1)

    with open(os.path.join('dicts','group.pkl'), 'rb') as f1:
        dict_group = pickle.load(f1)

    with open(os.path.join('model','model.pkl'), "rb") as f1:
        model = pickle.load(f1)

    # on remplit le tableau avec les encoding définis précédemment
    ville = dict_city[city]
    langue = dict_language[language]
    brands = dict_brand[brand]
    groupe = dict_group[group]

    hotel_features = hotel_features.append({
                "group" : groupe,
                "brand" : brands,
                "city" : ville, 
                "parking" : park,
                "pool" : pool,
                "children_policy" : children_policy,
                "stock" : stock,
                "date" : date,
                "language" : langue,
                "mobile" : mobile
                }
                , ignore_index=True)


    estimated_price = np.exp(model.predict(hotel_features))

    text = "You want to go to " + city + " in " + str(date) + " days. You are " + language + ". You have chosen an hotel" 
    pool = " with a pool and " if pool else " without pool and "
    parking = "with a parking. " if parking else "without parking. "
    brand_group = "You want the brand: " + brand + " and the group: " + group + ". "
    tel = "You want to book on your cellphone. " if mobile else "You want to book on your computer. "
    enfant = "You want a hotel without children." if (children_policy == 1 or children_policy == 2) else "The presence of children does not bother you."
    text = text + pool + parking + brand_group + tel + enfant
    price = "The price is " + str(int(estimated_price))
    print(text)
    return (text,price)


def gradio_interface():
    with open(os.path.join('dicts','city.pkl'), 'rb') as f1:
        dict_city = pickle.load(f1)

    with open(os.path.join('dicts','language.pkl'), 'rb') as f1:
        dict_language = pickle.load(f1)

    with open(os.path.join('dicts','brand.pkl'), 'rb') as f1:
        dict_brand = pickle.load(f1)

    with open(os.path.join('dicts','group.pkl'), 'rb') as f1:
        dict_group = pickle.load(f1)

    with open(os.path.join('model','model.pkl'), "rb") as f1:
        model = pickle.load(f1)


    date = gradio.inputs.Slider(minimum=1, maximum=44, step=1)
    children_policy = gradio.inputs.Radio(choices=["no_policy", "no_less_12", "no_children"])
    city1 = gradio.inputs.Radio(choices=list(dict_city.keys()))
    language1 = gradio.inputs.Radio(choices = list(dict_language.keys()))

    group1 = gradio.inputs.Radio(choices = list(dict_group.keys()))
    brand1 = gradio.inputs.Radio(choices = list(dict_brand.keys()))

    demo = gradio.Interface(
        fn = predict_price,
        inputs = [city1,language1,date,brand1,group1,"checkbox","checkbox","checkbox",children_policy],
        outputs = ["text","text"],
    )
    demo.launch(share=True)
    return demo.launch(share=True)

if __name__ == '__main__':

    with open(os.path.join('dicts','city.pkl'), 'rb') as f1:
        dict_city = pickle.load(f1)

    with open(os.path.join('dicts','language.pkl'), 'rb') as f1:
        dict_language = pickle.load(f1)

    with open(os.path.join('dicts','brand.pkl'), 'rb') as f1:
        dict_brand = pickle.load(f1)

    with open(os.path.join('dicts','group.pkl'), 'rb') as f1:
        dict_group = pickle.load(f1)

    with open(os.path.join('model','model.pkl'), "rb") as f1:
        model = pickle.load(f1)


    date = gradio.inputs.Slider(minimum=1, maximum=44, step=1)
    children_policy = gradio.inputs.Radio(choices=["no_policy", "no_less_12", "no_children"])
    city1 = gradio.inputs.Radio(choices=list(dict_city.keys()))
    language1 = gradio.inputs.Radio(choices = list(dict_language.keys()))

    group1 = gradio.inputs.Radio(choices = list(dict_group.keys()))
    brand1 = gradio.inputs.Radio(choices = list(dict_brand.keys()))

    demo = gradio.Interface(
        fn = predict_price,
        inputs = [city1,language1,date,brand1,group1,"checkbox","checkbox","checkbox",children_policy],
        outputs = ["text","text"],
    )
    demo.launch(share=True)