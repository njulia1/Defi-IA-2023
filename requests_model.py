import urllib.parse
import requests
import os
import pandas as pd

domain = "51.91.251.0"
port = 3000
host = f"http://{domain}:{port}"
path = lambda x: urllib.parse.urljoin(host, x)
user_id = '88be3640-82ae-46c6-ac59-92a9679d453d'

avatar_info = requests.get(path(f"avatars/{user_id}"))
avatar_names = []

for avatar in avatar_info.json():
    avatar_names.append(avatar['name'])

cities=['amsterdam','madrid','paris','rome','sofia','vienna']
dates=[25,20,15,10]
languages=['austrian','cypriot','dutch','estonian','finnish','french','german','greek','hungarian','irish','italian','lithuanian','luxembourgish', 'maltese','polish','slovakian','spanish','swedish']
mobiles=[0,1]
avatar = 'laulau' #nom de l'avatar à tester

if not os.path.exists('requests'):
    os.mkdir('requests')


#listes contenant toutes les combinaisons possibles pour un avatar
combination=[]
for count1,date in enumerate(dates):
    for count2,city in enumerate(cities):
        for count3,language in enumerate(languages):
            for count4,mobile in enumerate(mobiles):
                combination.append((city,date,language,mobile))
                print(combination)
                params = {
                "avatar_name": avatar,
                "language": language,
                "city": city,
                "date": date,
                "mobile": mobile,}
                print(params)

                ## REQUÊTE ##
                r = requests.get(path(f"pricing/{user_id}"), params=params)
                r.json()
                pricing_requests = []

                request = [r]
                for r in request:
                    pricing_requests.append(
                        pd.DataFrame(r.json()['prices']).assign(**r.json()['request'])
                    )

                pricing_requests = pd.concat(pricing_requests)
                print(pricing_requests)
                pricing_requests.to_csv(os.path.join('requests', str(avatar) +'_'+str(city)+ '_' + str(date) + '_' + str(language) + '_' + str(mobile)+'.csv'))