import pandas as pd
import requests
url = 'https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv'
res = requests.get(url, allow_redirects=True)
with open('titanic.csv','wb') as file:
    file.write(res.content)
sales_team = pd.read_csv('titanic.csv')