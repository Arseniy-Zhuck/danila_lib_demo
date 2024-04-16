import zipfile
import requests
from urllib.parse import urlencode

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/k6Wx0bD98fpswQ'  # Сюда вписываете вашу ссылку

# Получаем загрузочную ссылку
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

# Загружаем файл и сохраняем его
download_response = requests.get(download_url)
# print(download_response.content)

with open('17_model.zip','wb') as f:
    f.write(download_response.content)

with zipfile.ZipFile('17_model.zip', 'r') as zip_ref:
    zip_ref.extractall()