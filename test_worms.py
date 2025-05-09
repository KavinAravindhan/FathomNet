import requests
import urllib.parse

term = "Funiculina"
url = f"https://fathomnet.org/api/worms/search?term={urllib.parse.quote_plus(term)}"
resp = requests.get(url)
print("Status:", resp.status_code)
print(resp.json())
