import requests

url = "http://127.0.0.1:8000/ask"

payload = {"question": "Explain the main idea of the document?"}

res = requests.post(url, json=payload)

print(res.json())
