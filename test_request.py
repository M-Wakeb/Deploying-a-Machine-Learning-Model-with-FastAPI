import json
import requests

data = {"age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
        }
url = "https://deploying-a-machine-learning-model-with-tctu.onrender.com"
action = "/predict"
link = url + action
print(link)
response = requests.post(
    link,
    data=json.dumps(data)
)

print(response.status_code)
# print(response.json())
