# # Predict Test

import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 24, # modify to get 'TRUE' or 'FALSE' 
    'monthlycharges': 29.85, # modify to get 'TRUE' or 'FALSE'
    'totalcharges': (29.85 * 24) # modify to get 'TRUE' or 'FALSE'
}

response = requests.post(url,json=customer).json()
print(response)

if response['churn'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)


