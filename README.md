import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def custom_tokenizer(text):
    # split the text and value using regular expression
    import re
    pattern = re.compile(r'[a-zA-Z]+\d+')
    text_and_value = pattern.findall(text)
    return text_and_value

# Load TF-IDF and classifier models
with open('pickle_letest/payer_name.pkl', 'rb') as file:
    tfidf_payer_name = pickle.load(file)
    
with open('pickle_letest/payer_vpa.pkl', 'rb') as file:
    tfidf_payer_vpa = pickle.load(file)

with open('pickle_letest/payee_account_type.pkl', 'rb') as file:
    tfidf_payee_account_type = pickle.load(file)

with open('pickle_letest/payee_name.pkl', 'rb') as file:
    tfidf_payee_name = pickle.load(file)

with open('pickle_letest/payee_vpa.pkl', 'rb') as file:
    tfidf_payee_vpa = pickle.load(file)

with open('pickle_letest/payer_account_type.pkl', 'rb') as file:
    tfidf_payer_account_type = pickle.load(file)

with open('pickle_letest/classifier1.pkl', 'rb') as file:
    classifier_cat1 = pickle.load(file)

with open('pickle_letest/classifier2.pkl', 'rb') as file:
    classifier_cat2 = pickle.load(file)

def get_predictions(input_data):
    payer_name = input_data.get('payer_name')
    payer_vpa = input_data.get('payer_vpa')
    payee_account_type = input_data.get('payee_account_type')
    payee_name = input_data.get('payee_name')
    payee_vpa = input_data.get('payee_vpa')
    payer_account_type = input_data.get('payer_account_type')

    payer_name_tfidf = tfidf_payer_name.transform([payer_name])
    payer_vpa_tfidf = tfidf_payer_vpa.transform([payer_vpa])
    payee_account_type_tfidf = tfidf_payee_account_type.transform([payee_account_type])
    payee_name_tfidf = tfidf_payee_name.transform([payee_name])
    payee_vpa_tfidf = tfidf_payee_vpa.transform([payee_vpa])
    payer_account_type_tfidf = tfidf_payer_account_type.transform([payer_account_type])

    input_tfidf = pd.concat([pd.DataFrame(payer_name_tfidf.toarray()),
                             pd.DataFrame(payer_vpa_tfidf.toarray()),
                             pd.DataFrame(payee_account_type_tfidf.toarray()),
                             pd.DataFrame(payee_name_tfidf.toarray()),
                             pd.DataFrame(payee_vpa_tfidf.toarray()),
                             pd.DataFrame(payer_account_type_tfidf.toarray())], axis=1)

    # Predictions for cat1 and cat2
    predictions_cat1 = classifier_cat1.predict(input_tfidf)
    predictions_cat2 = classifier_cat2.predict(input_tfidf)

    predictions_dict = {
        'input_data': input_data,
        'cat1_prediction': predictions_cat1[0],
        'cat2_prediction': predictions_cat2[0]
    }

    return predictions_dict

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    input_data_sets = data.get('input_data', [])

    predictions_list = []
    for input_data in input_data_sets:
        predictions = get_predictions(input_data)
        predictions_list.append(predictions)

    return jsonify(predictions_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
