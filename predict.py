import os
import sys
import io
import email
import pandas as pd
import scripts.email_features as er
import json
import argparse
import joblib
import warnings

def create_dataframe_from_mails(mails):
    return pd.DataFrame({
        'subject': [m.get('subject') if m.get('subject') is not None else '' for m in mails],
        'body': [er.retrieve_payload_text(m) for m in mails],
        'content_types': [er.retrieve_content_type_list(m) for m in mails]
    }, columns=['content_types', 'subject', 'body'])


def load_and_parse_prediction_mails(stream):
    plain_mails = [plain_mail.encode('ascii', 'ignore') for plain_mail in json.load(stream)]
    mails = map(email.message_from_string, plain_mails)
    return mails

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model saved using joblib")
args = parser.parse_args()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(args.model)

mails = create_dataframe_from_mails(load_and_parse_prediction_mails(sys.stdin))

predictions = model.predict(mails)
for prediction in predictions:
    print prediction