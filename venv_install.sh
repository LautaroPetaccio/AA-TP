#!/bin/bash
virtualenv --no-site-packages venv
source venv/bin/activate
pip install nltk --upgrade
pip install joblib --upgrade
deactivate
