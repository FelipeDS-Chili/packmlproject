from datetime import datetime
import os
import joblib
import pandas as pd
import numpy as np
import pytz
from flask import Flask
from flask import request
from flask_cors import CORS
from termcolor import colored


app = Flask(__name__)

CORS(app)

PATH_TO_MODEL = os.path.join("model.joblib")


@app.route('/')
def index():
    return 'OK'


if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8080, debug=True)
