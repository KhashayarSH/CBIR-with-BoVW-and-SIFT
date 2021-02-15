from flask import Flask
from flask_bootstrap import Bootstrap
from config import Config
import pickle

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


app = Flask(__name__)

app.config.from_object(Config)

bootstap = Bootstrap(app)

cbir_results = load_obj('results_sift500')

from app import routes, errors, forms
