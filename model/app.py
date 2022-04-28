#https://github.com/mnguyenngo/flask-rest-setup/tree/master/sentiment-clf
from distutils.log import debug
import os
import pickle
import io
import flask
from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import json

#model_path = os.environ['MODEL_PATH']
model_path="./artifacts/"
clf_path = 'Classifier.pkl'
vec_path = 'TFIDFVectorizer.pkl'

with open(os.path.join(model_path, clf_path), 'rb') as f:
    clf = pickle.load(f)

with open(os.path.join(model_path, vec_path), 'rb') as f:
    vec = pickle.load(f)


#
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded
    vectoriser = None # Where we keep the vectoriser when it's loaded
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            with open(os.path.join(model_path, clf_path), 'rb') as f:
                cls.model = pickle.load(f)
        return cls.model
    
    @classmethod
    def get_vectoriser(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.vectoriser is None:
            with open(os.path.join(model_path, vec_path), 'rb') as f:
                cls.vectoriser = pickle.load(f)
        return cls.vectoriser

    @classmethod
    def predict(cls, x):        
        clf = cls.get_model()
        vec = cls.get_vectoriser()
        X_transformed = vec.transform(np.array([x]))
        return clf.predict(X_transformed)


# The flask app for serving predictions
app = Flask(__name__)

@app.route('/', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='Service Is Up And Running', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = request.json   
        phrase = data['Phrase']
        
    else:
        return flask.Response(response='This predictor only supports json data', status=415, mimetype='text/plain')
   
    # Do the prediction
    predictions = ScoringService.predict(phrase)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({'results': predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)