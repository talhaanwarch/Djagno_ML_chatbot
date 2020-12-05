from django.apps import AppConfig

from django.apps import AppConfig
from django.conf import settings
import os
import pickle
import os
class PredictorConfig(AppConfig):
    # create path to models
    model_path = os.path.join(settings.MODELS, 'models.p')
 
    # load models into separate variables
    # these will be accessible via this class
    with open(model_path, 'rb') as pickled:
	    data = pickle.load(pickled)
	    clf = data['classifier']
	    vectorizer = data['vectorizer']
# class SubAppConfig(AppConfig):
#     name = 'sub_app'
