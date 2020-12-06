from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse
from rest_framework.views import APIView
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json 
import os
from django.conf import settings
import pandas as pd
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))  

lemmatizer = WordNetLemmatizer()
class call_model(APIView):
    def get(self,request):
        if request.method == 'GET':
            # get sound from request
            try:
                  question = request.GET.get('question')
                  question=word_tokenize(question.lower())
                  question=[i for i in question if i not in stop_words]
                  question=" ".join([lemmatizer.lemmatize(i) for i in question])
                  vector = PredictorConfig.vectorizer.transform([question])
                  # predict based on vector
                  prediction = PredictorConfig.clf.predict(vector.toarray())
                  file_path= os.path.join(settings.MODELS, 'intents.json')
                  with open(file_path,encoding="utf8") as file:
                  	data = json.load(file)
                  df=pd.DataFrame(data['intents'])
                  df['tag']=df['tag'].str.lower()

                  out=df['responses'][df['tag']==prediction[0]]
                  # build response
                  print(out.to_list())
                  response = {'result': out.to_list()[0][0]}
                  # return response
                  return JsonResponse(response)
            except :
                  return JsonResponse({'result':"https://autismchatbot.herokuapp.com/?question=PASS_HERE"})