# djagno_chat_bot_api

Trained a chatbot to know information about autism. 
The code can be divided into two parts
* Machine Learnign Part
* Django Api Part

The machine learning part is present in [code_ml](https://github.com/talhaanwarch/djagno_chat_bot_api/tree/main/code_ml) whereas rest of files are django api files.
## Machine Learning part
[intents.json](https://github.com/talhaanwarch/djagno_chat_bot_api/blob/main/code_ml/intents.json) file contain the question and their responses. You can edit it according to your need.  
[code.py](https://github.com/talhaanwarch/djagno_chat_bot_api/blob/main/code_ml/code.py) contains the ML alogrithm in which features are extracted using TF-IDF and then NaiveBayes is used to classify them. Both trained classifier and vectorizers are saved as [models.p](https://github.com/talhaanwarch/djagno_chat_bot_api/blob/main/code_ml/models.p) to used in online web app.

## Django API part
[This tutorial](https://towardsdatascience.com/productionize-a-machine-learning-model-with-a-django-api-c774cb47698c) helped a lot to create django API



# Base API URL:
https://autiskabot.herokuapp.com/api/?question=

Add the question to baseline such as 
https://autiskabot.herokuapp.com/api/?question=What is autism 
It will generate result like this
```
{"result": "Autism is a complex, lifelong developmental disability that typically appears during early 
childhood and can impact a person's social skills, communication, relationships, and self-regulation"}
```

## Three important .txt files
* runtime.txt->> python version installed, some time heroku doesnot support the version, so you can check [here](https://devcenter.heroku.com/articles/python-support)
* requirements.txt->> python packages required
* nltk.txt->> nltk corpus used, herko required this for working

# API flow
![pipe line](https://github.com/talhaanwarch/djagno_chat_bot_api/blob/main/flow.jpg)
