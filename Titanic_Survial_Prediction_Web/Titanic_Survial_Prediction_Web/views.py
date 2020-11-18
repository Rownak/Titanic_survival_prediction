from django.shortcuts import render
from django.contrib import messages
import csv, io
import pandas as pd
import numpy as np
# our home page view
# def home(request):    
#     return render(request, 'index.html')


# custom method for generating predictions
def getPredictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    import pickle
    model = pickle.load(open("titanic_survival_ml_model.sav", "rb"))
    scaled = pickle.load(open("scaler.sav", "rb"))
    prediction = model.predict(scaled.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))
    
    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"
        

# our result page view
def result(request):
    pclass = int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])
    embC = int(request.GET['embC'])
    embQ = int(request.GET['embQ'])
    embS = int(request.GET['embS'])

    result = getPredictions(pclass, sex, age, sibsp, parch, fare, embC, embQ, embS)

    return render(request, 'result.html', {'result':result})

def home(request):
    # declaring template
    template = "index.html"
    prompt = {
    'order': 'Order of the CSV should be ...........',
    'profiles': 'data of csv should be...............'
          }
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    d_file = csv_file
    print("fileName:",csv_file)

    # let's check if it is a csv file
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')
    data_set = csv_file.read().decode('UTF-8')
    print("data_set:",data_set)
    # setup a stream which is when we loop through each line we are able to handle a data in a stream
    io_string = io.StringIO(data_set)
    print("io_string", io_string)
    dataset = pd.read_csv(io_string)
    print("dataframe: ", dataset.head())

    train_model(dataset)

    # for index, row in dataset.iterrows():
    #     print(row)
    # next(io_string)
    # for column in csv.reader(io_string, delimiter=',', quotechar="|"):
    #     _, created = Profile.objects.update_or_create(
    #         name=column[0],
    #         email=column[1],
    #         address=column[2],
    #         phone=column[3],
    #         profile=column[4]
    #     )
    context = {}

    return render(request, template, context)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model(dataset):
    # load dataset
    #dataset = pd.read_csv('train.csv', encoding='latin-1')
    dataset = dataset.rename(columns=lambda x: x.strip().lower())
    #dataset.head()

    # cleaning missing values
    dataset = dataset[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]
    dataset['sex'] = dataset['sex'].map({'male':0, 'female':1})
    dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce')
    dataset['age'] = dataset['age'].fillna(np.mean(dataset['age']))

    # dummy variables
    embarked_dummies = pd.get_dummies(dataset['embarked'])
    dataset = pd.concat([dataset, embarked_dummies], axis=1)
    dataset = dataset.drop(['embarked'], axis=1)
    dataset.head()

    X = dataset.drop(['survived'], axis=1)
    y = dataset['survived']

    # scaling features 
    sc = MinMaxScaler(feature_range=(0,1))
    X_scaled = sc.fit_transform(X)

    # model fit
    log_model = LogisticRegression(C=1)
    log_model.fit(X_scaled, y)

    import pickle
    pickle.dump(log_model,open("titanic_survival_ml_model.sav", "wb"))
    pickle.dump(sc, open("scaler.sav", "wb"))

    return 