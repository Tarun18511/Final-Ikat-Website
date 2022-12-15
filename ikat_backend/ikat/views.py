from django.shortcuts import render,HttpResponse,redirect
from django.contrib import messages
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User, auth
from django.conf import settings
from .models import ResultImage


import os

#crop predictor imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

#disease detection imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing import image

def home(request):
    return render(request,'home.html')


def register(request):
    if request.method=='POST':
        first_name=request.POST['first_name']
        last_name=request.POST['last_name']
        username=request.POST['username']
        password1=request.POST['password1']
        password2=request.POST['password2']
        email=request.POST['email']
        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,"Username taken")
            elif User.objects.filter(email=email).exists():
                messages.info(request,"Email exists")
            else:
                user=User.objects.create_user(username=username,password=password1,email=email,first_name=first_name,last_name=last_name)
                user.save();
                messages.info(request,'user created')
                return redirect('login')
        else:
            messages.info(request,"password not matching")

        return redirect('/')
    else:
        return render(request,'register.html')

def login(request):
    if request.method=='POST':
        username=request.POST['username']
        password=request.POST['password']

        user=auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request,user)
            return redirect("image")
        else:
            messages.info(request,'invalid credentials')
            return redirect('login')
    else:
        return render(request,'login.html')

def logout(request):
    auth.logout(request)
    return redirect('/')

def cropprediction(request):
    resultvalue=None
    # nitrogen=None
    # potassium=None
    temperature=None
    humidity=None
    ph=None
    rainfall=None 
    #88, 52, 30, 40, 73, 5, 190
    if request.method=='POST':
        # nitrogen=request.POST['nitrogen']
        # phosphorus=request.POST['phosphorus']
        # potassium=request.POST['potassium']
        temperature=request.POST['temperature']
        humidity=request.POST['humidity']
        ph=request.POST['ph']
        rainfall=request.POST['rainfall']
        module_dir = os.path.dirname(__file__)   #get current directory
        file_path = os.path.join(module_dir, 'Crop_recommendation.csv')
        df=pd.read_csv(file_path)
        df['label'].value_counts()
        features = df[['temperature', 'humidity', 'ph', 'rainfall']]
        target = df['label']
        labels = df['label']
        acc = []
        model = []
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
        from sklearn.ensemble import RandomForestClassifier
        RF = RandomForestClassifier(n_estimators=20, random_state=0)
        RF.fit(Xtrain,Ytrain)
        predicted_values = RF.predict(Xtest)
        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('RF')
        print("RF's Accuracy is: ", x)
        print(classification_report(Ytest,predicted_values))
        ls=[0,2,-4,5,-10]
        oglist=[32.603016, 65.3, 6.7, 140.91]
        a=[]
        for i in ls:
            oglist[0]=oglist[0] + i
            oglist[1]=oglist[1] - i
            oglist[2]=oglist[2] - i/10
            oglist[3]=oglist[3] + i*10
            data = np.array([oglist])
            prediction = RF.predict(data)
            a.append(prediction)
        # print(np.unique(a))

        ls=[0,2,-4,5,-10]
        oglist=[35, 70.3, 7.0, 150.9]
        a=[]
        for i in ls:
            oglist[0]=oglist[0] + i
            oglist[1]=oglist[1] - i
            oglist[2]=oglist[2] - i/10
            oglist[3]=oglist[3] + i*10
            data = np.array([oglist])
            prediction = RF.predict(data)
            a.append(prediction)
        # print(np.unique(a))

        ls=[0,2,-4,5,-10]
        oglist=[21,28, 4, 130]
        a=[]
        for i in ls:
            oglist[0]=oglist[0] + i
            oglist[1]=oglist[1] - i
            oglist[2]=oglist[2] - i/10
            oglist[3]=oglist[3] + i*10
            data = np.array([oglist])
            prediction = RF.predict(data)
            a.append(prediction)
        # print(np.unique(a))

        ls=[0,2,-4,5,-10]
        oglist=[21, 82, 6, 202.93]
        a=[]
        for i in ls:
            oglist[0]=oglist[0] + i
            oglist[1]=oglist[1] - i
            oglist[2]=oglist[2] - i/10
            oglist[3]=oglist[3] + i*10
            data = np.array([oglist])
            prediction = RF.predict(data)
            a.append(prediction)
        # print(np.unique(a))
        resultvalue=prediction
    context = {
        'resultvalue':resultvalue,
        'temperature':temperature,
        'humidity':humidity,
        'ph':ph,
        'rainfall':rainfall
    }
    return render(request, 'cropprediction.html',context)

def load_image(path):
    return image.load_img(path, target_size = (64, 64))

def pass_image(path):
    disease_image = image.img_to_array(load_image(path))
    disease_image = np.array(disease_image) / 255.0
    return disease_image

def index(request):
    filePathName=None
    image=None
    result=None
    if request.method == 'POST':
        print (request)
        print (request.POST.dict())
        fileObj=request.FILES['filePath']
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        # print(filePathName)
        image="."+filePathName
        print(image)

        path =image


        new_image = pass_image(path)

        module_dir = os.path.dirname(__file__)  
        file_path = os.path.join(module_dir, 'my_model.h5')

        new_model = tf.keras.models.load_model(file_path)

        new_model.summary()

        result = new_model.predict(new_image.reshape(1, 64, 64, 3))

        diseases = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',

                    'Potato___Early_blight', 'Potato___Late_blight',
                    'Potato___healthy', 'Tomato_Bacterial_spot',
                    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
                    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
                    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

        result = list(result)

        pos = result.index(max(result))

        print(diseases[pos])
    context = {
    'result':result,
    }
    return render(request, 'index.html',context)