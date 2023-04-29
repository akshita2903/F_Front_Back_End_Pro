import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)

model=pickle.load(open('trained_model.pkl' , 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
   

@app.route('/predict')
def predict():
#     print('in api form values')
#     for x in request.form.values():
#         print(x)
    
#     int_features=[int(x) for x in request.form.values()]
#     # print(int_features)
#     int_features=[2,1,1,1,1,1]
# #     print("features array is as follows:")
    
# #     features=[np.array(int_features)]
    
# #     print(features)

#     prediction=model.predict(int_features)

    # result=prediction[0]

    form_values = {
        'age': request.args.get('age'),
        'gender' : request.args.get('gender'),
        'family_history' : request.args.get('family_history'),
        'self_employed': request.args.get('self_employed'),
        'treatment': request.args.get('treatment'),
        'work_interfere': request.args.get('work_interfere'),
        'benefits':request.args.get('Benefits')
    }

    
    print("model is ",model)
    df = pd.DataFrame(columns = ['Age','Gender','family_history'
                                 ,'benefits', 'care_options', 'anonymity','leave', 'work_interfere'])# Add records to dataframe using the .loc functiondf.loc[0] = [2014,"toyota","corolla"] 
    df.loc[0] = [float(form_values['age']), int(form_values['gender']), int(form_values['family_history'])
                 ,int(form_values('benefits')), 1, 0, 0, int(form_values['work_interfere'])] 
    # gend = df.at[0, 'Age']
    # print(type(gend))
    # print("various data types are:")
    # print(df.dtypes)
    # df.loc[0] = [0.340909, 1, 1, 2, 1, 0, 0 ,4]
    prediction = model.predict(df)
   
    print("prediction result is " ,prediction)
    if int(prediction) == 1:
        prediction="Treatment Require"
    else:
        prediction="You are fine"    
        
    #print the form values by sending the form data to result.html
    return render_template('result.html', values = form_values, result=prediction)
    
    # result = 2
    # return render_template('index.html',prediction=result)


if __name__=="__main__":
    app.run(debug=True)





# print("hello")
# import numpy as np
# from flask import Flask, request,jsonify, render_template
# import pickle

# app=Flask(__name__)


# def ValuePredictor(to_predict_list):
#     to_predict=np.array(to_predict_list).reshape(1,4)
#     loaded_model=pickle.load(open('models/model.pkl','rb'))
# <<<<<<< HEAD
#     print("model is ",loaded_model)
# =======
# >>>>>>> 37e62877e1611daeaaf2615eb058e6b9ef164320
#     result=loaded_model.predict(to_predict)
#     return result[0]

# @app.route('/')
# def home():
#         return render_template('index.html')    
   

# @app.route('/predict',methods=['POST'])
# def predict():
    
#      to_predict_list=request.form.to_dict()
#      to_predict_list=list(to_predict_list.values())
#      to_predict_list=list(map(int, to_predict_list))
# <<<<<<< HEAD
#      print(to_predict_list)
# =======
# >>>>>>> 37e62877e1611daeaaf2615eb058e6b9ef164320
#      result=ValuePredictor(to_predict_list)
#      print(result)
#      if int(result)==1:
#             prediction="You require treatment"
#      else:
#             prediction="You are fine"
#      return render_template('index.html' , prediction=prediction)    

# if __name__=="__main__":
#       app.run(debug=True)
#       print("hello")







