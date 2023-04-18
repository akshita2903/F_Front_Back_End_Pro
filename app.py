import numpy as np
from flask import Flask, request,jsonify, render_template
import pickle


app=Flask(__name__)

model=pickle.load(open('models/model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
   

@app.route('/predict',methods=['POST'])
def predict():
    # print('in api')
    int_features=[int(x) for x in request.form.values()]
    # print(int_features)
    int_features=[2,1,1,1,1,1]
    features=[np.array(int_features)]
    print(features)
    prediction=model.predict(features)
    result=prediction[0]
    return render_template('index.html',prediction=result)

if __name__=="__main__":
    app.run(debug=True)
# # print("hello")
# import numpy as np
# from flask import Flask, request,jsonify, render_template
# import pickle

# app=Flask(__name__)


# def ValuePredictor(to_predict_list):
#     to_predict=np.array(to_predict_list).reshape(1,4)
#     loaded_model=pickle.load(open('models/model.pkl','rb'))
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







