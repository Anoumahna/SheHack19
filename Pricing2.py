import pandas as pd
import csv
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request , jsonify , render_template , flash , redirect, url_for, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pickle
import os

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'



@app.route("/entry",methods=['GET'])
def getEntry():
	name=str(request.args.get("type",""))
	

	inf = csv.reader(open('Database.csv','r'))
	for row in inf:
	  if(row[0]==name):
	  	X_t=row
	  	Y_t=row
	

	print(name)
	
	print("X_t" ,X_t)
	x_t=(int(X_t[4])-int(X_t[5]))/int(X_t[4])
	data=pd.read_csv(name+".csv")
	data=data.values

	if(not os.path.exists(name+'/.sav')):
		X_train=data[:,0:2]
		Y_train=data[:,-1]
		
		X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
		print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
		sk_linreg = LinearRegression()
		sk_linreg.fit(X_train, Y_train)

		pickle.dump(sk_linreg, open(name+'.sav', 'wb'))
	else:
		sk_linreg = pickle.load(open(name+'.sav', 'rb'))
	
	
	y_t=Y_t[3]
	ar = np.array([])
	W_t=np.append(ar, (x_t,y_t))
	
	W_t = W_t.astype(float)
	
	print("Result: ",(np.dot( W_t,sk_linreg.coef_)+sk_linreg.intercept_))
	return jsonify(result=np.dot( W_t,sk_linreg.coef_)+sk_linreg.intercept_)


def makeModel():
	print("Train: ")


@app.route("/foods", methods=['GET'])
def getAllFoodItems():
	hmap=dict({})
	inf = csv.reader(open('Book2.csv','r'))
	for row in inf:
		hmap[row[0]]=row[-1]

	return jsonify(foods=hmap)

@app.route("/", methods=['GET'])
def main():
	print('Answering')
	

app.run(port=7000)

	