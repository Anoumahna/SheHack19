import pandas as pd
import csv
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request,jsonify
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
	X_t=[]
	Y_t=[]
	count=1;
	for row in inf:
	  if(row[0]==name):
	  	X_t.append(row)
	  	Y_t.append(row)

	

	print(name)
	
	print("X_t" ,X_t)
	x_t=[]
	y_t=[]
	for row in X_t:
		x_t.append((int(row[4])-int(row[5]))/int(row[4]))
	#x_t=(int(X_t[4])-int(X_t[5]))/int(X_t[4])
	for row in Y_t:
		y_t.append(row[3])
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
	
	
	#y_t=Y_t[3]
	ar = np.array([])
	W_t=zip(x_t,y_t)
	W_t=list(W_t)
	
	#W_t = W_t.astype(float)
	
	#print("Result: ",(np.dot( W_t,sk_linreg.coef_)+sk_linreg.intercept_))
	price=[];
	for row in W_t:
		print("Row" ,row)
		#row=row.astype(float)
		row=list(map(float,row))
		price.append(np.dot(row,sk_linreg.coef_)+sk_linreg.intercept_)
	print("Prices: ",price)
	#json.dump()
	return jsonify(name=name, price=price)


def makeModel():
	print("Train: ")


@app.route("/foods", methods=['GET'])
def getAllFoodItems():
	hmap=dict({})
	inf = csv.reader(open('Database.csv','r'))
	for row in inf:
		hmap[row[0]]=row[-1]

	return jsonify(foods=hmap)

@app.route("/", methods=['GET'])
def main():
	print('Answering')

app.run(port=7000)

	