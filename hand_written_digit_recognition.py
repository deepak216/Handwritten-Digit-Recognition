import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# loading training data
df_train=pd.read_csv("./train.csv")

X_training_data=df_train.iloc[:, 1:]
y_training_data=df_train.iloc[:,0]
for index, row in X_training_data.iterrows():
	for c in range(len(row)):
		if row[c]!=0:
			row[c]=1

#loading testing data to predict the digits
df_test=pd.read_csv("./test.csv")
for index1, row_test in df_test.iterrows():
 	for c1 in range(len(row_test)):
  		if row_test[c1]!=0:
  			row_test[c1]=1

X_test_data=df_test.iloc[:, ]
X_train,X_test,y_train,y_test= train_test_split(X_training_data, y_training_data , test_size=0,random_state=4)

print("Neural Network Start:")
clf = MLPClassifier(solver='sgd', activation='logistic',alpha=0.0001,hidden_layer_sizes=(15,7),max_iter=2000, random_state=1)
clf.fit(X_train, y_train)
print "training done:"
y_pred=clf.predict(X_test_data)
#result is stored in the out.txt files
f=open('./out.txt','w')
id=1
for item in y_pred:
	s1=str(item)
	temp=str(id)+' '+str(s1)
	f.write(temp)
	id=id+1
	f.write('\n')
f.close()

#print('Accuracy Score:')
#print(metrics.accuracy_score(y_test,y_pred))