import pandas as pd
import pickle

dataset = 'SUV_Purchase.csv'
df = pd.read_csv(dataset)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df.drop('User ID', axis = 1, inplace=True)

#loading
X = df.drop(['Purchased'],axis=1)
Y = df[['Purchased']]

#splitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#scaling
from sklearn.preprocessing import StandardScaler
sst = StandardScaler()
X_train = sst.fit_transform(X_train)
X_test = sst.transform(X_test)

with open('sst.pkl','wb') as file:
    pickle.dump(sst,file)


#train the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,Y_train)

#test
Y_pred = rf.predict(X_test)
print(Y_pred)

#accuracy
print(rf.score(sst.transform(X),Y)*100)

#pickling- serialization of model
import pickle

pickle.dump(rf,open('model.pkl','wb')) #serializing our model
print("Model is dumped")


