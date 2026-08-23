import pandas as pd 
import numpy as np 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#train.head()
#print(train.head())

from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, n_jobs =-1,random_state=0)

def transform_sex(valor):
    if valor == 'male':
        return 0
    else:
        return 1

train['Sex_binario'] = train['Sex'].map(transform_sex)

#train.head()
#print(train.head())

variaveis = ['Sex_binario', 'Age']
X = train[variaveis]
y = train['Survived']

X = X.fillna(-1)
modelo.fit(X,y)

test['Sex_binario'] = test['Sex'].map(transform_sex)
X_prev = test[variaveis]
X_prev = X_prev.fillna(-1)

previsao = modelo.predict(X_prev)

sub = pd.Series(previsao, index=test['PassengerId'], name='Survived')
sub.to_csv('primeira_previsao.csv', header=True)

print(sub.head(10))
