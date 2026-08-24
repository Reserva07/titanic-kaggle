import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def tranformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0

train['Sex_binario'] = train['Sex'].map(tranformar_sexo)
test['Sex_binario'] = test['Sex'].map(tranformar_sexo)

variaveis= ['Sex_binario', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']

X = train[variaveis].fillna(-1)
y = train['Survived']
X_falso = np.arange(10)
X_falso
np.random.seed(0)
train_test_split(X_falso, test_size=0.5)

np.random.seed(1)
X_treino, X_valid, y_treino, y_valid = train_test_split(X, y, test_size=0.5)

modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(X_treino, y_treino)

p = modelo.predict(X_valid)

# Considerando apenas mulher = sobrevive. Feito para critério de comparaçao
p = (X_valid['Sex_binario'] == 1).astype(np.int64)
#print(np.mean(y_valid == p))

# Validaçao cruzada
resultados = []

kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)

for linhas_treino, linhas_valid in kf.split(X):
    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    modelo.fit(X_treino, y_treino)
    p = modelo.predict(X_valid)
    acc = np.mean(y_valid == p)
    resultados.append(acc)
    print("Acc:", acc)
    print()

plt.hist(resultados)
plt.show()
print(np.mean(resultados))

X = X.fillna(-1)
modelo.fit(X, y)

X_prev = test[variaveis]
X_prev = X_prev.fillna(-1)

p = modelo.predict(X_prev)

modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(X, y)
p = modelo.predict(test[variaveis])

sub = pd.Series(p, index=test['PassengerId'], name='Survived')
sub.to_csv("Segundo_modelo.csv", header = True)

