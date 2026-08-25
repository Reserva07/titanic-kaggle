import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Transforma o sexo em número
def tranformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0

train['Sex_binario'] = train['Sex'].map(tranformar_sexo)
test['Sex_binario'] = test['Sex'].map(tranformar_sexo)


#Coletar o Titulo
train['Titulo'] = train['Name'].str.extract(r',\s*([^\.]*)\.') #r faz o python nao entender \s como codigo, '' é espaço para escrever, a virgula indica onde começa, o \s* diz para pular espaço vazio e zero
test['Titulo'] = test['Name'].str.extract(r',\s*([^\.]*)\.') #() é o que deve ser devolvido, [^\.]* bate com qualquer sequencia de caracteres, desde que nao seja um ponto. \. bate com o ponto para encerrar

train_titulo_dummies = pd.get_dummies(train['Titulo'], prefix='Titulo')
test_titulo_dummies = pd.get_dummies(test['Titulo'], prefix = 'Titulo')

test_titulo_dummies = test_titulo_dummies.reindex(columns=train_titulo_dummies.columns, fill_value=0)

train = pd.concat([train, train_titulo_dummies], axis = 1)
test = pd.concat([test, test_titulo_dummies], axis = 1)

variaveis= ['Sex_binario', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare'] + list(train_titulo_dummies.columns)

X = train[variaveis].fillna(-1)
y = train['Survived']

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

plt.hist(resultados)
plt.show()
print(np.mean(resultados))

modelo.fit(X, y)

X_prev = test[variaveis]
X_prev = X_prev.fillna(-1)

modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(X, y)
p = modelo.predict(X_prev)

sub = pd.Series(p, index=test['PassengerId'], name='Survived')
sub.to_csv("Terceiro_modelo.csv", header = True)

