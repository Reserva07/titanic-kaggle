import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Pega dado da cabine
train['Deck'] = train['Cabin'].str[0]
train['Deck'] = train['Deck'].fillna('Desconhecido')

test['Deck'] = test['Cabin'].str[0]
test['Deck'] = test['Deck'].fillna('Desconhecido')

train['TemCabine'] = train['Cabin'].notna().astype(int)
test['TemCabine'] = test['Cabin'].notna().astype(int)

train_deck = pd.get_dummies(train['Deck'], prefix = 'Deck')
test_deck = pd.get_dummies(test['Deck'], prefix = 'Deck')

test_deck = test_deck.reindex(columns=train_deck.columns, fill_value=0)

train = pd.concat([train, train_deck], axis = 1)
test = pd.concat([test, test_deck], axis = 1)



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


#impute Age by the median age for each title
train['Age'] = train['Age'].fillna(train.groupby('Titulo')['Age'].transform('median'))
test['Age'] = test['Age'].fillna(train.groupby('Titulo')['Age'].transform('median'))


#Adicionar tamanho da família 
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

#Coletar Embarked
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

train_embarked = pd.get_dummies(train['Embarked'], prefix = 'Embarked')
test_embarked = pd.get_dummies(test['Embarked'], prefix = 'Embarked')

test_embarked = test_embarked.reindex(columns = train_embarked.columns, fill_value= 0)

train = pd.concat([train, train_embarked], axis = 1)
test = pd.concat([test, test_embarked], axis =1 )

variaveis= ['Sex_binario', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'TemCabine'] + list(train_titulo_dummies.columns) + list(train_embarked.columns) + list(train_deck.columns)

X = train[variaveis].fillna(-1)
y = train['Survived']

# Validaçao cruzada
resultados = []

kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=10)

# Search for the best hyperparameters
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 300],
}

grid = GridSearchCV(
    estimator = GradientBoostingClassifier(random_state=0),
    param_grid=param_grid,
    cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=10),
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X, y)
print(grid.best_params_)

b_params = grid.best_params_


for linhas_treino, linhas_valid in kf.split(X, y):
    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = GradientBoostingClassifier(n_estimators=100, random_state=0)
    modelo.fit(X_treino, y_treino)
    p = modelo.predict(X_valid)
    acc = np.mean(y_valid == p)
    resultados.append(acc)

plt.hist(resultados)
#plt.show()
print(np.mean(resultados))

modelo.fit(X, y)

X_prev = test[variaveis]
X_prev = X_prev.fillna(-1)

modelo = GradientBoostingClassifier(n_estimators=100, random_state=0)
modelo.fit(X, y)
p = modelo.predict(X_prev)

sub = pd.Series(p, index=test['PassengerId'], name='Survived')
sub.to_csv("Decimo_segundo_modelo.csv", header = True)

