#строим логистическую регрессию - угадываем людей с доходом более 50 тыс по признакам.
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv('adult.csv')


#посмотрим на большее число колонок
pd.set_option('display.expand_frame_repr', False)


#ячейки со знаком "?" заменяем "NaN" и удаляем 
data = data.replace('?', np.nan)
data = data.dropna()


#удалим целевую функцию и запомним ее отдельной переменной
income=data['income']
del data['income']

#попробуем угадать доход на основе: возраста, часов работы, рассы, пола
selectedColumns = ['gender', 'education', 'occupation', 'native-country', 'hours-per-week', 'workclass', 'age', 'race', 'capital-gain', 'relationship']
data = data[selectedColumns]


#переведем значения race, gender в 0 и 1
data = pd.get_dummies(data, columns= ['gender', 'education', 'occupation', 'native-country', 'workclass', 'race', 'relationship'])

#разделим выборку на тренировочную и тестовую
X_train, X_test, y_train, y_test = train_test_split(data, income)


#обучаем
model=LogisticRegression()
model.fit(X_train, y_train)
prediction=model.predict(X_test)


print('accuracy_score', accuracy_score(y_test, prediction))



