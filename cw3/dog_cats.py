#Kaggle competition: Shelter Animal Outcomes
#https://www.kaggle.com/c/shelter-animal-outcomes/

#Дано: данные о кошках и собаках, поступивших в приют

#Найти: что с ними станет? Возьмут в приют / вернётся хозяин / ...


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns



train = pd.read_csv('data/shelter/train.csv')
test = pd.read_csv('data/shelter/test.csv')
sample_submission = pd.read_csv('data/shelter/sample_submission.csv')

pd.set_option('display.expand_frame_repr', False)

train.head()
test.head()
sample_submission.head()



Xtrain = train.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype'], axis=1)
Xtest = test.drop(['ID'], axis=1)
Xtrain['is_test'] = False
Xtest['is_test'] = True
X = pd.concat([Xtrain, Xtest], axis=0)
X.index = range(len(X))
X.columns = X.columns.str.lower()

print(X.head())


print(X.shape)


X.info()

#Также нам необходимо закодировать значения целевой переменной
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(train.OutcomeType)
#Варим фичи
#будем в конце матрицы наращивать числовые фичи, не удаляя оригинальные: вдруг что всплывёт
#имя превращаем в:наличие имени
#длину имени
#частоту имени

X['has_name'] = X.name.isnull().astype(int)

X['name_len'] = X.name.str.len()
X.name_len.fillna(X.name_len.median(), inplace=True)

X['name_words_num'] = X.name.str.split().str.len()
X.name_words_num.fillna(X.name_words_num.median(), inplace=True)

names_freq = X.name.value_counts().to_dict()
X['name_freq'] = X.name.apply(lambda x: names_freq.get(x))
#Даты: переводим строки в даты

X.datetime = pd.to_datetime(X.datetime)
#X.datetime.hist()
#plt.show()

X['year'] = X.datetime.apply(lambda x: x.year)
X['month'] = X.datetime.apply(lambda x: x.month)
X['day'] = X.datetime.apply(lambda x: x.day)
X['hour'] = X.datetime.apply(lambda x: x.hour + x.minute/60)
X['weekday'] = X.datetime.apply(lambda x: x.weekday())
#С типом животного тут совсем просто

X.animaltype.value_counts()

X['is_dog'] = (X.animaltype=='Dog').astype(int)
#С полом сложнее..

#Neutered, Spayed - стерилизованные
#Intact - нетронутые

X.sexuponoutcome.value_counts()

X.sexuponoutcome.fillna('Unknown', inplace=True)

X['sterilization'] = X.sexuponoutcome.apply(lambda x: x.split()[0])
X.sterilization = X.sterilization.replace({'Neutered': 'Sterilized', 'Spayed': 'Sterilized'})

X['sex'] = X.sexuponoutcome.apply(lambda x: x.split()[-1])

from sklearn import preprocessing
le_sterilization = preprocessing.LabelEncoder()
le_sex = preprocessing.LabelEncoder()

X.sterilization = le_sterilization.fit_transform(X.sterilization)
X.sex = le_sex.fit_transform(X.sex)
#почему мы не вставили сюда сразу числовые фичи? потому что это категориальные фичи, у них по 3 значения

#print(X.ageuponoutcome.value_counts()[:30])
#print(X.ageuponoutcome.str.split().str[1].value_counts())
#print(X.ageuponoutcome.str.split().str[1].str.rstrip('s').value_counts())
t_dig = X.ageuponoutcome.str.split().str[0].fillna(0).astype(int)
t_int = X.ageuponoutcome.str.split().str[1].str.rstrip('s').replace({'year': 365, 'month': 365/12, 'week':7, 'day':1}).fillna(0)
X["years"] = t_dig*t_int/365
#порода достаточно разнообразна и содержит в себе также некое поле "Mix" и краткошёрстность животного
#print(X.breed.value_counts()[:10])
#Сегодня сделаем просто: соединим все описания в один большой текст и посчитаем в нём вхождение каждого слова. Флаги наличия самых популярных и включим как фичи

from collections import Counter

one_big_text = " ".join(X.breed)
words = one_big_text.replace('/',' / ').split()
most_common = Counter(words).most_common()
print(most_common[:20])

#Можно было бы просто вставить флаги вхождения первых, скажем, 4 слов.
#Но давайте посмотрим, а насколько они важны?
#Составим матрицу, состоящую только из вхождения первых N слов,
#обучим на них дерево и проверим важность фичей
Xbreed = pd.DataFrame()
for col, num in most_common[:10]:
    Xbreed[col] = X[~X.is_test].breed.str.contains(col).astype(int)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4)

clf.fit(Xbreed, train.OutcomeType)
print('feature_importances_', clf.feature_importances_)
print(Xbreed.columns)
#В принципе, оказалось, что влияет по большей части только Domestic. Но включим сюда на всякий случай ещё один признак
X['is_domestic'] = X.breed.str.contains('Domestic').astype(int)
X['is_mix'] = X.breed.str.contains('Mix').astype(int)
#последнее! цвет. Выглядит похоже на проблему породы, повторим те же операции
print(X.color.value_counts()[:10])

one_big_text = " ".join(X.color)
words = one_big_text.replace('/',' / ').split()
most_common = Counter(words).most_common()
print(most_common[:20])

Xcolor = pd.DataFrame()
for col, num in most_common[:10]:
    Xcolor[col] = X[~X.is_test].color.str.contains(col).astype(int)

clf.fit(Xcolor, train.OutcomeType)
print(clf.feature_importances_)
print(Xcolor.columns)
#Здесь важность менее сосредоточена, возьмём первые 5 фичей. Самая важная, Tabby - это полосатость

X['is_color_tabby'] = X.color.str.contains('Tabby').astype(int)
X['is_color_mix'] = X.color.str.contains('/').astype(int)
X['is_color_white'] = X.color.str.contains('White').astype(int)
X['is_color_black'] = X.color.str.contains('Black').astype(int)
X['is_color_brown'] = X.color.str.contains('Brown').astype(int)
#Осталось проверить итоговую таблицу на пропуски

print(X.isnull().sum())

X.name_freq.fillna(X.name_freq.median(), inplace=True)
#Разделим обратно на обучающую и тестовую выборки, дропнув при этом все лишние столбцы, которые были изначально
Xtrain_prep = X[~X.is_test].drop(Xtrain.columns.str.lower(), axis=1)
Xtest_prep = X[X.is_test].drop(Xtrain.columns.str.lower(), axis=1)
print(Xtrain_prep.shape)
#Всё

le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)
#Всё, у нас есть Xtrain, Xtest, ytrain. Осталось получить ytrain.
#Обучим решающее дерево, причём применим кросс-валидацию для подбора гиперпараметра глубины дерева
#Обращаю внимание, что мы в явном виде указали вид функционала качества: scoring='neg_log_loss'.
#Выбрана именно эта функция, так как именно она будет оценивать качество на Kaggle

from sklearn.model_selection import GridSearchCV

depths = np.arange(1,10)
#features_num = np.arange(5,15)
grid = {'max_depth': depths}#, 'max_features': features_num}
gridsearch = GridSearchCV(DecisionTreeClassifier(), grid, scoring='neg_log_loss', cv=5)

gridsearch.fit(Xtrain_prep, y)

print(sorted(gridsearch.cv_results_, key = lambda x: -x.mean_validation_score))

scores = [-x.mean_validation_score for x in gridsearch.cv_results_]
plt.plot(depths, scores)
plt.scatter(depths, scores)
best_point = np.argmin(scores)
plt.scatter(depths[best_point], scores[best_point], c='g', s=100)

#Так, отлично, спасли себя от переобучения (правда, в Grid можно было указать и другие параметры и найти более оптимальную точку).
#Фиксируем max_depth=5
clf_final = DecisionTreeClassifier(max_depth=5)

clf_final.fit(Xtrain_prep, y)
#Делаем предсказания
y_pred_proba = clf_final.predict_proba(Xtest_prep)
y_pred = clf_final.predict(Xtest_prep)
#Формируем сабмит
submit = pd.DataFrame(y_pred_proba, columns=sample_submission.columns[1:])
submit['ID'] = sample_submission.ID
submit = submit[[submit.columns[-1]]+list(submit.columns[:-1])]
submit.to_csv('data/shelter/res/submit.csv', index=False)
#Что оказалось под капотом?

from sklearn.tree import export_graphviz
def get_tree_dot_view(clf, feature_names=None, class_names=None):
    print(export_graphviz(clf, out_file=None, filled=True, feature_names=feature_names, class_names=class_names))
#http://www.webgraphviz.com

print(pd.Series(le.inverse_transform(y)).value_counts().sort_index())
print(get_tree_dot_view(clf_final, list(Xtrain_prep.columns), list(le.classes_)))
#Сохраним для следующего ноутбука

Xtrain_prep.to_pickle('data/shelter/res/xtrain.pkl')
Xtest_prep.to_pickle('data/shelter/res/xtest.pkl')
#Также можно сохранять и модели:

from sklearn.externals import joblib

joblib.dump(clf, 'data/shelter/res/clf_decisiontree_maxdepth5.pkl')
['data/shelter/clf_decisiontree_maxdepth5.pkl']
