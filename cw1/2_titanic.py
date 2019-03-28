#!/usr/bin/env python
# coding: utf-8

# # Титаник. Кто выживет?

# https://www.kaggle.com/c/titanic/

# In[12]:


import numpy as np
import pandas as pd


# In[13]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from pylab import rcParams
rcParams['figure.figsize'] = (9, 6)


# ### Данные

# In[14]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[15]:


train.head()


# In[16]:


test.head()


# In[17]:


train.info()


# # EDA

# In[18]:


train.columns


# In[19]:


train.Sex.value_counts(dropna=False)


# In[20]:


get_ipython().run_line_magic('pinfo', 'sns.boxplot')


# In[21]:


sns.boxplot(data=train, x='Fare', y='Sex')


# In[22]:


train.groupby('Sex')['Pclass'].value_counts(normalize=True)


# ### Фичи

# чтобы одинаковым образом обработать train и test и не дублировать все операции 2 раза, соединим эти два набора данных в один, не забыв при этом:
# 1. выкинуть целевую переменную из train
# 2. проверить на соответствие набора признаков друг другу
# 3. добавить флаг того, является ли объект тестовым или нет

# In[23]:


y_train = train.Survived
train.drop('Survived', axis=1, inplace=True)


# In[24]:


train.columns == test.columns


# In[26]:


train['is_test'] = 0
test['is_test'] = 1


# In[27]:


df = pd.concat([train, test])


# супер, теперь полный набор данных можно обрабатывать вместе и в любой момент, уже обработанными, обратно разъединить на обучающую и тестовую выборки

# Пол male/female закодируем в 1/0 и удалим переменные, с которыми мы не будем сейчас работать

# In[28]:


df["isMale"] = df.Sex.replace({"male": 1, "female":0})
df.drop(["Sex", "Cabin", "Ticket", "Name", "PassengerId"], axis=1, inplace=True)


# признаки, значения которых составляют небольшой перечислимый набор, закодируем в отдельные столбцы 

# In[29]:


df.Pclass.value_counts()


# In[30]:


df_dummies = pd.get_dummies(df, columns=['Pclass', 'Embarked'])


# In[31]:


df_dummies.head(10)


# In[32]:


df_dummies.isnull().sum()


# In[33]:


X_train = df_dummies[df_dummies.is_test==0].drop('is_test', axis=1)
X_test = df_dummies[df_dummies.is_test==1].drop('is_test', axis=1)


# In[34]:


columns = X_train.columns


# In[35]:


X_train.head(10)


# ### Заполнение пустых значений

# заполним пустые значения средними по соответственным признакам

# In[36]:


from sklearn.preprocessing import Imputer


# In[37]:


imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)


# In[38]:


imputer.fit(X_train)


# In[39]:


X_train_imputed = imputer.transform(X_train)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=columns)


# In[40]:


X_train_imputed.head(10)


# ### Нормировка значений

# In[41]:


from sklearn.preprocessing import StandardScaler


# In[42]:


scaler = StandardScaler()


# In[43]:


scaler.fit(X_train_imputed)


# In[44]:


X_train_imputed_scaled = scaler.transform(X_train_imputed)
X_train_imputed_scaled = pd.DataFrame(X_train_imputed_scaled, columns=columns)


# In[45]:


X_train_imputed_scaled.head(10)


# In[46]:


X_test_imputed_scaled = scaler.transform(imputer.transform(X_test))


# ### Offtop: попробуем визуализировать всех пассажиров: есть ли там кластеры?
# ### PCA + clustering

# In[47]:


from sklearn.decomposition import PCA


# In[48]:


pca = PCA(n_components=2)


# In[49]:


ppl = pca.fit_transform(X_train_imputed_scaled)


# всего 2 фичи объясняют 41% всего разнообразия пассажиров:

# In[51]:


pca.explained_variance_ratio_.sum()


# классно, видно 6 кластеров пассажиров: внутри кластера они похожи друг на друга, межу кластерами - нет:

# In[52]:


plt.plot(ppl[:,0], ppl[:,1], 'ro', alpha=0.1)
plt.title('Пассажиры Титаника')


# можно попробовать кластеризовать по 7 кластерам и проверить, что получится:

# In[53]:


from sklearn.cluster import KMeans


# In[54]:


n_clusters = 7


# In[55]:


kmeans = KMeans(n_clusters=n_clusters)


# In[56]:


kmeans.fit(X_train_imputed_scaled)


# In[57]:


cluster_labels = kmeans.predict(X_train_imputed_scaled)


# In[58]:


plt.title('Пассажиры Титаника')
for i,color in zip(range(n_clusters),{'blue','red','green','black','orange','yellow'}):
    t = ppl[cluster_labels==i]
    plt.plot(t[:,0], t[:,1], 'ro', alpha=0.1, c=color)


# эти номера кластеров можно было бы подать как ещё одна фича

# ### Разделение на обучающую и тестирующую выборки

# In[59]:


from sklearn.model_selection import train_test_split


# In[61]:


X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train_imputed_scaled, y_train, test_size=0.2)


# In[62]:


X_train_fin.shape


# In[63]:


X_val.shape


# In[64]:


X_test_imputed_scaled.shape


# In[66]:


y_train_fin.shape


# In[67]:


y_val.shape


# ### Обучение с кросс-валидацией

# кросс-валидация поможет нам подобрать лучший параметр регуляризации

# In[68]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# In[69]:


cs = 10**np.linspace(-3,1,5)
cs


# In[70]:


grid = {'C': cs}
gridsearch = GridSearchCV(LogisticRegression(), grid, scoring='accuracy', cv=5)


# In[71]:


get_ipython().run_cell_magic('time', '', 'gridsearch.fit(X_train_fin, y_train_fin)')


# In[72]:


sorted(gridsearch.grid_scores_, key = lambda x: -x.mean_validation_score)


# In[73]:


gridsearch.best_params_


# In[74]:


best_C = gridsearch.best_params_["C"]


# # Оценка точности

# In[75]:


from sklearn.metrics import accuracy_score


# In[76]:


clf = LogisticRegression(C=best_C)


# In[77]:


clf.fit(X_train_fin, y_train_fin)


# In[78]:


y_val_pred = clf.predict(X_val)


# In[79]:


accuracy_score(y_val, y_val_pred)


# # Финальное предсказание

# In[80]:


clf.fit(X_train_imputed_scaled, y_train)


# предсказание вероятностей принадлежности классу 0 и 1:

# In[81]:


clf.predict_proba(X_test_imputed_scaled)[:10]


# предсказание номера класса:

# In[82]:


predictions = clf.predict(X_test_imputed_scaled)
predictions


# In[83]:


submussion = 'PassengerId,Survived\n'
submussion += "\n".join(["{},{}".format(pid, prediction) for pid, prediction in zip(test.PassengerId, predictions)])


# In[84]:


with open('submission.txt', 'w') as file:
    file.write(submussion)


# In[85]:


for col, val in zip(X_train.columns, clf.coef_[0]):
    print("{:30} {:.2f}".format(col, val))


# Регрессия позволяет посмотреть влияние различных факторов на принятое решение. Так, видно, что женский пол, маленький возраст и первый класс являлись сильными предпосылками к выживанию
