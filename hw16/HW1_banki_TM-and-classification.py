import json
import re
import bz2
import regex
from tqdm import tqdm
from scipy import sparse
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import *
from pymystem3 import Mystem
from pymorphy2 import MorphAnalyzer

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.corpora import *
from gensim.models import  *
from gensim import similarities
from pylab import pcolor, show, colorbar, xticks, yticks

from nltk.stem.snowball import RussianStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score, accuracy_score, confusion_matrix

from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

import warnings
warnings.filterwarnings('ignore')




# #  Домашнее задание по NLP # 1 [100 баллов]
# ## Классификация по тональности 
# В этом домашнем задании вам предстоит классифицировать по тональности отзывы на банки с сайта banki.ru.
# Данные содержат непосредственно тексты отзывов, некоторую дополнительную информацию, а также оценку по шкале от 1 до 5. 
# Тексты хранятся в json-ах в массиве responses.
# Посмотрим на пример отзыва:



def main(df):
    # ## Часть 1. Анализ текстов [40/100]
    
    
    # 1. Посчитайте количество отзывов в разных городах и на разные банки
    counts = df.city.value_counts()
    print('counts cities ', counts)

    counts = df.bank_name.value_counts()
    print('counts bank_names ', counts)
    
    # 2. Постройте гистограмы длин слов в символах и в словаx
    labels = counts.index.tolist()
    values = counts.tolist()
    #values = [len(word) for word in labels]
    y_pos = np.arange(len(labels))


    fig, ax = plt.subplots()
    ax.set_title('гистограмма частот слов')
    ax.bar(y_pos, values, align='center', alpha=0.5)
    plt.tick_params(axis='x', which='major', labelsize=6)
    plt.xticks(y_pos, labels, color='red', rotation='vertical')
    fig.savefig('hist.jpg')
    #plt.show()
    
    
    # 3. Найдите 10 самых частых:
    #     * слов
    #     * слов без стоп-слов
    #     * лемм 
    # * существительных

    # функция для удаления стоп слов
    mystopwords = stopwords.words('russian') + ['это', 'наш' , 'тыс', 'млн', 'млрд', 'также',
                                                'т', 'д', 'который','прошлый', 'сей', 'свой',
                                                'мочь', 'в', 'я', '-', 'мой', 'ваш', 'и', '5']
    def remove_stopwords(text, mystopwords = mystopwords):
        try:
            return " ".join([token for token in text.lower().split() if not token in mystopwords])
        except:
            return ""

    # функция лемматизации
    def lemmatize(text, mystem=Mystem()):
        try:
            return " ".join(mystem.lemmatize(text)).strip()  
        except:
            return ""

    # функция фильтрации существительных
    def filter_nouns(text, pm2=MorphAnalyzer()):
        try:
            return " ".join([word.normal_form for word in [pm2.parse(word)[0] for word in text.split()] if word.tag.POS == 'NOUN']).strip()  
        except:
            return ""


    # вывод 10 самых часто используемых
    def most_common(fieldname, amount=10, n_types=None, n_tokens=None):
        cnt = Counter()
        tokens = []
        for index, row in df.iterrows():
            tokens = row[fieldname].split()
            cnt.update(tokens)
            if not n_types is None:
                n_types.append(len(cnt))
            if not n_tokens is None:
                n_tokens.append(sum(cnt.values()))

        print('\n')
        if fieldname == 'text':
            print('%s самых часто используемых слов:\n' % amount)
        elif fieldname == 'text_without_stopwords':
            print('%s самых часто используемых слов без стоп слов:\n' % amount)
        elif fieldname == 'lemmas':
            print('%s самых часто используемых лемм:\n' % amount)
        elif fieldname == 'nouns':
            print('%s самых часто используемых существительных:\n' % amount)
            
        for elem in cnt.most_common(amount):
            print(elem)

        return cnt

    

    #df['text'] = df['text']
    df['text_without_stopwords']= df.text.apply(remove_stopwords)
    df['lemmas'] = df['text_without_stopwords'].apply(lemmatize)
    df['lemmas'] = df['lemmas'].apply(remove_stopwords)
    df['nouns'] = df['lemmas'].apply(filter_nouns)
    


    n_types = []
    n_tokens = []
    
    most_common('text')

    most_common('text_without_stopwords')

    cnt = most_common('lemmas', n_types=n_types, n_tokens=n_tokens)
    most_common('nouns')

    


    # 4. Постройте кривые Ципфа и Хипса

    #Ципфа
    freqs = list(cnt.values())
    freqs = sorted(freqs, reverse = True)

    fig, ax = plt.subplots()
    ax.plot(freqs[:300], range(300))

    ax.set_xlabel('номер слова')
    ax.set_ylabel('обратная частота')
    ax.set_title('кривая Ципфа')
    fig.savefig('ципф.jpg')

    #Хипса
    fig, ax = plt.subplots()
    ax.plot(n_tokens, n_types)

    ax.set_xlabel('размер текста в словах')
    ax.set_ylabel('количество типов слов')
    ax.set_title('кривая Хипса')
    fig.savefig('хипс.jpg')
    #plt.show()



    # 5. Ответьте на следующие вопросы:
    #     * какое слово встречается чаще, "сотрудник" или "клиент"?


    if cnt['клиент'] > cnt['сотрудник']:
            print("Слово 'Клиент' встречается чаще, чем 'сотрудник'")
    elif cnt['клиент'] < cnt['сотрудник']:
            print("Слово 'Cотрудник' встречается чаще, чем 'клиент'")
    else:
            print("Оба слова встречаются одинаково")
            

    #     * сколько раз встречается слова "мошенничество" и "доверие"?
    print("Слово 'мошенничество' встречается раз", cnt['мошенничество'])
    print("Слово 'доверие' встречается раз", cnt['доверие'])
    print("Слово 'деньги' встречается", cnt['деньги'])


    # 6. В поле "rating_grade" записана оценка отзыва по шкале от 1 до 5.
    #Используйте меру $tf-idf$, для того, чтобы найти ключевые слова и биграмы
    #для положительных отзывов (с оценкой 5) и отрицательных отзывов (с оценкой 1)

    tokens_by_grade = []


    ratings = df['rating_grade'].dropna().unique()
    ratings.sort()

    amount_border = 1
    for rating_grade in ratings:        
        if df['rating_grade'].value_counts()[rating_grade] > amount_border: 
            tokens = []
            samples = df[df['rating_grade']==rating_grade]
            for i in range(len(samples)):
                tokens += samples.lemmas.iloc[i].split()
            tokens_by_grade.append(tokens)


    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0)
    tfidf_matrix =  tfidf.fit_transform([' '.join(tokens) for tokens in tokens_by_grade])
    feature_names = tfidf.get_feature_names() 
    dense = tfidf_matrix.todense()

    def get_main_gramms(grade_id, info, amount=30):  
        tfidf_ranking = []
        text = dense[grade_id].tolist()[0]
    
        phrase_scores = [pair for pair in zip(range(0, len(text)), text) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        phrases = []
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:amount]:
            tfidf_ranking.append(phrase)

        rankings = pd.DataFrame({'tf-idf': tfidf_ranking})
        print('\n', info, ':\n')
        print(tfidf_ranking)

    get_main_gramms(0, info='Ключевые слова для отрицательных отзывов')
    get_main_gramms(len(tokens_by_grade) - 1, info='Ключевые слова для положительных отзывов')
    


    
    # ## Часть 2. Тематическое моделирование [20/100]
    # 
    # 1. Постройте несколько тематических моделей коллекции документов с разным числом тем. Приведите примеры понятных
    #(интерпретируемых) тем.

    texts = [df.lemmas.iloc[i].split() for i in range(len(df))]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]


    lsi = lsimodel.LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=3)
    print('\nТемы при num_topics=3:\n')
    for topic in lsi.show_topics(3):
        print(topic, '\n')

    lsi = lsimodel.LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10)
    print('\nТемы при num_topics=10:\n')
    for topic in lsi.show_topics(10):
        print(topic, '\n')

    #Интерпретация тем:
    #(9, '0.444*"курс" + 0.277*"страховка" + 0.261*"валюта" + 0.223*"доллар" + 0.181*"рубль" + 0.176*"евро"
    # + 0.175*"комиссия" + -0.140*"заявление" + 0.139*"страхование" + -0.123*"карта"')
    #валютообменные операции с рублем, долларом и евро, покупка валюты по определенному курсу, покупка валюты это страховка от падения рубля

    #(7, '0.437*"банкомат" + -0.380*"кошелек" + -0.247*"киви" + 0.165*"сбербанк" + -0.132*"заблокировать" + -0.124*"идентификация"
    # + 0.121*"заявление" + 0.121*"купюра" + 0.115*"очередь" + 0.114*"отделение"')
    # проведение операций через банкомат сбербанка и электронный кошелек киви может быть заблокировано при проблеме с идентификацией клиента
    #проблема может быть решена через подачу заявления в любом отделении

    #(3, '-0.306*"кредит" + -0.283*"страховка" + 0.260*"номер" + -0.184*"страхование" + -0.182*"договор" + -0.169*"сумма"
    #+ 0.166*"звонок" + -0.160*"погашение" + -0.146*"платеж" + 0.142*"телефон"')
    #кредит на определенную сумму застрахован путем заключения договора, звонок  по телефону прояснит порядок погашения платежей

    #2, '0.478*"вклад" + -0.290*"карта" + 0.162*"очередь" + -0.155*"банкомат" + -0.151*"сбербанк"
    #+ 0.131*"открывать" + -0.127*"средство" + -0.126*"руб" + -0.118*"счет" + -0.112*"операция"')
    #открытие вклада, карты или счета в сбербанке в рублях для проведения операций можно сделать, отстояв очередь в отделении,
    #ряд операций можно сделать через банкомат


    # 2. Найдите темы, в которых упомянуты конкретные банки (Сбербанк, ВТБ, другой банк). Можете ли вы их
    #прокомментировать / объяснить?
    
    # Ответ:

    #(5, '-0.219*"карта" + 0.211*"кошелек" + 0.204*"страховка" + 0.181*"сбербанк" + 0.179*"страхование"
    #+ 0.162*"заявление" + 0.158*"документ" + -0.157*"задолженность" + -0.151*"лимит" + 0.148*"заявка"') 
    
    
    # Комментарий:
    # Тема на подачу заявления, заявки, документов для получения кредитной карты (с лимитом задолжности)
    # со страховкой в Сбербанке.
    









    # ## Часть 3. Классификация текстов [40/100]
    # 
    # Сформулируем для простоты задачу бинарной классификации: будем классифицировать на два класса,
    #то есть, различать резко отрицательные отзывы (с оценкой 1) и положительные отзывы (с оценкой 5). 
    # 
    # 1.  Составьте обучающее и тестовое множество: выберите из всего набора данных N1 отзывов с оценкой 1 и
    # N2 отзывов с оценкой 5 (значение N1 и N2 – на ваше усмотрение).
    # Используйте ```sklearn.model_selection.train_test_split``` для разделения множества
    # отобранных документов на обучающее и тестовое.

    # 2. Используйте любой известный вам алгоритм классификации текстов для решения задачи
    # и получите baseline. Сравните разные варианты векторизации текста: использование только униграм,
    # пар или троек слов или с использованием символьных $n$-грам. 
    # 3. Сравните, как изменяется качество решения задачи при использовании скрытых тем в качестве признаков:
    # * 1-ый вариант: $tf-idf$ преобразование (```sklearn.feature_extraction.text.TfidfTransformer```)
    #  и сингулярное разложение (оно же – латентый семантический анализ) (```sklearn.decomposition.TruncatedSVD```), 
    # * 2-ой вариант: тематические модели LDA (```sklearn.decomposition.LatentDirichletAllocation```). 
    # 
    # Используйте accuracy и F-measure для оценки качества классификации. 
    # 
    # Ниже написан примерный Pipeline для классификации текстов. 
    # 
    # Эта часть задания может быть сделана с использованием sklearn.

   

    N1 = 8000  # кол-во отзывов с оценкой 1
    N2 = 8000  # кол-вот отзывов с оценкой 2
        
    N1 = min(N1, df.rating_grade.value_counts()[1.0])  
    N2 = min(N2, df.rating_grade.value_counts()[5.0]) 
    
    df1 = df[df.rating_grade == 1].sample(N1)
    df2 = df[df.rating_grade == 5].sample(N2)
    
    target1 = df1['rating_grade']
    del df1['rating_grade']

    target2 = df2['rating_grade']
    del df2['rating_grade']

    X = pd.concat([df1, df2])

    y = pd.concat([target1, target2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def predict_clf(clf, text):
        print('\n')
        print(text)

        clf.fit(X_train.lemmas, y_train)

        predictions = clf.predict(X_test.lemmas)

        print("Precision: {0:6.2f}".format(precision_score(y_test, predictions, average='macro')))
        print("Recall: {0:6.2f}".format(recall_score(y_test, predictions, average='macro')))
        print("F1-measure: {0:6.2f}".format(f1_score(y_test, predictions, average='macro')))
        print("Accuracy: {0:6.2f}".format(accuracy_score(y_test, predictions)))
        print(classification_report(y_test, predictions))

        print('\n')

    def classifier(ngram_range):
        print('\n\n')
        print('ngram_range = ', ngram_range)
        #ngram_range = {4,6}
        predict_clf(
            clf = Pipeline([ 
                ('vect', CountVectorizer(analyzer = 'char', ngram_range=ngram_range)), 
                ('tfidf', TfidfTransformer()), 
                ('tm', TruncatedSVD(n_components=30, random_state=57)),
                ('clf', RandomForestClassifier(n_estimators=200, max_depth=30))
            ]),
            text='Для TfidfTransformer и TruncatedSVD'
        )

        predict_clf(
            clf = Pipeline([ 
                ('vect', CountVectorizer(analyzer = 'char', ngram_range=ngram_range)), 
                ('lda', LatentDirichletAllocation(n_components=30, max_iter=25)),
                ('clf', RandomForestClassifier(n_estimators=200, max_depth=30))
            ]),
            text='Для LatentDirichletAllocation'
        )


    classifier((1, 1))
    classifier((2, 2))
    classifier((3, 3))
    classifier((1, 3))
    classifier((3, 6))

# Сравнение разныx вариантов векторизации текста: использование только униграм, пар, троек, 1-3, 3-6 символьных n-грамм
# Униграммы показывают худший результат: precision=0.77 для первого варианта (TfidfTransformer и TruncatedSVD)
# и precision=0.75 для второго варианта (LatentDirichletAllocation).
# Биграммы показывают более хороший результат - 0.85 и 0.86 соответственно.
# Триграммы показывают результат лучше чем биграммы - 0.90 и 0.91 соответственно.

# В диапазоне от 1 до 3 символов показатель precision=0.87, что лучше чем показатель биграмм и хуже, чем показатель триграмм:
# для TfidfTransformer и TruncatedSVD. Для LatentDirichletAllocation этот показатель 0.91.

# Для диапазона от 3 до 6 символов показатель precision=0.91 и 0.92.




if __name__ == '__main__':
    df = pd.read_json(r'banki_responses.json', lines=True)
    df = df.iloc[:15]
    def words_only(text):
        return ' '.join(re.findall(r'[А-Яа-я]+', text)).strip()
    df['text'] = df['text'].apply(words_only)
    main(df)








