#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Kagle inclass https://www.kaggle.com/c/simplesentiment/overview

# In[1]:


from itertools import product
from collections import Counter
import os
from pathlib import Path
import re

import dill as pkl
import numpy as np
import nltk
from nltk.corpus import stopwords
import nltk.stem as st
import pandas as pd
import plotly.graph_objects as go
# from tqdm import tqdm


# In[3]:


nltk.download('omw-1.4')


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression#, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier


# In[37]:





# In[ ]:





# # Задаю переменные

# In[5]:


PATH_DATA = os.path.join(Path.cwd(), 'data')
PATH_SUBM = os.path.join(Path.cwd(), 'submissions')
PATH_MODL = os.path.join(Path.cwd(), 'models')


# In[ ]:





# In[ ]:





# # Загрузка, очистка и преобразование данных

# In[6]:


df = pd.read_csv(os.path.join(PATH_DATA, 'ru_train.csv'), )
df.shape


# 4 и 5 - positive   
# 1, 2, 3 - negative   

# In[8]:


df['target'] = df.rating.apply(lambda x: 0 if int(x) >= 4 else 1)
df['target'].value_counts(normalize=True)


# In[10]:


stop_words = stopwords.words('russian')

clean_text = lambda x: re.sub(r"\s+", ' ', 
                              re.sub(r"[\d+]", '',
                                     re.sub(r"[^\w\s]", ' ', x.lower()).strip()
                                    )
                             )
# отзывы содержат наименования телефонов, очищаю от них и рускоязычных 
# вариантов названий, т.к. они могут привести к переобучению.
phones = r'\b(samsung|galaxy|xiaomi|iphon|redmi|note|honor|huawei|apple|' +\
          'nokia|meizu|google|самсунг|айфон|mi|lenovo|lg|redme|asus|vivo|' +\
          'zte|helio|mediatek|oppo|htc|pixel|xperia|fly|realme|zenfone|' +\
          'alcatel|blade|philips|touch|lumia|oneplus|motorola|inoi|red|neo|' +\
          'moto|panasonic|band|honnor|bbk|vertex|lafleur|xiomi|редми|хонор|' +\
          'ноки|хуаве|мейзу|асус|галакси|иной|гэлакси|хонор)(|[a-zа-я]+)\b'
clean_names = lambda x: re.sub(phones, '', x)
clean_stopwords = lambda x: ' '.join([word for word in x.split() if word not in stop_words])

# приведение к начальным формам
lemm = st.WordNetLemmatizer()
lem_text = lambda x: ' '.join([lemm.lemmatize(el) for el in x.split()])

#stemm = st.ISRIStemmer()
stemm = st.RSLPStemmer()
stem_text = lambda x: ' '.join([stemm.stem(el) for el in x.split()])


# In[11]:


get_ipython().run_cell_magic('time', '', "df['text_cl'] = df.review.map(clean_text)\ndf['text_cl'] = df.text_cl.map(clean_names)\ndf['text_cl'] = df.text_cl.map(clean_stopwords)\ndf['text_cl'] = df.text_cl.map(lem_text)\ndf['text_cl'] = df.text_cl.map(stem_text)\n")


# Перемешиваю отзывы

# In[14]:


df = df.sample(frac=1).reset_index(drop=True)
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.head(3)


# In[ ]:





# In[ ]:





# # Создание и сохранение модели и токенайзера для инференса и демонстрации на flask

# Очищенные отзывы векторизую через tf-idf.   
# Полученные векторы в LogReg для подбора параметров.

# In[16]:


grid_pipe = Pipeline([('vector', TfidfVectorizer(analyzer = 'char_wb')),
                     ('model', LogisticRegression(solver = 'liblinear',  random_state = 111111))
                     ])

params_pipe = {'vector__max_features': [768, 1024], #256, 512, 768, 
               'vector__ngram_range': [(2, 3), (3, 4) ], #(2, 4)
               'vector__min_df': [0.2, 0.3], #0.5
               'vector__max_df': [0.75, 0.8], #0.7, 1.0
               'model__penalty': ('l2'), # 'l1'
               'model__class_weight': ['balanced'],   #{0: 0.5, 1: 0.5}, {0: 0.5894081632653061, 1: 0.41059183673469385},
               'model__max_iter': [ 20, 100], #10, 50],
               'model__C': [2, 3, 4]
               }


# In[17]:


get_ipython().run_cell_magic('time', '', "clf = GridSearchCV(grid_pipe, params_pipe, n_jobs=-1, cv=5,\n                   #scoring = [('roc-auc', roc_auc_score), ('acc', accuracy_score)]\n                   scoring = ['roc_auc', 'accuracy'],\n                   refit='roc_auc',\n                   verbose=100,\n                  )\nclf.fit(df.text_cl, df.target)\n")


# In[18]:


clf.best_score_


# In[19]:


clf.best_params_


# In[ ]:





# Результаты на всех возможных комбинациях заданных параметров

# In[24]:


res = pd.DataFrame(clf.cv_results_['params'])
res['mean_test_roc_auc'] = clf.cv_results_['mean_test_roc_auc']
res['mean_test_accuracy'] = clf.cv_results_['mean_test_accuracy']
res['mean_sum'] = (clf.cv_results_['mean_test_accuracy'] + clf.cv_results_['mean_test_roc_auc'])/2


# In[25]:


res.to_csv(os.path.join(PATH_DATA, 'grid2_res.csv'), index=False)
res1 = pd.read_csv(os.path.join(PATH_DATA, 'grid1_res.csv'))


# In[26]:


res.sort_values(by='mean_test_roc_auc', ascending=False).head(20)


# In[37]:


res.sort_values(by='mean_test_accuracy', ascending=False).head(20)


# In[38]:


res.sort_values(by='mean_sum', ascending=False).head(20)


# In[28]:


res.query('vector__max_features == 768').sort_values(by='mean_sum', ascending=False).head(20)


# In[ ]:





# In[34]:


res_group = res1.groupby('vector__max_features')['mean_test_roc_auc', 'mean_test_accuracy', 'mean_sum'].agg(['min', 'mean', 'max'])

res_group.columns = ['mean_test_roc_auc_min', 'mean_test_roc_auc_mean', 'mean_test_roc_auc_max',
                     'mean_test_accuracy_min', 'mean_test_accuracy_mean', 'mean_test_accuracy_max',
                     'mean_sum_min', 'mean_sum_mean', 'mean_sum_max'
                    ]


# In[35]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = res_group.index,
                         y=res_group['mean_test_roc_auc_max'],
                         name='mean_test_roc_auc_max')
             )
fig.add_trace(go.Scatter(x = res_group.index,
                         y=res_group['mean_test_roc_auc_mean'], 
                         name='mean_test_roc_auc_mean',
                         line=dict(dash='dot'))
             )
fig.add_trace(go.Scatter(x = res_group.index,
                         y=res_group['mean_test_accuracy_max'],
                         name='mean_test_accuracy_max')
             )
fig.add_trace(go.Scatter(x = res_group.index,
                         y=res_group['mean_test_accuracy_mean'], 
                         name='mean_test_accuracy_mean',
                         line=dict(dash='dot'))
             )
fig.add_trace(go.Scatter(x = res_group.index,
                         y=res_group['mean_sum_max'],
                         name='mean_sum_max')
             )
fig.add_trace(go.Scatter(x=res_group.index,
                         y=res_group['mean_sum_mean'], 
                         name='mean_sum_mean',
                         line=dict(dash='dot'))
             )
fig.update_layout(yaxis_range=[0.75, 0.95])
fig.show()


# In[ ]:





# In[ ]:





# Обучаю модель на лучших параметрах

# In[31]:


get_ipython().run_cell_magic('time', '', "vectorizer = TfidfVectorizer(analyzer = 'char_wb',\n                             ngram_range=clf.best_params_['vector__ngram_range'],\n                             max_df=clf.best_params_['vector__max_df'],\n                             min_df=clf.best_params_['vector__min_df'],\n                             max_features=clf.best_params_['vector__max_features'],\n                            )\nvectorizer.fit(df.text_cl)\ntrain = vectorizer.transform(df.text_cl)\n\nmodel = LogisticRegression(penalty=clf.best_params_['model__penalty'],\n                           solver='liblinear',\n                           C=clf.best_params_['model__C'],\n                           class_weight=clf.best_params_['model__class_weight'],\n                           max_iter=clf.best_params_['model__max_iter'],\n                           random_state=111111, \n                          )\nmodel.fit(train, df.target)\n")


# In[ ]:





# In[40]:


get_ipython().run_cell_magic('time', '', 'model_cb = CatBoostClassifier(random_seed=1984, verbose=100)\nmodel_cb.fit(train, df.target)\n')


# In[ ]:





# In[ ]:





# ## Сохраняю

# С учетом предполагаемого применения, будет необходимо отслеживать версии    
# скриптов очистки, приведения к начальной форме и модели. Для облегчения   
# этолй задачи объеденим все в класс. При изменении версий данный класс не   
# будет занимать много места, об этом не беспокоюсь.    

# In[ ]:


class ModelTfidf:
    '''
    '''
    def __init__(self, cleaner, lemmatizer, stemmer, tokenizer, model):
        self.cleaner_ = cleaner
        self.lemmatizer_ = lemmatizer
        self.stemmer_ = stemmer
        self.tokenizer_ = tokenizer
        self.model_ = model


    def prepare_text_(self, inp_text) -> str:
        '''
        Очистка/подготовка текста.
        \params
            inp_text - не обработанный (исходный) текст, для которого необходимо определить класс/тональность
        \return    
            str - Очищенный текст
        '''
        text = self.cleaner_(inp_text)
        text =' '.join([self.lemmatizer_.lemmatize(el) for el in text.split()])
        text = ' '.join([self.stemmer_.stem(el) for el in text.split()])

        return text


    def predict(self, inp_text) -> int:
        '''
        Предсказание класса.
        \params
            inp_text - не обработанный (исходный) текст, для которого необходимо определить класс/тональность
        \return
            int - класс текста
        '''
        text = self.prepare_text_(inp_text)
        tokens = self.tokenizer_.transform([text])

        return self.model_.predict(tokens)[0]


    def get_model(self):
        '''
        Понадобилась при конвертации в onnx
        '''
        return self.model_


    def get_tokenizer(self):
        '''
        Понадобилась при конвертации в onnx
        '''
        return self.tokenizer_


# In[151]:


tfidf_lr_model = ModelTfidf(clean_text, lemm, stemm, vectorizer, model)

with open(os.path.join(PATH_MODL, 'tfidf_lr_model.pkl'), 'wb') as fd:
    pkl.dump(tfidf_lr_model, fd)

with open(os.path.join(PATH_MODL, 'tfidf_lr_model.pkl'), 'wb') as fd:
    pkl.dump(model, fd)
    
with open(os.path.join(PATH_MODL, 'tfidf_lr_token.pkl'), 'wb') as fd:
    pkl.dump(vectorizer, fd)
# In[ ]:





# In[ ]:





# # Посмотрим на результат обучения на трейне

# In[32]:


pred_train_tfidf = model.predict(train)


# In[41]:


print(roc_auc_score(df.target, pred_train_tfidf), accuracy_score(df.target, pred_train_tfidf))


# In[42]:


confusion_matrix(df.target, pred_train_tfidf)


# In[ ]:





# In[ ]:





# In[43]:


pred_train_cb = model_cb.predict(train)
print(roc_auc_score(df.target, pred_train_cb), accuracy_score(df.target, pred_train_cb))


# In[44]:


confusion_matrix(df.target, pred_train_cb)


# In[ ]:





# In[ ]:





# In[ ]:





# acc = 0.8971134020618556    
# roc-auc = 0.8881304288117554

# array([[2609,   51],    
#        [ 448, 1742]], dtype=int64)

# In[ ]:




