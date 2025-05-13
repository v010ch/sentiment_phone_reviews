#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[38]:


import os
#import pickle as pkl
import dill as pkl

import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import re


# In[27]:


import onnx


# In[ ]:





# ## Convert
# ## Optimize

# In[ ]:





# In[ ]:





# In[28]:


DIR_MODELS = os.path.join(os.getcwd(), 'models')

with open(os.path.join(DIR_MODELS, 'tfidf_lr_model.pkl'), 'rb') as f:
    tfidf_lr_model = pkl.load(f)
    
with open(os.path.join(DIR_MODELS, 'tfidf_lr_token.pkl'), 'rb') as f:
    tfidf_lr_token = pkl.load(f)
# In[41]:


with open(os.path.join(DIR_MODELS, 'tfidf_lr_model.pkl'), 'rb') as f:
    tfidf_lr_model = pkl.load(f)


# Проверю, что работает после загрузки

# In[42]:


assert tfidf_lr_model.predict('Это был бы хороший комментарий, если бы я не пытался написать плохой'), 1
assert tfidf_lr_model.predict('А это уж точно ужасный токсичный и глупый комментарий'), 1
assert tfidf_lr_model.predict('Абсолютно позитивный комметарий'), 1


# In[43]:


tfidf_lr_model.predict('Абсолютно позитивный комметарий')


# In[44]:


tfidf_lr_model.predict('А это уж точно ужасный токсичный и глупый комментарий')


# In[46]:


model = tfidf_lr_model.get_model()
tokenizer = tfidf_lr_model.get_tokenizer()


# In[ ]:





# In[ ]:


# Convert into ONNX format

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())


# In[56]:


len(tokenizer.vocabulary_)


# In[57]:


#dir(tokenizer)


# In[ ]:


get_ipython().getoutput('!!!!!!!!!!!!!!!!!!!!')
model = skl2onnx.to_onnx(logreg, X.astype(np.float32))
model = onnx.load(name)
rep = backend.prepare(model, 'CPU')
label, proba = rep.run(x)
get_ipython().getoutput('!!!!!!!!!!!!!!!!!!!!')


# In[ ]:





# In[62]:


get_ipython().run_cell_magic('time', '', "initial_type = [('float_input', FloatTensorType([None, len(tokenizer.vocabulary_)]))]\nonx = convert_sklearn(model, initial_types=initial_type)\n")


# In[67]:


with open(os.path.join(DIR_MODELS, 'tfidf_lr_model.onnx'), 'wb') as f:
    f.write(onx.SerializeToString())


# In[64]:


type(onx)


# In[66]:


import onnxruntime as rt


# In[ ]:





# In[105]:


sess = rt.InferenceSession(os.path.join(DIR_MODELS, 'tfidf_lr_model.onnx'))


# In[106]:


input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


# In[113]:


#tt = tokenizer.transform(['Это был бы хороший комментарий, если бы я не пытался написать плохой'])
tt = tokenizer.transform(['Ужаснейшее качество, экран так вообще говно'])


# In[ ]:





# In[114]:


#pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
pred_onx = sess.run([label_name], {input_name: tt.todense().tolist()})


# In[115]:


pred_onx[0][0]


# In[ ]:





# In[104]:


from onnx.reference import ReferenceEvaluator

#sess = ReferenceEvaluator("model.onnx")
sess = ReferenceEvaluator('models/tfidf_lr_model.onnx')
#results = sess.run(None, {"X": X})
#print(results[0])  # display the first result


# In[ ]:





# In[ ]:





# In[ ]:





# In[99]:


class ModelTfidfOnnx:
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


# In[ ]:





# In[ ]:





# In[ ]:




