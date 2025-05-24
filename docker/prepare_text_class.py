from typing import Optional

import re
import nltk.stem as st

STOPWORDS = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как',
             'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у',
             'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 
             'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь',
             'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни',
             'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
             'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может',
             'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя',
             'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз',
             'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот',
             'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь',
             'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
             'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
             'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 
             'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них',
             'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем',
             'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть',
             'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно',
             'всю', 'между'
             ]


class TextPrepareClass:
    '''
    '''
    def __init__(self, inp_phones: str,
                 inp_stopwords: Optional[list] = STOPWORDS,
                 ):

        self.__stopwords = set(inp_stopwords)
        self.__phones = inp_phones
        self.__eng_lemm = st.WordNetLemmatizer()
        self.__eng_stemm = st.RSLPStemmer()  # st.ISRIStemmer()
        self.__ru_stemm = st.snowball.SnowballStemmer('russian')

    def __clean_text(self, inp_text: str) -> str:
        '''
        '''
        return re.sub(r"\s+", ' ', 
                      re.sub(r"[\d+]", '',
                             re.sub(r"[^\w\s]", ' ', inp_text).strip()
                             )
                      )

    def __clean_names(self, inp_text):
        '''
        '''
        return re.sub(self.__phones, '', inp_text)

    def __clean_stopwords(self, inp_text):
        '''
        '''
        return [word for word in inp_text.split()
                if word not in self.__stopwords
                ]

    def __to_norm_form(self, inp_text: str) -> str:
        '''
        '''
        inp_text = [self.__eng_lemm.lemmatize(el) for el in inp_text]
        inp_text = [self.__eng_stemm.stem(el) for el in inp_text]
        inp_text = [self.__ru_stemm.stem(el) for el in inp_text]

        return ' '.join(inp_text)

    def clean_all(self, inp_text: str) -> str:
        '''
        '''
        inp_text = inp_text.lower()
        inp_text = self.__clean_text(inp_text)
        inp_text = self.__clean_names(inp_text)
        inp_text = self.__clean_stopwords(inp_text)

        inp_text = self.__to_norm_form(inp_text)

        return inp_text
