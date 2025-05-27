''' Файл класса для унификации очистки и подготовки текста'''

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
exclude_stopwords = set(['не', 'ничего', 'никогда', 'больше', 'много', 'нельзя',
                        # 'более', 'больше', 'много', 'хорошо', 'лучше',
                         ])


class TextPrepareClass:
    '''
    Класс для унификации очистки и предобработки входных текстов.
    '''
    def __init__(self, inp_phones: str,
                 inp_stopwords: Optional[list] = STOPWORDS,
                 ):
        '''
        args:
            inp_phones: str - перечисление "стопслов" - именований телефонов
                        для очистки в формате для  regexp.sub
            inp_stopwords: list - стопслова, опционально при отсутствии
                           использует nltk.stopwords.words('russian')
        '''
        for el in inp_stopwords:
            if el in inp_stopwords:
                inp_stopwords.remove(el)

        self.__stopwords = set(inp_stopwords)
        self.__phones = inp_phones
        self.__eng_lemm = st.WordNetLemmatizer()
        self.__eng_stemm = st.RSLPStemmer()  # st.ISRIStemmer()
        self.__ru_stemm = st.snowball.SnowballStemmer('russian')

        self.__for_coloring = False

    def __clean_sequence(self, inp_text: str) -> str:
        '''
        Метод для очистки входного слова/текста от вспомогательных и 
        не нужных символов
        args:
            inp_text: str - входное слово/текст
        returns:
            str - очищенное слово/текст
        '''
        return re.sub(r"\s+", ' ', 
                      re.sub(r"[\d+]", ' ',
                             re.sub(r"[^\w\s]", ' ', inp_text)#.strip()
                             )
                      )

    def __clean_text(self, inp_text: str,) -> list[str]:
        '''
        Метод очистки входного текста от вспомогательных и не нужных символов.
        При этом для окрашивания текста существует возможность сохранить 
        количество слов, заменяя пустые после очистки слова (например только 
        цифры) - спецсимволом.
        args:
            inp_text: str - входной текст
        returns:
             list[str] - массив очищенных слов
        '''
        inp_text = [self.__clean_sequence(el) for el in inp_text.split(' ')]
        if self.__for_coloring:
            inp_text = [el if el != ' ' else '_' for el in inp_text]

        return ' '.join(inp_text)

    def __clean_names(self, inp_text: str) -> str:
        '''
        Метод очистки входного текста от стопслов - наименований телефонов
        (необходимо для минимизации переобучения на наименованиях телефонов).
        При этом для окрашивания текста существует возможность сохранить 
        количество слов, заменяя удаленные слова - спецсимволом.
        args:
            inp_text: str - входной текст
        returns:
            str - очищенный текст
        '''
        if self.__for_coloring:
            return re.sub(self.__phones, '_', inp_text)

        return re.sub(self.__phones, '', inp_text)

    def __clean_stopwords(self, inp_text: str) -> list[str]:
        '''
        Метод удаления стопслов
        args:
            inp_text: str - входной текст
        returns:
            str - очищенный текст
        returns:
        '''
        return [word for word in inp_text.split()#(' ')
                if word not in self.__stopwords
                ]

    def __to_norm_form(self, inp_text: list[str]) -> str:
        '''
            Метод приведения входящих слов (как русских, так и английских)
            к нормальной форме.
        args:
            inp_text: list[str] - массив входных слов
        returns:
            str - очищенный текст
        '''
        inp_text = [self.__eng_lemm.lemmatize(el) for el in inp_text]
        inp_text = [self.__eng_stemm.stem(el) for el in inp_text]
        inp_text = [self.__ru_stemm.stem(el) for el in inp_text]

        return ' '.join(inp_text)

    def clean_all(self, inp_text: str,
                  for_coloring: Optional[bool] = False,
                  ) -> str:
        '''
        Матод применяющий все вышеперечисленные методы очистки к входному
        тексту
        args:
            inp_text: str - входной текст
        returns:
            str - очищенный с приведенными к нормальной форме словами
        '''
        inp_text = inp_text.lower()
        inp_text = self.__clean_text(inp_text)
        inp_text = self.__clean_names(inp_text)
        inp_text = self.__clean_stopwords(inp_text)

        inp_text = self.__to_norm_form(inp_text)

        return inp_text
