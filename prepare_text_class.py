import re
import nltk.stem as st


class TextPrepareClass:
    '''
    '''
    def __init__(self, inp_stopwords, inp_phones):

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
