FROM python:3

RUN pip install onnxruntime
RUN pip install scikit-learn
RUN pip install nltk

RUN mkdir -p /usr/src/app/nltk/corpora/stopwords
RUN mkdir -p /usr/src/app/nltk/corpora/wordnet
RUN mkdir -p /usr/src/app/nltk/stemmers/rslp

WORKDIR /usr/src/app

COPY ./corpora/stopwords/* /usr/src/app/nltk/corpora/stopwords
COPY ./corpora/wordnet/* /usr/src/app/nltk/corpora/wordnet
COPY ./stemmers/rslp/* /usr/src/app/nltk/stemmers/rslp

COPY ./models/* ./

COPY prepare_text_class.py ./
COPY server_docker.py ./


CMD [ "python" , "server_docker.py"]
