''' flask main app for phone review sentiment analysis'''
import os
from pathlib import Path
from typing import Optional
from flask import Flask, render_template, request
from sentiment_analyze_models import ModelTfIdf, ModelROBERTA

app = Flask(__name__)


setup = {'tfidf': True, 'roberta': 'base'}

print('init')
model_selected = False
model = None

# проверка можем ли использовать лемматизатор и стеммер для легковесной модели
try:
    import nltk.stem as st
    import nltk.data
    nltk.data.find('stemmers/rslp')
    setup['stem'] = True
except ImportError:
    print('lemm/stem fail')
    setup['stem'] = False


# проверка можем ли использовать среднюю и тяжелую модель
try:
    import torch
    from transformers import pipeline
    setup['roberta'] = True
except ImportError:
    print('transformers fail')
    setup['roberta'] = False


# print(os.path.join(Path.cwd().parents[0], 'models'))
# MODELS_PATH = os.path.join(Path.cwd().parents[0], 'models')


def get_model(inp_type: str):
    global setup

    if inp_type == 'tf-idf':
        return ModelTfIdf(setup, 'tfidf_lr')
    elif inp_type == 'base':
        setup['roberta'] = 'base'
        return ModelROBERTA(setup, 'cardiffnlp/twitter-roberta-base-sentiment')
    else:  # inp_type == 'large'
        setup['roberta'] = 'large'
        return ModelROBERTA(setup, 'siebert/sentiment-roberta-large-english')



@app.route("/", methods=["POST", "GET"])
def index_page(text: Optional[str] = '',
               model_type: Optional[str] = '',
               prediction_message: Optional[str] = '',
               ):
    global model_selected
    global model

    if request.method == "POST":
        if model_selected: 
            text = request.form["text"]
        else:
            model_type = request.form["model_type"]
            model_selected = True
            print(model_type)
            model = get_model(model_type)
            print('model loaded')

    # различные страницы для выбора типа модели и проверки отзыва
    if model_selected:
        if text != '':
            prediction_message = model.predict_tonality(text)
        else:
            prediction_message = 'neutral'
        return render_template('input_review.html', text=text,
                               model_type=model_type,
                               prediction_message=prediction_message,
                               )

    # no model choosed. check can we use medium and large models
    if setup['roberta']:
        return render_template('index.html', text=text,
                               model_type=model_type,
                               prediction_message=prediction_message,
                               )
    else:
        return render_template('index_noroberta.html', text=text,
                               model_type=model_type,
                               prediction_message=prediction_message,
                               )


if __name__ == "__main__":
    app.run()
