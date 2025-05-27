''' Преобразование моделей logreg и catboost в onnx формат'''

import os
import pickle as pkl

import numpy as np
import onnxruntime as rt
from skl2onnx import to_onnx


MODEL_PATH = os.path.join('.', 'models')
DATA_PATH = os.path.join('.', 'data')


if __name__ == '__main__':
    # данные нужны, что бы предоставить трансформеру to_onnx
    # вид входных данных
    review = 'Очень хороший отзыв, вызывающий обоснованные сомнения'

    # logreg
    with open(os.path.join(MODEL_PATH, 'logreg_model.pkl'), 'rb') as fd:
        model_logreg = pkl.load(fd)
    with open(os.path.join(MODEL_PATH, 'logreg_vektorizer.pkl'), 'rb') as fd:
        vectorizer_logreg = pkl.load(fd)

    # функция to_onnx ничего не знает о формате входных данных.
    # подготаовливаем на примере и показываем ей (to_onnx)
    inp_data_form = vectorizer_logreg.transform([review])[:1].toarray()

    onx = to_onnx(model_logreg, inp_data_form.astype(np.float32))
    with open(os.path.join(MODEL_PATH, 'logreg_model.onnx'), 'wb') as fd:
        fd.write(onx.SerializeToString())

    # проверка на работу рантайма
    sess = rt.InferenceSession(os.path.join(MODEL_PATH, 'logreg_model.onnx'),
                               providers=['CPUExecutionProvider']
                               )
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name
    pred_onx = sess.run([label_name],
                        {input_name: inp_data_form.astype(np.float32)}
                        )
    print(f'sklearn logreg pred_proba: {pred_onx[0]}')

    # catboost
    with open(os.path.join(MODEL_PATH, 'catboost_model.pkl'), 'rb') as fd:
        model_cb = pkl.load(fd)
    with open(os.path.join(MODEL_PATH, 'catboost_vektorizer.pkl'), 'rb') as fd:
        vectorizer_cb = pkl.load(fd)

    # с catboost легче - просто пересохраняем модель в формате onnx
    model_cb.save_model(
        os.path.join(MODEL_PATH, 'catboost_model.onnx'),
        format='onnx',
        export_parameters={
            'onnx_domain': 'ai.catboost',
            'onnx_model_version': 1,
            'onnx_doc_string': 'CatBoostClassifier for phone review sentiment',
            'onnx_graph_name': ''
        }
    )

    # проверка на работу рантайма
    sess = rt.InferenceSession(os.path.join(MODEL_PATH, 'catboost_model.onnx'),
                               providers=['CPUExecutionProvider'],
                               )
    inp_data = vectorizer_cb.transform([review]).toarray()
    probabilities = sess.run(['probabilities'],
                             {'features': inp_data.astype(np.float32)}
                             )
    print(f'catboost pred_proba: {probabilities[0]}')

    print('done')
