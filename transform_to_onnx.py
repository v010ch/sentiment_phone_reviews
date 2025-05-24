import os
import pickle as pkl

import numpy as np
import onnxruntime as rt
from skl2onnx import to_onnx


MODEL_PATH = os.path.join('.', 'models')
DATA_PATH = os.path.join('.', 'data')

if __name__ == '__main__':
    # данные нужны, что бы предоставить трансформатору to_onnx
    # вид входных данных
    review = 'Очень хороший отзыв, вызывающий обоснованные сомнения'

    # logreg
    with open(os.path.join(MODEL_PATH, 'logreg_model.pkl'), 'rb') as fd:
        model_logreg = pkl.load(fd)
    with open(os.path.join(MODEL_PATH, 'logreg_vektorizer.pkl'), 'rb') as fd:
        vectorizer_logreg = pkl.load(fd)

    inp_data = vectorizer_logreg.transform([review])
    # print(sum(inp_data[:1].toarray()[0]))
    onx = to_onnx(model_logreg, inp_data[:1].toarray().astype(np.float32))
    with open(os.path.join(MODEL_PATH, 'logreg_model.onnx'), 'wb') as fd:
        fd.write(onx.SerializeToString())

    # проверка на работу рантайма
    sess = rt.InferenceSession(os.path.join(MODEL_PATH, 'logreg_model.onnx'),
                               providers=['CPUExecutionProvider']
                               )
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name
    pred_onx = sess.run([label_name],
                        {input_name: inp_data.toarray().astype(np.float32)}
                        )
    print(f'sklearn pred_proba: {pred_onx[0]}')

    # catboost
    with open(os.path.join(MODEL_PATH, 'catboost_model.pkl'), 'rb') as fd:
        model_cb = pkl.load(fd)
    with open(os.path.join(MODEL_PATH, 'catboost_vektorizer.pkl'), 'rb') as fd:
        vectorizer_cb = pkl.load(fd)

    model_cb.save_model(
        os.path.join(MODEL_PATH, 'catboost_model.onnx'),
        format='onnx',
        export_parameters={
            'onnx_domain': 'ai.catboost',
            'onnx_model_version': 1,
            'onnx_doc_string': 'CatBoostClassifier phone review sentiment',
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
