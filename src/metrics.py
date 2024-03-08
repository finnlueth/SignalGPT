import pandas as pd
import numpy as np
import sklearn.metrics
import evaluate


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
roc_auc_score_metric = evaluate.load("roc_auc", "multiclass")
matthews_correlation_metric = evaluate.load("matthews_correlation")


def batch_eval_elementwise(predictions: np.ndarray, references: np.ndarray):
    results = {}
    # print(predictions, references)
    # print(type(predictions), type(references))

    # if np.isnan(predictions).any():
    #     print('has nan')
    #     predictions = np.nan_to_num(predictions)

    argmax_predictions = predictions.argmax(axis=-1)
    references = references[:, 1:-1]
    # print(argmax_predictions.shape, references.shape)
    vals = list(
        (np.array(p)[(r != -100)], np.array(r)[(r != -100)])
        for p, r in zip(argmax_predictions.tolist(), references)
    )

    lst_pred, lst_true = zip(*vals)
    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=np.concatenate(lst_true), y_pred=np.concatenate(lst_pred)
    )

    results.update(
        {
            "accuracy_metric": np.average(
                [
                    accuracy_metric.compute(predictions=x, references=y)["accuracy"]
                    for x, y in vals
                ]
            )
        }
    )
    results.update(
        {
            "precision_metric": np.average(
                [
                    precision_metric.compute(
                        predictions=x, references=y, average="micro"
                    )["precision"]
                    for x, y in vals
                ]
            )
        }
    )
    results.update(
        {
            "recall_metric": np.average(
                [
                    recall_metric.compute(predictions=x, references=y, average="micro")[
                        "recall"
                    ]
                    for x, y in vals
                ]
            )
        }
    )
    results.update(
        {
            "f1_metric": np.average(
                [
                    f1_metric.compute(predictions=x, references=y, average="micro")[
                        "f1"
                    ]
                    for x, y in vals
                ]
            )
        }
    )
    # results.update({'roc_auc': [roc_auc_score_metric.compute(prediction_scores=x, references=y, multi_class='ovr', average=None)['roc_auc'] for x, y in zip(softmax_predictions, references)]})
    results.update(
        {
            "matthews_correlation": np.average(
                [
                    matthews_correlation_metric.compute(
                        predictions=x, references=y, average="micro"
                    )["matthews_correlation"]
                    for x, y in vals
                ]
            )
        }
    )
    results.update({"confusion_matrix": confusion_matrix})

    return results


def compute_metrics(p):
    predictions, references = p
    # if type(predictions) is tuple:
    #     predictions = predictions[0]
    # print(predictions)
    # print(type(predictions))
    # print(predictions[0].shape)
    # print(predictions[1].shape)
    # print(predictions[1])

    # print('---- eval ----')
    # print('predictions', predictions.shape, predictions)
    # print('references', references.shape, references)

    results = batch_eval_elementwise(predictions=predictions, references=references)
    return results


# def compute_metrics_crf(p):
#     predictions, references = p
#     results = {}
#     if np.isnan(predictions).any():
#         print('has nan')
#         predictions = np.nan_to_num(predictions)

#     vals = list((np.array(p)[(r != -100)], np.array(r)[(r != -100)]) for p, r in zip(predictions.tolist(), references))

#     lst_pred, lst_true = zip(*vals)
#     confusion_matrix = sklearn.metrics.confusion_matrix(y_true=np.concatenate(lst_true), y_pred=np.concatenate(lst_pred))

#     results.update({'accuracy_metric': np.average([accuracy_metric.compute(predictions=x, references=y)['accuracy'] for x, y in vals])})
#     results.update({'precision_metric': np.average([precision_metric.compute(predictions=x, references=y, average='micro')['precision'] for x, y in vals])})
#     results.update({'recall_metric': np.average([recall_metric.compute(predictions=x, references=y, average='micro')['recall'] for x, y in vals])})
#     results.update({'f1_metric': np.average([f1_metric.compute(predictions=x, references=y, average='micro')['f1'] for x, y in vals])})
#     # results.update({'roc_auc': [roc_auc_score_metric.compute(prediction_scores=x, references=y, multi_class='ovr', average=None)['roc_auc'] for x, y in zip(softmax_predictions, references)]})
#     results.update({'matthews_correlation': np.average([matthews_correlation_metric.compute(predictions=x, references=y, average='micro')['matthews_correlation'] for x, y in vals])})
#     results.update({'confusion_matrix': confusion_matrix})

#     return results


def one_vs_all_encoding(seq: list, label: str):
    return [1 if x == label else 0 for x in seq]


def encode_transitions():
    # todo
    pass