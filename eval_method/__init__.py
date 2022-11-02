import numpy as np
from sklearn import metrics

from eval_method.method import Eval_method


def get_eval_method(seed,class_num):
    return Eval_method(seed, class_num)
