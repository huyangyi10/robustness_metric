#-*- coding = utf-8 -*-
#@Time : 2021-5-29 11:41
#@Author : CollionsHu
#@File : calculate_probability_difference.py
#@Software : PyCharm

def calculate(model, data, true_label, target_label):
    prob = model.predict(data)[0]
    prob_difference = prob[target_label] - prob[true_label]
    return prob_difference