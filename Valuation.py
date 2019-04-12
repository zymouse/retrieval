#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time  : 2019/3/15 14:05

from Analysis import reader_text, long_participle
import hashlib
import jieba
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

"""
估值数据,返回四个数值[准确率,精确率,召回率,F值]
"""
# 测试集
tests = [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1]


def content_to_dict(content, features, f):
    """
    将内容转成字典格式
    :param content: 文本内容
    :param features: 特征值
    :param f: simhash的bit位数
    :return: simhash值
    """
    c = []
    for jb in jieba.cut(content):
        if len(jb) > 1:
            v = features.get(jb, 1)
            c.append((jb, v))
    return features_dict(c, f)


def hash_func(x):
    """hash算法"""
    return int(hashlib.md5(x).hexdigest(), 16)


def features_dict(features, f):
    """
    特征值字典
    :param features: 特征值
    :param f: simhash的bit位数
    :return: simhash值
    """
    v = [0] * f
    masks = [1 << i for i in range(f)]
    for feature in features:
        h = hash_func(feature[0].encode('utf-8'))
        w = feature[1]
        for i in range(f):
            v[i] += w if h & masks[i] else -w
    values = 0
    for i in range(f):
        if v[i] > 0:
            values |= masks[i]
    return values


def distance(sim_hash, another, f):
    """
    计算两个simhash的距离
    :param sim_hash: simhash值
    :param another: 另一个simhash的值
    :param f: simhash的bit位数
    :return: 海明距离
    """
    x = (sim_hash ^ another) & ((1 << f) - 1)
    value = 0
    while x:
        value += 1
        x &= x - 1
    return value


def confusion_matrix(test, forecast):
    """
    混淆矩阵
    TP—实际为正类,预测为正类
    FN—实际为正类,预测为负类
    FP—实际为负类,预测为正类
    TN—实际为负类,预测为负类
    :param test:测试集
    :param forecast:预测集
    """
    tp, fn, fp, tn = 0, 0, 0, 0
    for t, f in zip(test, forecast):
        if t == f == 1:
            tp += 1
        elif t == 1 and f == 0:
            fn += 1
        elif t == f == 0:
            tn += 1
        else:
            fp += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn)  # 准确率
    precision = tp / (tp + fp)  # 精确率
    recall = tp / (tp + fn)  # 召回率
    f_measure = (2 * tp) / (2 * tp + fp + fn)  # F得分
    return accuracy, precision, recall, f_measure


def drawing(forecast_ls, ranges):
    """画图"""
    """获取a p r f值"""
    a = [forecast[0] for forecast in forecast_ls]
    p = [forecast[1] for forecast in forecast_ls]
    r = [forecast[2] for forecast in forecast_ls]
    f = [forecast[3] for forecast in forecast_ls]
    t = np.array(ranges)
    new = np.linspace(t.min(), t.max(), 300)  # 模拟直线
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    for smooth, name in zip([a, p, r, f], ['准确率', '查全率', '查准率', 'F值']):
        power = np.array(smooth)
        power_smooth = spline(t, power, new)
        plt.plot(new, power_smooth, label=name)
    plt.xlabel('SimHash的bit位数')
    plt.ylabel('百分比%')
    plt.legend(['准确率', '查全率', '查准率', 'F值'])
    plt.title('SimHash相似性比较')
    plt.xticks(range(ranges.start, ranges.stop, 2))
    plt.show()


def start(bit_range):
    text = reader_text('./data')  # 加载文本
    weight = long_participle(text[0])  # 加载权重
    c_ms = []
    for f in bit_range:  # 遍历simhash的位数
        v0 = content_to_dict(text[0], weight, f)  # 获取文本一的距离
        forecasts = [0]  # 第一个文本距离,肯定是0(0表示相似.自己和自己肯定相似)
        one, zero = [], []
        for i in range(1, 40):
            v = content_to_dict(text[i], weight, f)
            s = distance(v0, v, f)  # 获得海明距离
            forecasts.append(s)
            if tests[i] == 1:
                one.append(s)
            else:
                zero.append(s)
        max_one, min_one, max_zero, min_zero = max(one), min(one), max(zero), min(zero)  # 分别获取相似的最大和最小
        one_set, zero_set = set(range(min_one, max_one)), set(range(min_zero, max_zero))  # 转为集合
        tup = one_set & zero_set  # 取交集
        average = sum(tup) / len(tup)  # 取平均数
        forecast = [1 if d < average + 3 else 0 for d in forecasts]  # 获取期望值
        c_m = confusion_matrix(tests, forecast)  # 进行判断,获得四个值
        print(f, c_m)  # 打印四个值数据
        c_ms.append(c_m)
    drawing(c_ms, bit_range)
