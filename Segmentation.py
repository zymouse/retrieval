#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time  : 2019/3/18 13:34

import jieba
from Analysis import long_participle
import math

Weight = {}


def reader_text(file=r'./data/00.txt'):
    """
    读取文件内容
    :param file: 文件地址
    :return: 将文件的每一句话放入列表中
    """
    global Weight
    line_ls = []
    with open(file, encoding='GBK')as f:
        lines = f.read()
        Weight = long_participle(lines)
        for line in lines.split('。'):
            line_ls.append(line)
    return line_ls


def weighted(lines):
    """
    将每一句话拆分进行加权值
    :param lines: 句子集合
    """
    for line in lines:
        cs = jieba.cut(line)
        yield [Weight.get(c, 0) for c in cs if len(c) > 1]


def NDCG(weight_ls):
    """计算NDCG值"""
    NDCG_ls, IDCG = [], 0
    for index, w in enumerate(sorted(Weight.values(), reverse=True), 2):  # 将所有权值进行计算值
        IDCG += (2 * w - 1) / math.log2(index)

    for ws in weighted(weight_ls):
        DCG = 0
        for index, w in enumerate(sorted(ws, reverse=True), 2):  # 将每一句话的权重进行计算
            DCG += (2 * w - 1) / math.log2(index)
        NDCG_ls.append(DCG / IDCG)  # 每一句的权重值/所有句子权重值
    return NDCG_ls


def most_similar(n):
    """
    输入一个阿拉伯数字获取文章相似的句数
    :param n: 返回最相似的n句
    """
    ls = reader_text()
    normalized = NDCG(ls)
    similar_ls = sorted(zip(ls, normalized), key=lambda x: x[1], reverse=True)
    return similar_ls[:n]


if __name__ == '__main__':
    ms = most_similar(10)
    for m in ms:
        print(m)
