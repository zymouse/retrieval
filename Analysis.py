#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time : 2019/3/14 0014

import os
import jieba
from TrainWords import analysis
import re

"""
分析长文本数据,获取权重.
"""

Numeral = '[一二三四五六七八九十百千万亿兆0-9]*'
Headline = r'【[\u4e00-\u9fa5]*】'
Subheading = r'[一二三四五六七八九十、]+[\u4e00-\u9fa5]*'
num_rg = re.compile(Numeral)  # 数词正则表达式
headline_rg = re.compile(Headline)  # 标题的正则表达式
subheading_rg = re.compile(Subheading)  # 小标题的正则表达式

_All_Weight = {}  # 总的权重


def reader_text(dirs):
    """
     读文本数据,存放在列表中
    :return: 列表
    """
    file_text = []
    for root, dirs, files in os.walk(dirs):
        for file in files:
            with open(root + os.sep + file) as f:
                file_text.append(f.read())
    return file_text


def weight(file):
    # 领域词算法
    neologism_words = analysis(file, 4, 2, 0.0001, 100, 0.1, True)
    for k, _ in neologism_words.items():
        _All_Weight.setdefault(k, 5)


def long_participle(long_text):
    """
    拆分词语的顺序:
    1 大标题 10
    2 小标题 8
    3 摘要 6
    4 领域词 5
    5 名词 3
    6 数量词 1
    7 其他 1
    :param long_text: 长文本
    """
    weight(long_text)  # 领域词

    abstract = long_text.split('\n')[0]  # 摘要
    if abstract:
        a = {k: 6 for k in jieba.cut(abstract) if len(k) > 1}
        _All_Weight.update(**a)

    subs = subheading_rg.findall(long_text)  # 小标题
    if subs:
        su = {s: 8 for sub in subs for s in jieba.cut(sub) if len(s) > 1}
        _All_Weight.update(**su)

    if headline_rg.match(long_text):  # 大标题词语
        headline = headline_rg.match(long_text).group()[1:-1]
        h = {k: 10 for k in jieba.cut(headline) if len(k) > 1}
        _All_Weight.update(**h)

    for jb in jieba.cut(long_text):
        has_num = num_rg.match(jb).group()  # 数词
        if has_num:
            _All_Weight.setdefault(has_num, 1)
        elif len(jb) >= 2:  # 名词
            _All_Weight.setdefault(jb, 3)
        else:  # 什么都不是的词语
            _All_Weight.setdefault(jb, 1)

    return _All_Weight
