from __future__ import unicode_literals, print_function, division
from io import open  # 处理文件
import unicodedata  # 处理unicode字符相关事项
import string
import re  # 正则表达式相关
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()  # 如果您的计算机支持cuda，则优先在cuda下运行

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}  # 单词对应的在字典里的索引号
        self.word2count = {}  # 记录某一个单词在语料库里出现的次数
        self.index2word = {0: "SOS", 1: "EOS"}  # 索引对应的单词
        self.n_words = 2  # Count SOS and EOS # 语料库里拥有的单词数量

    def addSentence(self, sentence):  # 往语料库里增加一句话：扩充语料库
        for word in sentence.split(' '):  # 要增加的一句话是以空格来分割不同的单词
            self.addWord(word)  # 把单词一个个加入语料库

    def addWord(self, word):  # 把单词加入到语料库中具体要做的事情
        if word not in self.word2index:  # 对于语料库中不存在的新词
            self.word2index[word] = self.n_words  # 索引号依据先来后到的次序分配
            self.word2count[word] = 1  # 更新该次的出现次数
            self.index2word[self.n_words] = word  # 同时更新该字典
            self.n_words += 1
        else:
            self.word2count[word] += 1  # 对于已存在于语料库中的词，仅增加其出现次数。


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):  # lang1,lang2仅是字符串，代表对应的语言
    print("Reading lines...")
    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')
    # 把文本文件变为语句对列表
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 提供一个反向的操作，即原来是英文->法语，使用reverse后则为法语->英语
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):  # 作者仅对训练数据中句子长度都小于10，且以一定字符串开头的英文句子感兴趣
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):  # 从所有pairs中选出作者感兴趣的pair
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    # 构建两个语料库
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)  # 筛选感兴趣的语句对
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")  # 统计词频
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))
