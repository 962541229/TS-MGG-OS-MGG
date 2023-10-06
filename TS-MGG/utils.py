# coding: UTF-8
import torch
import math
import time
import random
from datetime import timedelta
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def parseSentence(line):

    stop = stopwords.words('english')

    # 将句子转化为list，并去掉标点符号.
    text_token = CountVectorizer().build_tokenizer()(line.lower())

    # 不去标点符号
    # text_token = WordPunctTokenizer().tokenize(line.lower())

    # 词干化
    lmtzr = WordNetLemmatizer()
    text_token = [lmtzr.lemmatize(w) for w in text_token]

    # 停用词
    # text_token = [i for i in text_token if i not in stop]

    return text_token


def read_file(config, is_train):
    """读取文件数据"""
    contents_raws, contents_cleans, labels, train_indexs = [], [], [], []

    dataset_path = config.dataset_dir
    dataset_raw_text = config.dataset_text
    dataset_clean_text = config.dataset_clean
    with open(dataset_path, 'r', errors='ignore') as f:
        for line in f:
            _, train_index, label = line.strip().split('\t')
            labels.append(label)
            train_indexs.append(train_index)

    with open(dataset_raw_text, 'r', errors='ignore') as f:
        for line in f:
            contents_raws.append(parseSentence(line.strip()))
    with open(dataset_clean_text, 'r', errors='ignore') as f:
        for line in f:
            contents_cleans.append(parseSentence(line.strip()))
    if is_train == "train":
        contents_raws = [contents_raws[i] for i in range(len(train_indexs)) if 'train' in train_indexs[i]]
        contents_cleans = [contents_cleans[i] for i in range(len(train_indexs)) if 'train' in train_indexs[i]]
        labels = [labels[i] for i in range(len(train_indexs)) if 'train' in train_indexs[i]]
    elif is_train == "test":
        contents_raws = [contents_raws[i] for i in range(len(train_indexs)) if 'test' in train_indexs[i]]
        contents_cleans = [contents_cleans[i] for i in range(len(train_indexs)) if 'test' in train_indexs[i]]
        labels = [labels[i] for i in range(len(train_indexs)) if 'test' in train_indexs[i]]
    elif is_train == "all":
        contents_raws = [contents_raws[i] for i in range(len(train_indexs))]
        contents_cleans = [contents_cleans[i] for i in range(len(train_indexs))]
        labels = [labels[i] for i in range(len(train_indexs))]

    return contents_raws, contents_cleans, labels


# def process_dataset(config, is_train):
#     content_raws, content_cleans, labels = read_file(config, is_train)
#
#     data_id, text_lengths, label_id = [], [], []
#     for i in tqdm(range(len(labels))):
#         content = []
#         if config.is_raw_text:
#             for x in config.words:
#                 content.append(1) if x in content_raws[i] else content.append(0)
#             seq_len = sum(content)
#         else:
#             for x in config.words:
#                 content.append(1) if x in content_cleans[i] else content.append(0)
#             seq_len = sum(content)
#
#         data_id.append(content)
#         label_id.append(config.cat_to_id[labels[i]])
#         text_lengths.append([seq_len])
#     #
#     # # 使用keras提供的pad_sequences来将文本pad为固定长度
#     word_documents_adjmatrix = np.array(data_id)
#     y = np.array(label_id)
#     length_id = np.array(text_lengths)
#
#     return word_documents_adjmatrix, y, length_id


def process_dataset_sequence(config, is_train):
    content_raws, content_cleans, labels = read_file(config, is_train)

    data_id, text_lengths, label_id = [], [], []
    for i in tqdm(range(len(labels))):
        if config.is_raw_text:
            content = [config.word_to_id[x] for x in content_raws[i] if x in config.word_to_id]
        else:
            content = [config.word_to_id[x] for x in content_cleans[i] if x in config.word_to_id]

        if len(content) < config.max_length:
            seq_len = len(content)
            content.extend([len(config.words)] * (config.max_length - len(content)))
        else:
            content = content[:config.max_length]
            seq_len = config.max_length

        if seq_len > 0:
            data_id.append(content)
            label_id.append(config.cat_to_id[labels[i]])
            text_lengths.append([seq_len])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = np.array(data_id)
    y = np.array(label_id)
    length_id = np.array(text_lengths)

    return x_pad, y, length_id


def split_train_val(train_mask, y, train_size=0.9):
    # 利用sklearn包函数train_test_split实现从train数据集中切分部分数据作为val数据集，且保证各个类别的样本平衡
    train_indexs = []
    for i in range(y.shape[0]):
        if train_mask[i] == 1:
            train_indexs.append(i)
    train_y = y[train_indexs]

    X_real_train, X_val, y_real_train, y_val = train_test_split(train_indexs, train_y, train_size=train_size, stratify=train_y)
    # stratify的作用是：保持测试集与整个数据集里result的数据分类比例一致
    # print(X_real_train)
    # print(X_val)

    real_train_mask = np.zeros(y.shape[0])
    val_mask = np.zeros(y.shape[0])
    real_train_mask[X_real_train] = 1
    val_mask[X_val] = 1

    return torch.tensor(val_mask, dtype=torch.bool), torch.tensor(real_train_mask, dtype=torch.bool)


def split_train_val_own(train_mask, y, num_classes, train_size=0.9):
    train_indexs = []
    for i in range(y.shape[0]):
        if train_mask[i] == 1:
            train_indexs.append(i)
    random.shuffle(train_indexs)

    real_train_index, val_index = [], []

    split_classes = [[] for i in range(num_classes)]

    for i in range(len(train_indexs)):
        split_classes[y[train_indexs[i]]].append(train_indexs[i])

    for i in range(len(split_classes)):
        index_list = split_classes[i]
        real_train_index.extend(index_list[: math.ceil(len(split_classes[i])*train_size)])
        val_index.extend(index_list[math.ceil(len(split_classes[i])*train_size):])

    real_train_mask = np.zeros(y.shape[0])
    val_mask = np.zeros(y.shape[0])
    real_train_mask[real_train_index] = 1
    val_mask[val_index] = 1

    print('实际训练集数量：', np.sum(real_train_mask))
    print('实际验证集数量：', np.sum(val_mask))

    return torch.tensor(val_mask, dtype=torch.bool), torch.tensor(real_train_mask, dtype=torch.bool)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    from Dataset.R52.config import Config
    config = Config()

    word_documents_adjmatrix_train, y_train, length_id_train = process_dataset_sequence(config, 'train')
    train_indexs = np.zeros(y_train.shape[0])

    word_documents_adjmatrix_test, y_test, length_id_test = process_dataset_sequence(config, 'test')
    test_indexs = np.ones(y_test.shape[0])

    labels_total = np.concatenate((y_train, y_test)).astype(np.int)
    word_documents_adjmatrix_total = np.concatenate((word_documents_adjmatrix_train, word_documents_adjmatrix_test), axis=0)
    length_id_total = np.concatenate((length_id_train, length_id_test)).astype(np.float)

    test_mask = np.concatenate((train_indexs, test_indexs)).astype(np.bool)
    train_mask = ~test_mask

    # val_mask, real_train_mask = split_train_val(train_mask, labels_total)
    val_mask, real_train_mask = split_train_val_own(train_mask, labels_total, config.num_classes)
