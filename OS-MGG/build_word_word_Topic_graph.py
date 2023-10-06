import os
import dgl
import torch
import math
import numpy as np
import scipy.sparse as sp
from gensim.models.ldamodel import LdaModel
from utils import process_dataset_sequence
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def train(config, random_state=None):
    word_documents_adjmatrix_train, y_train, length_id_train = process_dataset_sequence(config, 'train')
    train_indexs = np.zeros(y_train.shape[0])
    word_documents_adjmatrix_test, y_test, length_id_test = process_dataset_sequence(config, 'test')
    test_indexs = np.ones(y_test.shape[0])

    labels_total = np.concatenate((y_train, y_test)).astype(np.int)
    word_documents_adjmatrix_total = np.concatenate((word_documents_adjmatrix_train, word_documents_adjmatrix_test), axis=0)
    length_id_total = np.concatenate((length_id_train, length_id_test)).astype(np.int)

    document_total = []
    for i in range(word_documents_adjmatrix_total.shape[0]):
        words = word_documents_adjmatrix_total[i][0:length_id_total[i][0]]
        words_str = [str(w) for w in words]
        document_total.append(words_str)

    dictionary = Dictionary(document_total)
    dictionary.filter_extremes(no_below=3,  # 去掉出现次数低于no_below的
                               no_above=0.8,  # 去掉出现在50%以上文章中的单词
                               keep_n=100000)  # 在1，2的基础上保留3000个高频单词

    print("LDA主题模型的词表大小：", str(len(dictionary.token2id)))

    corpus = [dictionary.doc2bow(text) for text in document_total]

    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=config.num_topics,  # 主题数量
                         passes=config.lda_epoch,  # 类似于在机器学习中常见的epoch，也就是训练了多少轮
                         random_state=random_state)  # 一个随机状态对象或生成一个随机状态对象的种子。用于再现性。(保持每次模型训练的一致性)
    lda_model.save(config.topic_model_dir)

    ldacm = CoherenceModel(model=lda_model, texts=document_total, corpus=corpus, dictionary=dictionary, coherence='c_v')
    print("模型Coherence：", ldacm.get_coherence())
    ldapl = lda_model.log_perplexity(corpus)
    print("模型perplexity：", str(ldapl))

    w = open('Dataset/' + config.dataset + '/LDA/' + config.dataset + '.topic_documents.txt', 'w')
    for j in range(1, 501):
        for i in range(lda_model.num_topics):
            word_values = lda_model.show_topic(i, topn=j)
            word_id, value = word_values[j - 1]
            word = config.words[int(word_id)]
            msg = '{0:>15} {1:6} '
            w.write(msg.format(word, word_id))
        w.write('\n')
    w.flush()
    w.close()


def test(config, document_total, dictionary, corpus):
    lda_model = LdaModel.load('graph/LDA/mr.lda_model.topic_11.gensim')

    ldacm = CoherenceModel(model=lda_model, texts=document_total, corpus=corpus, dictionary=dictionary, coherence='c_v')
    print("模型Coherence：", ldacm.get_coherence())
    ldapl = lda_model.log_perplexity(corpus)
    print("模型perplexity：", str(ldapl))

    w = open('graph/LDA/mr.topic_documents.txt', 'w')
    for j in range(1, 501):
        for i in range(lda_model.num_topics):
            word_values = lda_model.show_topic(i, topn=j)
            word_id, value = word_values[j-1]
            word = config.words[int(word_id)]
            msg = '{0:>15} {1:6} '
            w.write(msg.format(word, word_id))
        w.write('\n')
    w.flush()
    w.close()


def construct_topic_graph(config):
    if not os.path.exists(config.topic_model_dir):
        print('训练LDA模型')
        train(config, random_state=None)

    lda_model = LdaModel.load(config.topic_model_dir)

    s_nodes, g_nodes = [], []
    for i in range(lda_model.num_topics):
        word_values = lda_model.show_topic(i, topn=config.word_tops)
        word_ids = []
        for j in range(len(word_values)):
            word_id_j, value_j = word_values[j]
            word_ids.append(int(word_id_j))
        for w in range(len(word_ids)):
            s_n = [word_ids[w] for x in range(len(word_ids))]
            s_nodes.extend(s_n)
            g_nodes.extend(word_ids)

    g = dgl.graph((s_nodes, g_nodes), num_nodes=len(config.words))
    g = dgl.add_self_loop(g)
    g = dgl.to_simple(g)

    dgl.save_graphs(config.word_word_Topic_dir, g)
    (g,), _ = dgl.load_graphs(config.word_word_Topic_dir)


def construct_topic_graph_two(config):
    lda_model = LdaModel.load(config.topic_model_dir)
    s_nodes, g_nodes = [], []
    for i in range(lda_model.num_topics):
        word_values = lda_model.show_topic(i, topn=config.word_tops)


# def build_graph_weight(config):
#     lda_model = LdaModel.load('graph/LDA/mr.lda_model.topic_11.gensim')
#     total_adjacent_matrix = []
#     for i in range(lda_model.num_topics):
#         word_values = lda_model.show_topic(i, topn=500)
#         topic_ids = []
#         adjacent_matrix = np.zeros((len(config.words), len(config.words)), dtype=np.float32)
#         for j in range(len(word_values)):
#             word_id, value = word_values[j]
#             topic_ids.append(word_id)


if __name__ == '__main__':
    from Dataset.ng20.config import Config
    config = Config()
    # train(config)
    construct_topic_graph(config)