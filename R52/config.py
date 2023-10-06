import os
import dgl
import torch
import random
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self):
        self.dataset = 'R52'

        root_dir = ''

        self.dataset_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '.txt'
        self.dataset_text = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '.raw.txt'
        self.dataset_clean = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '.clean.txt'

        self.vocab_path = root_dir + 'Dataset/' + self.dataset + '/' + 'vocab_dict.txt'
        self.vocab_embedding = root_dir + 'Dataset/' + self.dataset + '/' + 'vocab_dict_embeddings.txt'
        self.vocab_embedding_glove = root_dir + 'Dataset/' + self.dataset + '/' + 'vocab_dict_glove.txt'

        self.is_raw_text = False

        self.device = 'cuda'

        self.batch_size = 128

        # self.words = [x.strip() for x in open(self.vocab_path, encoding='utf-8').readlines()]
        # self.embedding_pretrained = torch.tensor(np.loadtxt(self.vocab_embedding, dtype=np.float), dtype=torch.float32)  # 预训练词向量
        self.words, self.embedding_pretrained = read_glove_embedding(self.vocab_embedding_glove)

        # self.pading_tesor = torch.zeros([1, self.embedding_pretrained.size()[-1]], dtype=torch.float32)

        self.word_to_id = dict(zip(self.words, range(len(self.words))))
        self.n_vocab = len(self.words)
        print(f"Vocab size: {len(self.words)}")

        self.categories = sorted(list(set([x.strip().split('\t')[2] for x in open(self.dataset_dir, encoding='utf-8').readlines()])))
        self.num_classes = len(self.categories)  # 类别数
        self.cat_to_id = dict(zip(self.categories, range(len(self.categories))))
        print(self.categories)
        print("类别数量：", len(self.categories))

        self.max_length = 60

        self.each_word_out_degree_dis = 1000
        self.each_word_out_degree_pmi = 1000
        self.each_word_out_degree_top = 1000

        self.PMI_windows_size = 60

        self.num_topics = 50
        self.lda_epoch = 500
        self.topic_model_random_state = 0
        self.topic_model_dir = root_dir + 'Dataset/' + self.dataset + '/LDA/' + self.dataset + '.lda_model.topic_' + str(self.num_topics) + '.gensim'

        self.word_word_Dis_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_word_to_word_Dis_Graph_' + str(self.each_word_out_degree_dis) + '.dgl'
        self.word_word_Pmi_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_word_to_word_PMI_Graph_' + str(self.each_word_out_degree_pmi) + '.dgl'
        self.word_word_Topic_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_word_to_word_LDA_Graph_' + str(self.each_word_out_degree_top) + '.dgl'

        if os.path.exists(self.word_word_Dis_dir):
            (self.g_word_word_Dis,), _ = dgl.load_graphs(self.word_word_Dis_dir)
            # self.g_word_word_Dis = self.g_word_word_Dis.to('cuda')
        if os.path.exists(self.word_word_Pmi_dir):
            (self.g_word_word_Pmi,), _ = dgl.load_graphs(self.word_word_Pmi_dir)
            # self.g_word_word_PMI = self.g_word_word_PMI.to('cuda')
        if os.path.exists(self.word_word_Topic_dir):
            (self.g_word_word_Top,), _ = dgl.load_graphs(self.word_word_Topic_dir)
            # self.g_word_word_Top = self.g_word_word_Top.to('cuda')

        self.GCN_hidden_in_feat_dim = 300
        self.GCN_hidden_out_feat_dim = 300

        self.LSTM_hidden_in_feat_dim = 300
        self.LSTM_hidden_out_feat_dim = 300
        self.layer_num = 2
        self.droup_out_lstm = 0.5

        self.classifer_in_feat_dim = 300


def read_glove_embedding(read_dir):
    data = [x.strip().split(' ') for x in open(read_dir, encoding='utf-8').readlines()]
    words = [x[0] for x in data]
    embedding = []
    for i in range(len(data)):
        e = [float(x) for x in data[i][1:]]
        embedding.append(e)
    embedding_pretrained = torch.tensor(embedding, dtype=torch.float32)  # 预训练词向量
    return words, embedding_pretrained


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call
    # this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call
    # this function if CUDA is not available; in that case, it is silently ignored.
    dgl.seed(seed)

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 对模型里的卷积层进行预先的优化，也就是在每一个卷积层中测试 cuDNN 提供的所有卷积实现算法，然后选择最快的那个
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True