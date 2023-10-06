import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.factory import KNNGraph
from multi_head_attention import Multi_Head_Attention

from wzj_GraphConv import GraphConv
from wzj_SAGEConv import SAGEConv
# from dgl.nn.pytorch.conv import GATConv
# from dgl.nn.pytorch.conv import SAGEConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        # print(h.shape)
        # h = F.relu(h)
        return h


# class SAGEConvModel(nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(SAGEConvModel, self).__init__()
#         self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
#
#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         return h


# class GATModel(nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(GATModel, self).__init__()
#         self.gat1 = GATConv(in_feats, h_feats, num_heads=1)
#
#     def forward(self, g, in_feat):
#         h = self.gat1(g, in_feat)
#         return h


class Merge_Model(nn.Module):
    def __init__(self, config):
        super(Merge_Model, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)

        # self.embedding = nn.Embedding.from_pretrained(
        #     torch.cat([config.embedding_pretrained, config.pading_tesor], dim=0), freeze=False)

        # word_index_tensor = torch.arange(start=0, end=config.n_vocab, step=1, dtype=torch.int64)
        # one_hot_embedding = F.one_hot(word_index_tensor, num_classes=-1)
        # self.embedding = nn.Embedding.from_pretrained(one_hot_embedding, freeze=False)

        (self.g_word_word_Dis,), _ = dgl.load_graphs(config.word_word_Dis_dir)
        self.g_word_word_Dis = self.g_word_word_Dis.to('cuda')

        (self.g_word_word_PMI,), _ = dgl.load_graphs(config.word_word_PMI_dir)
        self.g_word_word_PMI = self.g_word_word_PMI.to('cuda')

        (self.g_word_word_Top,), _ = dgl.load_graphs(config.word_word_Topic_dir)
        self.g_word_word_Top = self.g_word_word_Top.to('cuda')

        self.word_level_Dis_GCN1 = GCN(300, 300)
        self.word_level_Dis_GCN2 = GCN(300, 300)

        self.word_level_PMI_GCN1 = GCN(300, 300)
        self.word_level_PMI_GCN2 = GCN(300, 300)

        self.word_level_Top_GCN1 = GCN(300, 300)
        self.word_level_Top_GCN2 = GCN(300, 300)

        self.attention = Multi_Head_Attention(300, 30)

        self.dense_layer = nn.Linear(300, 300)
        self.fc = nn.Linear(300, config.num_classes)

        self.doc_level_GCN1 = GCN(300, 150)
        self.doc_level_GCN2 = GCN(150, config.num_classes)

    def forward(self, word_documents_adjmatrix, iter):
        word_h = self.embedding.weight

        word_Dis_h = self.word_level_Dis_GCN1(self.g_word_word_Dis, word_h)
        # word_Dis_h = F.relu(word_Dis_h)
        word_Dis_h = self.word_level_Dis_GCN2(self.g_word_word_Dis, word_Dis_h)
        word_Dis_h = torch.add(word_h, word_Dis_h)  # 残差连接

        word_PMI_h = self.word_level_PMI_GCN1(self.g_word_word_PMI, word_h)
        # word_PMI_h = F.relu(word_PMI_h)
        word_PMI_h = self.word_level_PMI_GCN2(self.g_word_word_PMI, word_PMI_h)
        word_PMI_h = torch.add(word_h, word_PMI_h)  # 残差连接

        word_Top_h = self.word_level_Top_GCN1(self.g_word_word_Top, word_h)
        # word_Top_h = F.relu(word_Top_h)
        word_Top_h = self.word_level_Top_GCN2(self.g_word_word_Top, word_Top_h)
        word_Top_h = torch.add(word_h, word_Top_h)  # 残差连接

        cat_tensor = torch.cat((word_Dis_h, word_PMI_h, word_Top_h), dim=1).reshape(word_h.size()[0], 3, 300)
        doc_embedding = self.attention(cat_tensor)
        # 词-文档转换
        doc_embedding = torch.cat([torch.squeeze(doc_embedding), word_h.new_zeros([1, word_h.size()[-1]])], dim=0)
        doc_h = F.embedding(word_documents_adjmatrix, doc_embedding)

        # doc_h = torch.mean(doc_h, dim=1, keepdim=False)
        doc_h = self.dense_layer(doc_h)
        doc_h = torch.mean(doc_h, dim=1, keepdim=False)
        gcn_re = self.fc(doc_h)
        # doc_h = torch.div(torch.sum(doc_h, dim=1), length_id_total)

        # print("计算g_doc_doc_cosine")
        # kg = KNNGraph(300)
        # g_doc_doc_cosine = kg(doc_h, dist='cosine')
        # dgl.add_self_loop(g_doc_doc_cosine).to('cuda')
        # g_doc_doc_cosine.ndata['feat'] = doc_h
        #
        # doc_h = self.doc_level_GCN1(g_doc_doc_cosine, doc_h)
        # doc_h = F.relu(doc_h)
        # gcn_re = self.doc_level_GCN2(g_doc_doc_cosine, doc_h)

        # total = torch.div(torch.add(dense_re, gcn_re), 2)

        return gcn_re
