import dgl
from dgl.nn.pytorch.factory import KNNGraph


def construct_dis_graph(config):
    kg_words = KNNGraph(config.word_tops)
    g = kg_words(config.embedding_pretrained, dist='euclidean')
    g = dgl.add_self_loop(g)
    g = dgl.to_simple(g)
    dgl.save_graphs(config.word_word_Dis_dir, g)
