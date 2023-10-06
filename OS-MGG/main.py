import os
import dgl
import time
import torch
import random
import numpy as np

from model import Merge_Model

import torch.nn.functional as F
from dgl.nn.pytorch.factory import KNNGraph
from utils import process_dataset_sequence, split_train_val, split_train_val_own, get_time_dif

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def build_graphs_sequence(config):
#     kg1 = KNNGraph(config.word_tops)
#     g_word_word_cosine = kg1(config.embedding_pretrained, dist='cosine')
#     g_word_word_cosine = dgl.add_self_loop(g_word_word_cosine)
#     g_word_word_cosine.ndata['feat'] = config.embedding_pretrained
#
#     word_documents_adjmatrix_train, y_train, length_id_train = process_dataset_sequence(config, 'train')
#     train_indexs = np.zeros(y_train.shape[0])
#
#     word_documents_adjmatrix_test, y_test, length_id_test = process_dataset_sequence(config, 'test')
#     test_indexs = np.ones(y_test.shape[0])
#
#     labels_total = np.concatenate((y_train, y_test)).astype(np.int)
#     word_documents_adjmatrix_total = np.concatenate((word_documents_adjmatrix_train, word_documents_adjmatrix_test), axis=0)
#     length_id_total = np.concatenate((length_id_train, length_id_test)).astype(np.float)
#
#     test_mask = np.concatenate((train_indexs, test_indexs)).astype(np.bool)
#     train_mask = ~test_mask
#     val_mask, real_train_mask = split_train_val(train_mask, labels_total)
#
#     doc_sequence_embedding = F.embedding(torch.tensor(word_documents_adjmatrix_total, dtype=torch.int),
#                         torch.cat([config.embedding_pretrained, torch.zeros([1, config.embedding_pretrained.size()[-1]], dtype=torch.float32)], dim=0))
#     # doc_h = torch.mean(doc_sequence_embedding, dim=1, keepdim=False)
#     doc_h = torch.div(torch.sum(doc_sequence_embedding, dim=1), torch.tensor(length_id_total, dtype=torch.float32))
#
#     kg2 = KNNGraph(config.doc_tops)
#     g_doc_doc_cosine = kg2(doc_h, dist='cosine')
#     g_doc_doc_cosine = dgl.add_self_loop(g_doc_doc_cosine)
#
#     g_doc_doc_cosine.ndata['feat'] = doc_h
#     g_doc_doc_cosine.ndata['label'] = torch.tensor(labels_total, dtype=torch.int64)
#     g_doc_doc_cosine.ndata['length'] = torch.tensor(length_id_total, dtype=torch.int64)
#     g_doc_doc_cosine.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
#     g_doc_doc_cosine.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)
#     g_doc_doc_cosine.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
#     g_doc_doc_cosine.ndata['real_train_mask'] = torch.tensor(real_train_mask, dtype=torch.bool)
#
#     # dgl.save_graphs(config.word_to_word_grah_save_path, g_word_word_cosine)
#     # dgl.save_graphs(config.doc_to_doc_grah_save_path, g_doc_doc_cosine)
#
#     return g_word_word_cosine, g_doc_doc_cosine, word_documents_adjmatrix_total


def train(config, model):
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

    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    labels_total =torch.LongTensor(labels_total).to('cuda')
    word_documents_adjmatrix_total = torch.tensor(word_documents_adjmatrix_total, dtype=torch.int).to('cuda')
    length_id_total = torch.tensor(length_id_total, dtype=torch.float32).to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    best_val_acc = 0
    best_test_acc = 0
    for e in range(1000):
        logits = model(word_documents_adjmatrix_total, e)
        loss = F.cross_entropy(logits[real_train_mask], labels_total[real_train_mask])

        # Compute prediction
        pred = logits.argmax(1)

        if e % 1 == 0:
            train_acc = (pred[real_train_mask] == labels_total[real_train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels_total[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels_total[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                improve = "*"
            else:
                improve = ""
            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' \
                  '  Val Acc: {3:>6.2%}, Test Acc: {4:>6.2%}(Best Acc: {5:>6.2%}){6}'
            print(msg.format(e, loss.item(), train_acc, val_acc, test_acc, best_test_acc, improve))

            # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    try:
        from Dataset.mr.config import Config, init_seeds
        start_time = time.time()

        init_seeds()
        config = Config()

        model = Merge_Model(config).to('cuda')

        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        train(config, model)
    except KeyboardInterrupt as e:
        pass
