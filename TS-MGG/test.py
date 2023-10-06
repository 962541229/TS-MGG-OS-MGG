import torch
import random
import torchvision.models as models


if __name__ == '__main__':
    from Dataset.ng20.config import Config

    config = Config()

    positive_path = '../..opinion-lexicon-English/positive-words.txt'
    negative_path = '../..opinion-lexicon-English/negative-words.txt'

    p_words = []
    n_words = []

    with open(positive_path, 'r') as f:
        for line in f:
            p_words.append(line.strip('\n'))

    with open(negative_path, 'r', encoding='gbk') as f:
        for line in f:
            n_words.append(line.strip('\n'))

    ours_p_words = []
    ours_n_words = []

    for word in config.words:
        if word in p_words:
            ours_p_words.append(word)
        if word in n_words:
            ours_n_words.append(word)

    random_integers = [random.randint(0, len(ours_n_words) - 1) for _ in range(50)]

    random_words = [ours_n_words[i] for i in random_integers]

    print(random_words)