import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils.batchloader import BatchLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='word2vec visualization')
    parser.add_argument('--lang', type=str, default='ru', metavar='L',
                        help='lang (default: ru)')
    args = parser.parse_args()

    if not os.path.exists('./data/preprocessed_data/word_embeddings_{}.npy'.format(args.lang)):
        raise FileNotFoundError("word embeddings file was't found")

    tsne = TSNE(n_components=2)
    word_embeddings = np.load('./data/preprocessed_data/word_embeddings_{}.npy'.format(args.lang))
    word_embeddings_pca = tsne.fit_transform(word_embeddings)

    batch_loader = BatchLoader()
    words = batch_loader.idx_to_word

    fig, ax = plt.subplots()
    fig.set_size_inches(150, 150)
    x = word_embeddings_pca[:, 0]
    y = word_embeddings_pca[:, 1]
    ax.scatter(x, y)

    for i, word in enumerate(words):
        ax.annotate(word, (x[i], y[i]))

    fig.savefig('word_embedding.png', dpi=130)
