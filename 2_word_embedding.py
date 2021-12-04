import os
import random
import time
import warnings

warnings.filterwarnings('ignore')
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import gensim
from gensim.models.callbacks import CallbackAny2Vec

np.random.seed(2021)
os.environ['PYTHONHASHSEED'] = '0'

# set the save_path for embedding
save_path_word2vec = 'embedding/embedding/word2vec'
save_path_glove = 'embedding/embedding/glove'
save_path_fasttext = 'embedding/embedding/fasttext'
for path in [save_path_word2vec, save_path_glove, save_path_fasttext]:
    if not os.path.exists(path):
        os.makedirs(path)

import logging

logging.basicConfig(filename='embedding/embedding/word2vec/train.log', format='%(asctime)s:%(message)s', level=logging.CRITICAL)

df = pd.read_pickle('data/processed_data/processed_data_numerical.pkl')


df = df.head(100)

# Word2Vec Training log
class EpochLogger(CallbackAny2Vec):
    def __init__(self, name, path):
        self.path = path
        self.epoch = 0
        self.best_loss = None
        self.name = name

    def on_epoch_end(self, model):
        cur_loss = float(model.get_latest_training_loss())
        #         if self.best_loss is None or cur_loss <= self.best_loss:
        #             self.best_loss = cur_loss
        #             model.wv.save_word2vec_format(self.path)
        message = "[{}] Epoch #{} {:.2f}".format(self.name, self.epoch, cur_loss)
        print(message)
        logging.critical(message)
        model.running_training_loss = 0.0  # word2vec默认是累计损失，会溢出
        self.epoch += 1


# construct the embedding matrix:
# output: embedding_matrix[id or category]=vector
def get_word_embedding(embed_path, vocab_size, glove=False):
    pre_embedding = {}
    # 用python的生成器读取大文件，并且选取在index中的读入，减少内存消耗
    with open(embed_path, encoding='utf8') as f:
        first_line = next(f)
        word_num, embed_size = int(first_line.split()[0]), int(first_line.split()[1])
        if glove:
            # glove 是context vector 和 bias vector 的concat
            word_num -= 1
            embed_size = 2 * embed_size
        embedding_matrix = np.zeros((vocab_size, embed_size))
        for line in tqdm(f, total=word_num):
            tmp = line.strip().split()
            if tmp[0] == '<unk>':
                continue
            embedding_matrix[int(tmp[0]), :] = np.array(tmp[1:embed_size + 1]).astype(np.float)
    return embedding_matrix

# train the model and get the word vectors
for name, epochs in zip(['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry'],
                        [50, 50, 20, 20, 20, 20]):
    path = os.path.join(save_path_word2vec, '{}_word2vec_sg1_hs0_win20_mc1_size300.txt'.format(name))
    input_docs = list(df[name].apply(lambda x: list(x.astype(str))))

    # w2v = gensim.models.Word2Vec(input_docs, size=300, sg=1, hs=0, alpha=0.025, min_alpha=0, window=20, seed=2020, workers=32, min_count=1, iter=epochs, compute_loss=True, callbacks=[EpochLogger(name, path)])
    w2v = gensim.models.Word2Vec(input_docs, vector_size=300, sg=1, hs=0, alpha=0.025, min_alpha=0, window=20,
                                 seed=2020, workers=32, min_count=1, epochs=epochs, compute_loss=True,
                                 callbacks=[EpochLogger(name, path)])
    w2v.wv.save_word2vec_format(path)
    del input_docs, w2v
    gc.collect()



# vocab_size: the largest number in this id/category list
embedding_path = './embedding/word2vec'
creative_w2v = get_word_embedding(embed_path=os.path.join(embedding_path, 'creative_id_word2vec_sg1_hs0_win20_mc1_size300.txt'), vocab_size=999977, glove=False)

ad_w2v = get_word_embedding(embed_path=os.path.join(embedding_path, 'ad_id_word2vec_sg1_hs0_win20_mc1_size300.txt'), vocab_size=879075, glove=False)

advertiser_w2v = get_word_embedding(embed_path=os.path.join(embedding_path, 'advertiser_id_word2vec_sg1_hs0_win20_mc1_size300.txt'), vocab_size=62959, glove=False)

product_w2v = get_word_embedding(embed_path=os.path.join(embedding_path, 'product_id_word2vec_sg1_hs0_win20_mc1_size300.txt'), vocab_size=17483, glove=False)

industry_w2v = get_word_embedding(embed_path=os.path.join(embedding_path, 'industry_word2vec_sg1_hs0_win20_mc1_size300.txt'), vocab_size=337, glove=False)

product_cate_w2v = get_word_embedding(embed_path=os.path.join(embedding_path, 'product_category_word2vec_sg1_hs0_win20_mc1_size300.txt'), vocab_size=16, glove=False)

print(creative_w2v.shape)
print(ad_w2v.shape)
print(advertiser_w2v.shape)
print(product_w2v.shape)
print(industry_w2v.shape)
print(product_cate_w2v.shape)

# 保存好embedding，便于下次直接读取
np.savez(os.path.join(embedding_path, 'embedding_w2v_sg1_hs0_win20_size300'),
         creative_w2v=creative_w2v.astype(np.float16),
         ad_w2v=ad_w2v.astype(np.float16),
         advertiser_w2v=advertiser_w2v.astype(np.float16),
         product_w2v=product_w2v.astype(np.float16),
         industry_w2v=industry_w2v.astype(np.float16),
         product_cate_w2v=product_cate_w2v.astype(np.float16))
