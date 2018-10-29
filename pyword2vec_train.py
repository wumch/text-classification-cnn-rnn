#!/opt/miniconda/bin/python

from typing import List
import os
import time
import re
import jieba
import word2vec


class Train:

    def __init__(self, extra_dict: str='../pubu/config/data/extra-dict.txt'):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.extra_dict = extra_dict
        if self.extra_dict and not os.path.isabs(self.extra_dict):
            self.extra_dict = os.path.join(self.root_dir, self.extra_dict)
        self.data_dir = os.path.join(self.root_dir, 'data', 'word_embedding')
        self.input_file = os.path.join(self.data_dir, 'all.txt')
        self.seged_file = os.path.join(self.data_dir, 'all.seged.txt')
        self.output_file = os.path.join(self.data_dir, 'embeddings.bin')
        self.vocab_all = os.path.join(self.data_dir, 'vocab_all.txt')
        if not os.path.exists(self.vocab_all):
            self.vocab_all = None
        self.save_vocab = os.path.join(self.data_dir, 'vocab.txt')
        self.embedding_dim = 128
        self.skip_window = 2
        self.min_tf = 3
        self.cbow = True
        self.respace = re.compile('\s+')
        jieba.load_userdict(self.extra_dict)

    def run(self):
        self._seg_file()
        self._train()

    def _train(self):
        begin = time.time()
        print(f'begin training at {begin}')
        word2vec.word2vec(
            self.seged_file, output=self.output_file, size=self.embedding_dim,
            window=self.skip_window, min_count=self.min_tf,
            cbow=1 if self.cbow else 0,
            read_vocab=self.vocab_all, save_vocab=self.save_vocab)
        end = time.time()
        print(f'finished training at {end}, elapse: {end-begin}s')

    def _seg_file(self):
        if os.path.exists(self.seged_file):
            print(f'seged file {self.seged_file} already exists, abort segment')
            return
        with open(self.input_file) as ifp, open(self.seged_file, 'w') as ofp:
            for line in ifp:
                ofp.write(' '.join(self.seg(line)) + os.linesep)
        print(f'seged file: {self.seged_file}')

    def seg(self, text: str) -> List[str]:
        text = text.replace('　', ' ')  # 空格 全角转半角
        text = self.respace.sub(' ', text).strip()
        return list(jieba.cut(text, cut_all=False)) if text else []


if __name__ == '__main__':
    _train = Train()
    _train.run()
