#!/opt/miniconda/bin/python
# coding: utf-8

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

base_dir = os.path.join(os.path.dirname(__file__), 'data/comments')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:

    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        content = message
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


def test(samples_file: str = os.path.join(base_dir, 'test.txt')):
    model = CnnModel()
    total = cor = err = 0
    for sample in open(samples_file):
        label, content = sample.rstrip().split(',', 1)
        pred_label = model.predict(content)
        total += 1
        if label != pred_label:
            err += 1
        else:
            cor += 1
    print('total: {}, correct: {} as {:>.2f}%, error: {} as {:>.2f}%'.format(total, cor, (cor * 100) / total, err, (err * 100) / total))


if __name__ == '__main__':
    samples_file = os.path.join(base_dir, 'test.txt')
    print('samples file: {}'.format(samples_file))
    test(samples_file)
