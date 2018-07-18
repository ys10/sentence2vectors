# coding=utf-8
import jieba
import re


class Sentences(object):
    def __init__(self, data_path, stop_words_file_path):
        self.stop_words_set = self.get_stop_words(stop_words_file_path)
        self.pattern = re.compile(u'<content>(.*?)</content>')
        with open(data_path, 'r', encoding='GBK', errors='ignore') as f:
            self.contents = [self.pattern.findall(line) for line in f if line.startswith('<content>')]

    def __iter__(self):
        for content in self.contents:
            if len(content) != 0:
                words = self.sentence2words(content[0].strip(), True, self.stop_words_set)
                yield words

    @staticmethod
    def get_stop_words(stop_words_file_path):
        with open(stop_words_file_path, 'rb') as f:
            stop_words_set = {line.strip().decode('utf-8') for line in f}
        return stop_words_set

    @staticmethod
    def sentence2words(sentence, stop_words=False, stop_words_set=None):
        """
        Split a sentence into words based on jieba.
        """
        # seg_words is a generator
        seg_words = jieba.cut(sentence)
        if stop_words:
            words = [word for word in seg_words if word not in stop_words_set and word != ' ']
        else:
            words = [word for word in seg_words]
        return words
