# coding=utf-8
import jieba
import re
from gensim.models import Word2Vec


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


def train_save(data_path, stop_words_file_path, model_path):
    sentences = Sentences(data_path, stop_words_file_path)
    print(sentences)
    num_features = 256
    min_word_count = 10
    num_workers = 48
    context = 20
    epoch = 20
    sample = 1e-5
    model = Word2Vec(
        sentences,
        size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        sample=sample,
        window=context,
        iter=epoch,
    )
    model.save(model_path)
    return model


def test(model):
    # get the word vector
    for w in model.most_similar(u'记者'):
        print(w[0], w[1])
    # calculate similarity
    print(model.similarity(u'中国', u'北京'))
    # show vector
    country_vec = model[u"国家"]
    print(country_vec)


def main():
    data_path = 'data/news_tensite_xml.smarty.dat'
    stop_words_file_path = 'data/all_stop_words.txt'
    model_path = 'w2v_model/w2v.model'
    # train model
    model = train_save(data_path, stop_words_file_path, model_path)
    # check model by some simple tests.
    test(model)


if __name__ == '__main__':
    main()
