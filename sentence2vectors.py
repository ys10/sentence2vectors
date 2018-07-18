# coding=utf-8
from sentences import Sentences
from gensim.models import Word2Vec


def train(data_path, stop_words_file_path):
    sentences = Sentences(data_path, stop_words_file_path)
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
    return model


def save(model, model_path):
    model.save(model_path)


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
    model = train(data_path, stop_words_file_path)
    # save model
    save(model, model_path)
    # check model by some simple tests.
    test(model)


if __name__ == '__main__':
    main()
