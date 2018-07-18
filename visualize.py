from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def visualize(model, output_path):
    meta_file_path = os.path.join(output_path, "w2v_metadata.tsv")
    placeholder = np.zeros((len(model.wv.index2word), 200))

    with open(meta_file_path, 'wb') as f:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensor-board")
                f.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                f.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    with tf.Session() as sess:
        embedding = tf.Variable(placeholder, trainable=False, name='w2v_metadata')
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(output_path, sess.graph)

        # adding into projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'w2v_metadata'
        embed.metadata_path = meta_file_path

        # Specify the width and height of a single thumbnail.
        projector.visualize_embeddings(writer, config)
        saver.save(sess, os.path.join(output_path, 'w2v_metadata.ckpt'))
        print('Run `tensorboard --logdir={0}` to run visualize result on tensor-board'.format(output_path))


def main():
    model = Word2Vec.load("w2v_model/w2v.model")
    visualize(model, "log/")


if __name__ == "__main__":
    main()
