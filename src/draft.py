import tensorflow as tf

xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
#
# embeddings
embs = tf.get_variable('emb', [5, 10])
rnn_inputs = tf.nn.embedding_lookup(embs, xs_)

print(tf.transpose(rnn_inputs, [1,0,2]).get_shape())