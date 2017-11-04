import tensorflow as tf
import tflearn as tfl
import numpy as np
import os

class RNNSim:
    def __init__(self, config, obs_dim, act_dim):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_units = self.config['n_state'] # Lstm state units
        self.env_init_state = np.expand_dims(config["env"].reset(), 0).astype(np.float32)

        self.weights_path = 'models/fullsys_rnn'
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)

        self.g = tf.Graph()
        with self.g.as_default():
            with tf.name_scope("Placeholders"):
                self.obs_ph = tf.placeholder(dtype=tf.float32,
                                             shape=(None, self.obs_dim),
                                             name='obs_ph')

                self.cost_ph = tf.placeholder(dtype=tf.float32,
                                             shape=(None, 1),
                                             name='cost_ph')

                self.act_ph = tf.placeholder(dtype=tf.float32,
                                             shape=(None, self.act_dim),
                                             name='act_ph')

                self.seq_len = tf.placeholder(dtype=tf.int32,
                                              name='ptrain_ph')

                lstm_state_ph_c = tf.placeholder(tf.float32,
                                                 (None, self.n_units),
                                                 'lstm_c_ph')

                lstm_state_ph_h = tf.placeholder(tf.float32,
                                                 (None, self.n_units),
                                                 'lstm_h_ph')

                self.init_state = tf.nn.rnn_cell.LSTMStateTuple(
                    lstm_state_ph_c, lstm_state_ph_h)

                self.zero_init_state = tf.nn.rnn_cell.LSTMStateTuple(
                tf.zeros((1, self.n_units)), tf.zeros((1, self.n_units)))

            # Single state (and weights initializations)
            self.ss_policy = self._policy(self.obs_ph, reuse=False)
            self.ss_lstm_state, self.ss_cost_pred, self.ss_state_pred = \
                self.ss_fullstep(self.act_ph, self.init_state, reuse=False)

            # Outputs for model and cost optimization
            self.m_lstm_states, self.m_cost_pred, self.m_state_pred = \
                self._modelpred(self.act_ph)

            # Outputs for policy optimization
            self.a_lstm_states, self.a_cost_pred, self.a_state_pred = \
                self._actpred()

            self.mean_cost_pred = tf.reduce_mean(self.a_cost_pred)

            # Variables
            self.policyvars = tf.get_collection(
                key=tf.GraphKeys.GLOBAL_VARIABLES,
                scope='policy')
            self.costvars = tf.get_collection(
                key=tf.GraphKeys.GLOBAL_VARIABLES,
                scope='c_pred')
            self.modelvars = tf.get_collection(
                key=tf.GraphKeys.GLOBAL_VARIABLES,
                scope='s_pred')
            self.lstmvars = tf.get_collection(
                key=tf.GraphKeys.GLOBAL_VARIABLES,
                scope='lstm')

            # Loss functions
            self.cost_loss = tfl.mean_square(self.m_cost_pred[1:],
                                             self.cost_ph[1:])
            self.model_loss = tfl.mean_square(self.m_state_pred[1:],
                                              self.obs_ph[1:])

            self.env_optim = tf.train.AdamOptimizer(self.config["env_lr"]).minimize(
                self.cost_loss + self.model_loss,
                var_list=[self.costvars, self.modelvars, self.lstmvars]
            )

            self.policy_optim = tf.train.AdamOptimizer(
                self.config["policy_lr"]).minimize(
                -self.a_cost_pred,
                var_list=[self.policyvars, self.lstmvars]
            )

            self.init = tf.global_variables_initializer()

        tfconfig = tf.ConfigProto(
            device_count={'GPU': self.config['gpu']}
        )

        self.sess = tf.Session(graph=self.g, config=tfconfig)
        self.sess.run(self.init)


    def _policy(self, obs, reuse):
        with tf.variable_scope('policy', reuse=reuse):
            w_init = tfl.initializations.xavier(uniform=True)
            fc1 = tfl.fully_connected(incoming=obs,
                                      n_units=32,
                                      activation='tanh',
                                      weights_init=w_init,
                                      name='fc1')
            fc1 = tfl.dropout(fc1, keep_prob=0.7)

            fc2 = tfl.fully_connected(incoming=fc1,
                                      n_units=32,
                                      activation='tanh',
                                      weights_init=w_init,
                                      name='fc2')
            fc2 = tfl.dropout(fc2, keep_prob=0.7)

            fc3 = tfl.fully_connected(incoming=fc2,
                                      n_units=self.act_dim,
                                      activation='linear',
                                      weights_init=w_init,
                                      name='fc3')

        return fc3


    def _c_pred(self, obs, reuse):
        with tf.variable_scope('c_pred', reuse=reuse):
            w_init = tfl.initializations.xavier(uniform=True)
            fc1 = tfl.fully_connected(incoming=obs,
                                      n_units=16,
                                      activation='tanh',
                                      weights_init=w_init,
                                      name='fc1')
            fc1 = tfl.dropout(fc1, keep_prob=0.7)

            fc2 = tfl.fully_connected(incoming=fc1,
                                      n_units=16,
                                      activation='tanh',
                                      weights_init=w_init,
                                      name='fc2')
            fc2 = tfl.dropout(fc2, keep_prob=0.7)

            fc3 = tfl.fully_connected(incoming=fc2,
                                      n_units=1,
                                      activation='linear',
                                      weights_init=w_init,
                                      name='fc3')

        return fc3


    def _s_pred(self, obs, reuse):
        with tf.variable_scope('s_pred', reuse=reuse):
            w_init = tfl.initializations.xavier(uniform=True)
            fc1 = tfl.fully_connected(incoming=obs,
                                      n_units=16,
                                      activation='tanh',
                                      weights_init=w_init,
                                      name='fc1')
            fc1 = tfl.dropout(fc1, keep_prob=0.7)

            fc2 = tfl.fully_connected(incoming=fc1,
                                      n_units=16,
                                      activation='tanh',
                                      weights_init=w_init,
                                      name='fc2')
            fc2 = tfl.dropout(fc2, keep_prob=0.7)

            fc3 = tfl.fully_connected(incoming=fc2,
                                      n_units=self.obs_dim,
                                      activation='linear',
                                      weights_init=w_init,
                                      name='fc3')

        return fc3


    def ss_fullstep(self, act, init_state, reuse):
        with tf.variable_scope("lstm", reuse=reuse):
            w_init = tfl.initializations.xavier(uniform=True)
            lstm_output, state = tfl.lstm(incoming=tf.expand_dims(act, 1),
                                    n_units=self.n_units,
                                    dropout=(1, 0.7),
                                    weights_init=w_init,
                                    initial_state=init_state,
                                    return_state=True,
                                    name='lstm_layer')

        c_pred = self._c_pred(lstm_output, reuse=reuse)
        s_pred = self._s_pred(lstm_output, reuse=reuse)

        return (state, c_pred, s_pred)


    def _modelpred(self, actions):
        def step(prev, X):
            lstm_init_s, _, _ = prev
            return self.ss_fullstep(tf.expand_dims(X, 0),
                                    lstm_init_s, reuse=True)

        initializer = (self.zero_init_state, tf.zeros((1,1)), self.env_init_state)
        lstm_outputs, c_predictions, s_predictions = tf.scan(step, actions,
                                                         initializer=initializer)

        return lstm_outputs, c_predictions, s_predictions


    def _actpred(self):
        def step(prev, _):
            # Unpack previous state
            lstm_init_s, _, s_pred = prev
            # Get action from policy
            act = self._policy(s_pred, reuse=True)
            return self.ss_fullstep(act, lstm_init_s, reuse=True)

        initializer = (self.zero_init_state, tf.zeros((1,1)), self.env_init_state)
        lstm_outputs, c_predictions, s_predictions = tf.scan(step,
                                     tf.zeros((self.seq_len, 1)), initializer=initializer)

        return lstm_outputs, c_predictions, s_predictions


    def predict(self, obs):
        fd = {self.obs_ph : np.expand_dims(obs, 0)}
        return self.sess.run(self.ss_policy, feed_dict=fd)[0]


    def trainmodel(self, observs, acts, rews):
        fd = {self.obs_ph : observs,
              self.act_ph : acts,
              self.cost_ph : np.reshape(rews, (-1, 1)) }
        fetches = [self.env_optim, self.cost_loss, self.model_loss]
        _, cost_loss, model_loss = self.sess.run(fetches, fd)
        return cost_loss, model_loss


    def trainpolicy(self, seq_len):
        fd = {self.seq_len : seq_len}
        _, cost_pred = self.sess.run([self.policy_optim, self.mean_cost_pred],
                                     feed_dict=fd)
        return cost_pred


    def save_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, self.weights_path + "/trained_model")


    def restore_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(self.weights_path))



