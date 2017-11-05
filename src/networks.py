import os
import numpy as np
import tensorflow as tf
import tflearn as tfl

class ReactivePolicy:

    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.g = tf.Graph()

        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')

            # Make policy
            self._policy()

            self.mse = tf.losses.mean_squared_error(self.act_ph, self.output)
            self.used_lr = 5e-4
            print("Calculated learning rate: {}".format(self.lr))
            print("Used learning rate: {}".format(self.used_lr))
            self.optim = tf.train.AdamOptimizer(self.used_lr).minimize(self.mse)

            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )

        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)


    def _policy(self):

        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * 10  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        self.output = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)),
                                     name="output")


    def train(self, observes, actions, batchsize=32):

        total_loss = 0
        ctr = 0

        for i in range(0, len(observes) - batchsize, batchsize):
            fd = {self.obs_ph : observes[i:i + batchsize],
                  self.act_ph : actions[i:i + batchsize]}
            loss, _ = self.sess.run([self.mse, self.optim], feed_dict=fd)
            total_loss += loss
            ctr += 1

        return total_loss/ctr


    def predict(self, obs):
        return self.sess.run(self.output, feed_dict={self.obs_ph : obs})


    def evaluate(self, observes, actions):
        return self.sess.run(self.mse, feed_dict={self.obs_ph: observes,
                                                   self.act_ph: actions})


    def save_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, "/home/shagas/Data/SW/RL_ALGOS/trpo/src/models/mimic_reactive/trained_model")


    def restore_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint("/home/shagas/Data/SW/RL_ALGOS/trpo/src/models/mimic_reactive/"))


    def visualize(self, env, n_episodes=3):

        for i in range(n_episodes):
            obs = env.reset()
            done = False

            while not done:
                env.render()
                action = self.predict(np.expand_dims(obs, axis=0))[0]
                obs, _, done, _ = env.step(action)


class RecurrentPolicy:
    def __init__(self, obs_dim, act_dim, n_units_list, seq_len, lr):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seq_len = int(seq_len)
        self.n_layers = len(n_units_list)
        self.n_units_list = n_units_list
        self.lr = lr

        self.g = tf.Graph()

        with self.g.as_default():
            with tf.name_scope("Placeholders"):
                with tf.name_scope("State-action-ph"):
                    self.obs_ph = tf.placeholder(tf.float32,
                                                 (None, self.seq_len, self.obs_dim),
                                                 'obs')
                    self.act_ph = tf.placeholder(tf.float32,
                                                 (None, self.seq_len, self.act_dim),
                                                 'act')

                    self.ss_obs_ph = tf.placeholder(tf.float32,
                                                 (None, 1, self.obs_dim),
                                                 'obs')

                with tf.name_scope("lstm-state-ph"):
                    self.lstm_state_ph_tuple_list = []

                    for i, n in enumerate(self.n_units_list):
                        lstm_state_ph_c = tf.placeholder(tf.float32,
                                                        (None, n),
                                                        'lstm-c-ph-l{}'.format(i))

                        lstm_state_ph_h = tf.placeholder(tf.float32,
                                                        (None, n),
                                                        'lstm-h-ph-l{}'.format(i))

                        self.lstm_state_ph_tuple_list.append(
                            tf.nn.rnn_cell.LSTMStateTuple(lstm_state_ph_c,
                                                          lstm_state_ph_h))

            # Make training lstm. Output from all layers through whole sequence
            with tf.name_scope("Training-lstm"):
                self.output_list, self.lstm_state_list = self._lstmnet(
                    self.obs_ph, reuse=False)
                self.output = self.output_list[-1]

                # self.trn_summary_list = []
                # for i, o in enumerate(self.output_list):
                #     power_o = tf.reduce_sum(tf.square(o), axis=2)
                #     self.trn_summary_list.append(tf.summary.histogram("tr-hist-outputs-l-{}".format(i), power_o))
                #     self.trn_summary_list.append(tf.summary.tensor_summary("tr-tensor-outputs-l-{}".format(i), power_o))


            # Make single step lstm. Output from all layers.
            with tf.name_scope("SS-lstm"):
                self.ss_output_list, self.ss_lstm_state_list = self._lstmnet(
                    self.ss_obs_ph, reuse=True)
                self.ss_output = self.ss_output_list[-1]

                joint_state_list = tf.stack(self.ss_lstm_state_list, axis=0)

                self.tst_summary_list = []
                for i, (o,s) in enumerate(zip(self.ss_output_list, self.ss_lstm_state_list)):
                    power_o = tf.reduce_sum(tf.square(o))
                    power_s = tf.reduce_sum(tf.square(s))
                    self.tst_summary_list.append(tf.summary.scalar("tst-scalar-outputs-l-{}".format(i), power_o))
                    #self.tst_summary_list.append(tf.summary.scalar("tst-scalar-states-l-{}".format(i), power_s))

            # # Make single step lstm. Output from all layers.
            # with tf.name_scope("SS-lstm"):
            #     self.ss_output_list, self.ss_lstm_state_list = \
            #         self._singlestepLSTM(self.ss_obs_ph,
            #                              self.lstm_state_ph_tuple_list,
            #                              reuse=False)
            #
            #     self.ss_output = self.ss_output_list[-1]
            #
            # # Make training lstm. Output from all layers through whole sequence
            # with tf.name_scope("Training-lstm"):
            #     self.output_list, self.lstm_state_list = self._trainingLSTM(
            #         self.obs_ph, self.lstm_state_ph_tuple_list, reuse=True)
            #     self.output = self.output_list[-1]
            #
            # # TODO: new states list will need choppin because states are now full

            # Training error for imitation
            with tf.name_scope("Training-error"):
                self.mse = tf.losses.mean_squared_error(self.act_ph, self.output)

            with tf.name_scope("Optimizer"):
                self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.mse)

            with tf.name_scope("Init-op"):
                self.init = tf.global_variables_initializer()

            #self.merged_trn = tf.summary.merge(self.trn_summary_list)
            self.merged_tst = tf.summary.merge(self.tst_summary_list)

        self.writer = tf.summary.FileWriter("/tmp/rnn_mimic/1", self.g)

        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)


    def _mlp(self, input, reuse):
        with tf.variable_scope("mlp", reuse=reuse):
            w_init = tfl.initializations.xavier(uniform=True)
            out = tfl.fully_connected(incoming=input,
                                      n_units=self.act_dim,
                                      weights_init=w_init)
        return out


    def _singlestepLSTM(self, obs, state_list, reuse=False):
        with tf.variable_scope("ss-lstm", reuse=reuse):
            w_init = tfl.initializations.xavier(uniform=True)

            # List holding final lstm states of all layers(list is n_layers long)
            lstm_output_states = []

            # List holding outputs of each layer (list is [n_layers + 1] long)
            outputs = []

            # Current layer input
            curr_output = obs

            name_ctr = 0
            for n, st in zip(self.n_units_list, state_list):
                curr_output, lstm_state = tfl.lstm(incoming=curr_output,
                                                    n_units=n,
                                                    activation='tanh',
                                                    dropout=(1, 1),
                                                    weights_init=w_init,
                                                    dynamic=False,
                                                    initial_state=st,
                                                    return_state=True,
                                                    name="layer_{}".format(
                                                        name_ctr))

                # Append state from layer to state list
                lstm_output_states.append(lstm_state)

                # Append all outputs from whole sequence from this layer
                outputs.append(curr_output)

                name_ctr += 1

            out = self._mlp(curr_output, reuse)
            outputs.append(out)

        return outputs, lstm_output_states


    def _trainingLSTM(self, obs, state_list, reuse=True):
        output_list = []
        lstm_state_list = []

        curr_lstm_states = state_list

        for i in range(self.seq_len):
            o = tf.slice(obs, [-1, i, -1], [-1, i + 1, -1])
            outputs, lstm_states = self._singlestepLSTM(o,
                                                        curr_lstm_states,
                                                        reuse=reuse)

            curr_lstm_states = lstm_states

            output_list.append(outputs)
            lstm_state_list.append(lstm_states)

        # Stack output and state lists
        output_tensor = tf.stack(output_list, axis=0)
        output_list = tf.unstack(output_tensor, axis=1)
        states_tensor = tf.stack(lstm_state_list, axis=0)
        states_list = tf.unstack(states_tensor, axis=1)

        return output_list, states_list


    def _lstmnet(self, obs_ph, reuse=False):
        with tf.variable_scope("lstm-layers", reuse=reuse):
            w_init = tfl.initializations.xavier(uniform=True)

            # List holding final lstm states of all layers(list is n_layers long)
            lstm_output_state_list = []

            # List holding outputs of each layer (list is [n_layers + 1] long)
            output_list = []

            # Current layer input
            curr_output = obs_ph
            curr_output_list = []
            name_ctr = 0
            for n, st in zip(self.n_units_list, self.lstm_state_ph_tuple_list):
                curr_output_list, lstm_state = tfl.lstm(incoming=curr_output,
                                                        n_units=n,
                                                        activation='tanh',
                                                        dropout=(1, 1),
                                                        weights_init=w_init,
                                                        dynamic=False,
                                                        initial_state=st,
                                                        return_seq=True,
                                                        return_state=True,
                                                        name="layer_{}".format(
                                                            name_ctr))

                # Stack list of tensors int single tensor
                curr_output = tf.stack(curr_output_list, axis=1)

                # Append state from layer to state list
                lstm_output_state_list.append(lstm_state)

                # Append all outputs from whole sequence from this layer
                output_list.append(curr_output)

                name_ctr += 1

            # Lstm output in action-space
            out_act_list = []
            reuse_mlp = False
            for t in curr_output_list:
                out_act_list.append(self._mlp(t, reuse_mlp or reuse))
                reuse_mlp = True

            out = tf.stack(out_act_list, axis=1)
            output_list.append(out)

        return output_list, lstm_output_state_list


    def train(self, observes, actions, state_list):

        observes_exp = np.expand_dims(observes, axis=0)
        actions_exp = np.expand_dims(actions, axis=0)

        fd = {self.obs_ph : observes_exp,
              self.act_ph : actions_exp}

        for (st_ten_c, st_ten_h), (st_num_c, st_num_h) in \
                zip(self.lstm_state_ph_tuple_list, state_list):
            fd[st_ten_c] = st_num_c
            fd[st_ten_h] = st_num_h

        runlist = [self.optim, self.mse]

        for s in self.lstm_state_list:
            runlist.append(s)

        ret = self.sess.run(runlist, feed_dict=fd)
        mse = ret[1]
        ret_states = ret[2:2+len(state_list)]

        return mse, ret_states


    def predict(self, obs, state_list):

        # Add sequence and batch dimension
        obs_exp = np.expand_dims(obs, axis=0)
        obs_exp = np.expand_dims(obs_exp, axis=0)
        fd={self.ss_obs_ph: obs_exp}

        for (st_ten_c, st_ten_h), (st_num_c, st_num_h) in \
                zip(self.lstm_state_ph_tuple_list, state_list):
            fd[st_ten_c] = st_num_c
            fd[st_ten_h] = st_num_h

        runlist = [self.ss_output, self.tst_summary_list]
        for s in self.ss_lstm_state_list:
            runlist.append(s)

        ret = self.sess.run(runlist, feed_dict=fd)
        action = ret[0]; summary = ret[1]; ret_states = ret[2:]

        self.writer.add_summary(summary)

        return action[0], ret_states


    def visualize(self, env, n_episodes=3):

        for i in range(n_episodes):
            obs = env.reset()
            done = False

            state_list = []
            for n in self.n_units_list:
                st_c = np.zeros((1, n))
                st_h = np.zeros((1, n))
                state_list.append((st_c, st_h))

            while not done:
                env.render()
                action, state_list = self.predict(obs, state_list)
                obs, _, done, _ = env.step(action[0])


    def save_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, "/home/shagas/Data/SW/RL_ALGOS/trpo/src/models/mimic_recurrent/trained_model")


    def restore_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint("/home/shagas/Data/SW/RL_ALGOS/trpo/src/models/mimic_recurrent/"))


class Hembedder:

    # TODO: add time penalty

    def __init__(self, obs_dim, z_dim, n_units_list, dropout_list, seq_len, lr):
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.seq_len = int(seq_len)
        self.n_rnn_layers = len(n_units_list)
        self.n_units_list = n_units_list
        self.dropout_list = dropout_list
        self.lr = lr

        self.g = tf.Graph()

        with self.g.as_default():
            with tf.name_scope("Placeholders"):
                with tf.name_scope("State-ph"):
                    self.obs_ph = tf.placeholder(tf.float32,
                                                 (None, self.seq_len, self.obs_dim),
                                                 'obs')

                with tf.name_scope("lstm-state-ph"):
                    self.lstm_state_ph_tuple_list = []

                    for i, n in enumerate(self.n_units_list):
                        lstm_state_ph_c = tf.placeholder(tf.float32,
                                                        (None, n),
                                                        'lstm-c-ph-l{}'.format(i))

                        lstm_state_ph_h = tf.placeholder(tf.float32,
                                                        (None, n),
                                                        'lstm-h-ph-l{}'.format(i))

                        self.lstm_state_ph_tuple_list.append(
                            tf.nn.rnn_cell.LSTMStateTuple(lstm_state_ph_c,
                                                          lstm_state_ph_h))

            # Make training lstm. Output from all layers through whole sequence
            self.output_list, \
            self.lstm_complete_c,\
            self.lstm_complete_h,\
            self.lstm_state_list = self._trainingLSTM_AE(self.obs_ph,
                                             self.lstm_state_ph_tuple_list)
            self.output = self.output_list[-1]

            self.FTT_pen_list = []

            with tf.name_scope("smooth_z"):
                # Smoothness penalties
                z_tensor = self.output_list[int(self.n_rnn_layers/2)]

                filter = np.ones((7, 1, 1), dtype=np.float32)
                smooth_z_list = []
                for i in range(self.z_dim):
                    # Smoothen the signal
                    smooth_z = tf.nn.conv1d(z_tensor[:,:,i:i+1],
                                           filter,
                                           stride=1,
                                           padding='SAME')
                    smooth_z_list.append(smooth_z)

                    # Add summary
                    tf.summary.tensor_summary('smooth_z_slice_{}'.format(i),
                                              tf.squeeze(smooth_z, 2))

                    # Make FFT on signal
                    z_fft = tf.real(tf.spectral.rfft(tf.squeeze(z_tensor[:, :, i:i + 1], 2)))
                    smooth_z_fft = tf.real(tf.spectral.rfft(tf.squeeze(smooth_z, 2)))

                    # Add FFT summaries
                    tf.summary.tensor_summary('fft_z_slice_{}'.format(i),
                                              z_fft)
                    tf.summary.tensor_summary('fft_smooth_z_slice_{}'.format(i),
                                              smooth_z_fft)

                    # FFT penalty
                    t = tf.range(0, z_fft.get_shape()[1], 1)
                    coeff_arr = tf.cast(tf.pow(t, 2), tf.float32)
                    pen_arr = tf.multiply(coeff_arr, z_fft)
                    FFT_pen = tf.reduce_sum(pen_arr)
                    self.FTT_pen_list.append(FFT_pen)

                    tf.summary.scalar('fft_pen_chan_{}'.format(i),FFT_pen)


                smooth_z_tensor = tf.squeeze(tf.stack(smooth_z_list, 2), 3)
                tf.summary.tensor_summary('smooth_z_tensor', smooth_z_tensor)


                self.smooth_mse = tfl.mean_square(z_tensor, smooth_z_tensor)
                tf.summary.scalar('smooth_mse', self.smooth_mse)


            # ===============================================

            # for i, o in enumerate(self.output_list):
            #     power_o = tf.reduce_sum(tf.square(o))
            #     tf.summary.scalar("outputs_l_{}".format(i), power_o)
            #
            # for i, s in enumerate(self.lstm_complete_c):
            #     power_s = tf.reduce_sum(tf.square(s))
            #     tf.summary.scalar("state_c_l_{}".format(i), power_s)
            #
            # for i, s in enumerate(self.lstm_complete_h):
            #     power_s = tf.reduce_sum(tf.square(s))
            #     tf.summary.scalar("state_h_l_{}".format(i), power_s)


            # Training error for imitation
            with tf.name_scope("Training-error"):
                self.mse = tf.losses.mean_squared_error(self.obs_ph, self.output)
                tf.summary.scalar("mse", self.mse)

            with tf.name_scope("Optimizer"):
                self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.mse +
                                                                      self.smooth_mse)

            with tf.name_scope("Init-op"):
                self.init = tf.global_variables_initializer()

            self.merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter("/tmp/Hembedder/1", self.g)

        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)


    def _singlestepLSTM_AE(self, obs, lstm_input_states, reuse=False):
        with tf.variable_scope("lstm_tower", reuse=reuse):
            w_init = tfl.initializations.xavier()

            # List holding final lstm states of all layers(list is n_layers long)
            lstm_output_states = []

            # List holding outputs of each layer
            outputs = []

            # Current layer input
            curr_t = tf.squeeze(obs, axis=1)


            # Encoding layers =================================================
            for i, (n, st, d) in enumerate(zip(self.n_units_list[:int(self.n_rnn_layers / 2)],
                                               lstm_input_states[:int(self.n_rnn_layers / 2)],
                                               self.dropout_list[:int(self.n_rnn_layers / 2)])):
                curr_t, lstm_state = tfl.lstm(incoming=tf.expand_dims(curr_t, axis=1),
                                              n_units=n,
                                              activation='tanh',
                                              dropout=d,
                                              weights_init=w_init,
                                              dynamic=False,
                                              initial_state=st,
                                              return_state=True,
                                              name="enc_layer_{}".format(i))

                # Append state from layer to state list
                lstm_output_states.append(lstm_state)

                # Append all outputs from whole sequence from this layer
                outputs.append(curr_t)


            crossover = tfl.fully_connected(curr_t,
                                            self.z_dim,
                                            activation='tanh',
                                            weights_init=w_init,
                                            name='crossover')

            outputs.append(crossover)

            # Current layer input
            curr_t = crossover

            # Decoding layers =================================================
            for i, (n, st, d) in enumerate(zip(self.n_units_list[int(self.n_rnn_layers / 2):],
                                               lstm_input_states[int(self.n_rnn_layers / 2):],
                                               self.dropout_list[int(self.n_rnn_layers / 2):])):
                curr_t, lstm_state = tfl.lstm(incoming=tf.expand_dims(curr_t, axis=1),
                                              n_units=n,
                                              activation='tanh',
                                              dropout=d,
                                              weights_init=w_init,
                                              dynamic=False,
                                              initial_state=st,
                                              return_state=True,
                                              name="dec_layer_{}".format(i))

                # Append state from layer to state list
                lstm_output_states.append(lstm_state)

                # Append all outputs from whole sequence from this layer
                outputs.append(curr_t)

            out_obs = tfl.fully_connected(curr_t,
                                        self.obs_dim,
                                        activation='linear',
                                        weights_init=w_init,
                                        name='out_obs')

            outputs.append(out_obs)


        return outputs, lstm_output_states


    def _trainingLSTM_AE(self, obs, state_list):
        output_list = []
        lstm_state_list_c = []
        lstm_state_list_h = []

        curr_lstm_states = state_list

        for i in range(self.seq_len):
            o = obs[:, i:i+1, :]
            outputs, lstm_states = self._singlestepLSTM_AE(o, curr_lstm_states,
                                                        reuse=(i > 0))

            # Assign current
            curr_lstm_states = lstm_states

            # Append outputs
            output_list.append(outputs)

            # Append lstm states
            c_list = []
            h_list = []
            for (c, h) in lstm_states:
                c_list.append(c)
                h_list.append(h)

            lstm_state_list_c.append(c_list)
            lstm_state_list_h.append(h_list)

        # Outputs
        output_tensor_list = []
        for tlist in zip(*output_list):
            stensor = tf.stack(tlist, axis=1)
            output_tensor_list.append(stensor)

        # Lstm states
        state_tensor_c_list = []
        for tlist in zip(*lstm_state_list_c):
            stensor = tf.stack(tlist, axis=1)
            state_tensor_c_list.append(stensor)

        state_tensor_h_list = []
        for tlist in zip(*lstm_state_list_h):
            stensor = tf.stack(tlist, axis=1)
            state_tensor_h_list.append(stensor)

        # Last state for passing
        lstm_passing_state = curr_lstm_states

        return output_tensor_list, state_tensor_c_list, \
               state_tensor_h_list, lstm_passing_state


    def train(self, observes, state_list):

        observes_exp = np.expand_dims(observes, axis=0)
        fd = {self.obs_ph : observes_exp}

        for (st_ten_c, st_ten_h), (st_num_c, st_num_h) in \
                zip(self.lstm_state_ph_tuple_list, state_list):
            fd[st_ten_c] = st_num_c
            fd[st_ten_h] = st_num_h

        runlist = [self.optim, self.mse, self.merged]

        for s in self.lstm_state_list:
            runlist.append(s)

        ret = self.sess.run(runlist, feed_dict=fd)
        mse = ret[1]; summary=ret[2]; ret_states = ret[3:]
        self.writer.add_summary(summary)

        return mse, ret_states


    def run(self, observes, state_list):

        observes_exp = np.expand_dims(observes, axis=0)
        fd = {self.obs_ph : observes_exp}

        for (st_ten_c, st_ten_h), (st_num_c, st_num_h) in \
                zip(self.lstm_state_ph_tuple_list, state_list):
            fd[st_ten_c] = st_num_c
            fd[st_ten_h] = st_num_h

        return self.sess.run([s for s in self.lstm_state_list], feed_dict=fd)


    def save_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, "/home/shagas/Data/SW/RL_ALGOS/trpo/src/models/h_embedder/trained_model")


    def restore_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint("/home/shagas/Data/SW/RL_ALGOS/trpo/src/models/h_embedder/"))


class SimpleQNet:
    def __init__(self, obs_dim, act_dim, gamma):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.Q_max = 0

        self.weights_path = 'models/qnet'
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)

        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
            self.y_ph = tf.placeholder(tf.float32, (None, 1), 'y')

            # Make policy
            self.Q = self._qnet(self.obs_ph, self.act_ph, reuse=False)

            self.lr = 1e-3
            self.tderr = tf.losses.mean_squared_error(self.Q, self.y_ph)
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.tderr)

            self.opt_act = tfl.variable("opt_act", shape=(act_dim))
            self.Q_opt = self._qnet(self.obs_ph, tf.expand_dims(self.opt_act, 0), reuse=True)
            self.init_act_rnd = tf.assign(self.opt_act, tf.random_normal(tf.shape(self.opt_act)))
            self.init_act_zero = tf.assign(self.opt_act, tf.zeros(tf.shape(self.opt_act)))
            self.optimize_action = tf.train.AdamOptimizer(1e-3).minimize(-self.Q_opt, var_list=self.opt_act)

            self.rnd_acts = tf.random_uniform((1000, self.act_dim), minval=-2, maxval= 2)
            self.rnd_acts = tf.random_normal((1000, self.act_dim))
            self.Q_max = self._qnet(self.obs_ph, self.rnd_acts, reuse=True)
            self.qmax = tf.reduce_max(self.Q_max)
            self.actmax = self.rnd_acts[tf.arg_max(self.Q_max, 0)]

            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)


    def _qnet(self, obs_ph, act_ph, reuse=False):
        with tf.variable_scope("qnet", reuse=reuse):
            input = tf.concat([obs_ph, act_ph], 1)
            w_init = tfl.initializations.xavier()
            fc1 = tfl.fully_connected(input, 64, 'tanh', weights_init=w_init)
            fc2 = tfl.fully_connected(fc1, 64, 'tanh', weights_init=w_init)
            fc3 = tfl.fully_connected(fc2, 1, 'linear', weights_init=w_init)
        return fc3


    def train(self, observes, actions, rewards, batchsize=32):
        N = len(observes)
        assert N > batchsize

        total_loss = 0.
        for i in range(int(N/batchsize)):

            # Sample batch of s, a, r, _s
            rndvec = np.random.choice(N - 2, batchsize, replace=False)

            observes_batch = observes[rndvec]
            actions_batch = actions[rndvec]
            rewards_batch = rewards[rndvec]
            observes_new_batch = observes[rndvec + 1]
            actions_new_batch = actions[rndvec + 1]

            q_target = self.predict_Q(observes_new_batch, actions_new_batch)
            y_hat = np.expand_dims(rewards_batch, 1) + self.gamma * q_target

            fd = {self.obs_ph : observes_batch,
                  self.act_ph : actions_batch,
                  self.y_ph : y_hat}

            Q, loss, _ = self.sess.run([self.Q, self.tderr, self.optim], feed_dict=fd)
            total_loss += loss

            if np.max(Q) > self.Q_max + 3:
                self.Q_max = np.max(Q)
                print("Max Q: {}".format(self.Q_max))

        return total_loss/(int(N/batchsize))


    def predict_Q(self, obs, act):
        return self.sess.run(self.Q, feed_dict={self.obs_ph : obs,
                                                self.act_ph : act})


    def save_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, self.weights_path + "/trained_model")


    def restore_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(self.weights_path))


    def get_act(self, obs):
        # action = np.random.rand(self.act_dim)
        # for i in range(300):
        #     fd = {self.obs_ph: np.expand_dims(obs, 0),
        #           self.act_ph: np.expand_dims(action, 0)}
        #     grad = self.sess.run(self.act_q_grad, feed_dict=fd)[0][0]
        #     action += 0.1 * grad

        # Initialize action
        self.sess.run([self.init_act_rnd])

        # Optimize
        for i in range(200):
            self.sess.run(self.optimize_action, {self.obs_ph : np.expand_dims(obs,0)})

        # Get variable value
        action = self.sess.run(self.opt_act)
        return action


    def get_act_sampling(self, obs):
        return  self.sess.run(self.actmax,
                               {self.obs_ph : np.expand_dims(obs, 0)})


    def visualize(self, env, n_episodes=5):
        for i in range(n_episodes):
            obs = env.reset()
            done = False

            while not done:
                env.render()
                obs, _, done, _ = env.step(self.get_act_sampling(obs))

