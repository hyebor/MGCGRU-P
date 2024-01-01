import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model.utils_tf import DCGRU, GRU, training

class PRNNModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        nParticle = int(model_kwargs.get('nParticle', 1))
        nParticle_test = int(model_kwargs.get('nParticle_test', 2))

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        horizon = int(model_kwargs.get('horizon', 1))
        num_nodes = int(model_kwargs.get('num_nodes', 545))
        rnn_units = int(model_kwargs.get('rnn_units', 64))
        seq_len = int(model_kwargs.get('seq_len', 12))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))
        embed_dim = int(model_kwargs.get('embed_dim', 10))
        rho = float(model_kwargs.get('rho', 1.0))
        rnn_type = model_kwargs.get('rnn_type', 'dcgru')  # dcgru/gru

        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        if rnn_type == 'dcgru':
            rnn_0 = DCGRU(adj_mx=adj_mx, input_dim=input_dim, num_units=rnn_units,
                          max_diffusion_step=max_diffusion_step, scope="layer0")
            rnn_1 = DCGRU(adj_mx=adj_mx, input_dim=rnn_units, num_units=rnn_units,
                          max_diffusion_step=max_diffusion_step, scope="layer1")
        elif rnn_type == 'gru':
            rnn_0 = GRU(adj_mx=adj_mx, input_dim=input_dim, num_units=rnn_units, scope="layer0")
            rnn_1 = GRU(adj_mx=adj_mx, input_dim=rnn_units, num_units=rnn_units, scope="layer1")
        else:
            print('ERROR! ERROR! ERROR!')
            exit(0)

        global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('DCRNN_SEQ', reuse=tf.AUTO_REUSE):
            weight = tf.get_variable("weight", [rnn_units, output_dim], dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer())
            weight_delta = tf.get_variable("weight_delta", [rnn_units, output_dim], dtype=tf.float32,
                                           initializer=tf.glorot_uniform_initializer())
            node_embedding = tf.get_variable("node_embedding", [num_nodes, embed_dim], dtype=tf.float32,
                                             initializer=tf.glorot_uniform_initializer())
            curriculam_prob = self._compute_sampling_threshold(global_step, cl_decay_steps)

            if is_training:
                output_samples, output_mu, output_sigma = training(self._inputs, self._labels, rnn_0, rnn_1, weight,
                                                                   weight_delta,
                                                                   rho, node_embedding, rnn_units, nParticle,
                                                                   is_training, curriculam_prob, rnn_type)
            else:
                output_samples, output_mu, output_sigma = training(self._inputs, self._labels, rnn_0, rnn_1, weight,
                                                                   weight_delta,
                                                                   rho, node_embedding, rnn_units, nParticle_test,
                                                                   is_training, curriculam_prob, rnn_type)

        # Project the output to output_dim.
        outputs = tf.reduce_mean(output_samples, axis=3)
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._outputs_samples = output_samples
        self._outputs_mu = output_mu
        self._outputs_sigma = output_sigma
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

    @property
    def outputs_samples(self):
        return self._outputs_samples

    @property
    def outputs_mu(self):
        return self._outputs_mu

    @property
    def outputs_sigma(self):
        return self._outputs_sigma

