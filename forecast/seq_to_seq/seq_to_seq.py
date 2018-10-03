import tensorflow as tf


GO_TOKEN = -100.0


class EncoderDecoder(object):

    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_steps,
            n_neurons,
            batch_size,
            learning_rate,
            adam_epsilon=1e8,
            name='encoder_decoder',
    ):
        with tf.variable_scope(f'{name}/input'):
            self.X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
            self.Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='Y')

        with tf.variable_scope(f'{name}/encoder'):
            encoder_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
            self.encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.X, dtype=tf.float32)

        def decode(helper, reuse=None):
            with tf.variable_scope(f'{name}/decoder', reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=n_neurons,
                    memory=self.encoder_outputs,
                )
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                    tf.contrib.rnn.GRUCell(num_units=n_neurons),
                    attention_mechanism,
                )
                decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attention_cell,
                    output_size=n_outputs,
                    reuse=reuse,
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=encoder_state,
                )
                outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    impute_finished=True,
                )
                return outputs

        dec_input = _prepend_go_tokens(self.Y, GO_TOKEN, batch_size)

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=dec_input,
            sequence_length=[n_outputs] * batch_size
        )

        start_tokens = tf.constant(GO_TOKEN, shape=[batch_size, 1])

        inference_helper = tf.contrib.seq2seq.InferenceHelper(
            sample_fn=lambda outputs: outputs,
            sample_shape=[1],
            sample_dtype=tf.float32,
            start_inputs=start_tokens,
            end_fn=lambda sample_ids: False,
        )

        self.train_outputs = decode(training_helper)
        self.pred_outputs = decode(inference_helper, reuse=True)

        with tf.variable_scope(f'{name}/loss'):
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.train_outputs, self.Y)), name='mse')

        with tf.variable_scope(f'{name}/train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
            self.training_op = optimizer.minimize(self.loss)

        with tf.variable_scope('summary'):
            tf.summary.scalar('mse', tf.reduce_mean(self.loss))

        self.summary = tf.summary.merge_all()


def _prepend_go_tokens(output_data, go_token, batch_size):
    go_tokens = tf.constant(
        go_token,
        shape=[batch_size, 1, 1],
    )
    return tf.concat([go_tokens, output_data], axis=1)
