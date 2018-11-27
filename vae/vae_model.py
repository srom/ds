import tensorflow as tf
import tensorflow_probability as tfp


class VariationalAutoEncoder(object):

    def __init__(self, x_size, encoding_size, hidden_layer_size=100, name='vae'):
        with tf.variable_scope('{name}/input'.format(name=name)):
            self.x = tf.placeholder(tf.float32, [None, x_size])
            self.learning_rate = tf.placeholder(tf.float32, ())
            self.num_samples = tf.placeholder(tf.int32, ())

        with tf.variable_scope('{name}/probabilities'.format(name=name), reuse=tf.AUTO_REUSE):
            self.prior = make_prior(encoding_size)
            self.posterior = make_encoder(self.x, encoding_size, hidden_layer_size)
            self.encoding = self.posterior.sample()
            self.likelihood = make_decoder(self.encoding, x_size, hidden_layer_size)
            self.samples = make_decoder(
                self.prior.sample(self.num_samples), x_size, hidden_layer_size).sample()

        with tf.variable_scope('{name}/loss'.format(name=name)):
            self.elbo_loss = make_elbo_loss(self.x, self.prior, self.posterior, self.likelihood)

        with tf.variable_scope('{name}/train'.format(name=name)):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.elbo_loss)


def make_prior(encoding_size):
    return tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(encoding_size),
        scale_diag=tf.ones(encoding_size),
        name='prior',
    )


def make_encoder(x, encoding_size, hidden_layer_size):
    hidden = tf.layers.dense(x, hidden_layer_size, tf.nn.relu)
    loc = tf.layers.dense(hidden, encoding_size)
    scale = tf.layers.dense(hidden, encoding_size, tf.nn.softplus)
    return tfp.distributions.MultivariateNormalDiag(
        loc=loc,
        scale_diag=scale,
        name='posterior',
    )


def make_decoder(encoding, x_size, hidden_layer_size):
    hidden = tf.layers.dense(encoding, hidden_layer_size, tf.nn.relu)
    loc = tf.layers.dense(hidden, x_size)
    scale = tf.layers.dense(hidden, x_size, tf.nn.softplus)
    return tfp.distributions.MultivariateNormalDiag(
        loc=loc,
        scale_diag=scale,
        name='likelihood',
    )


def make_elbo_loss(x, prior, posterior, likelihood):
    divergence = tfp.distributions.kl_divergence(posterior, prior)
    log_likelihood = likelihood.log_prob(x)
    return tf.reduce_mean(log_likelihood - divergence, name='elbo_loss')
