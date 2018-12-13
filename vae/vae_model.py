import logging
import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


logger = logging.getLogger(__name__)


class VariationalAutoEncoder(object):

    def __init__(
        self,
        x_size,
        encoding_size,
        hidden_layer_size=100,
        multivariate_tril_decoder=False,
        name='vae',
    ):
        with tf.variable_scope('{name}/input'.format(name=name)):
            self.x = tf.placeholder(tf.float64, [None, x_size])
            self.learning_rate = tf.placeholder(tf.float32, ())
            self.num_samples = tf.placeholder(tf.int32, ())

        with tf.variable_scope('{name}/probabilities'.format(name=name), reuse=tf.AUTO_REUSE):
            self.prior = make_prior(encoding_size)
            self.posterior = make_encoder(self.x, encoding_size, hidden_layer_size)
            self.encoding = self.posterior.sample()
            self.likelihood = make_decoder(
                self.encoding,
                x_size,
                hidden_layer_size,
                multivariate_tril_decoder,
            )
            self.samples = make_decoder(
                self.prior.sample(self.num_samples),
                x_size,
                hidden_layer_size,
                multivariate_tril_decoder,
            ).sample()

        with tf.variable_scope('{name}/loss'.format(name=name)):
            self.elbo_loss = make_elbo_loss(self.x, self.prior, self.posterior, self.likelihood)

        with tf.variable_scope('{name}/train'.format(name=name)):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.elbo_loss)

        with tf.variable_scope('summary'):
            tf.summary.scalar('elbo', self.elbo_loss)
            self.summary = tf.summary.merge_all()


def make_prior(encoding_size):
    return tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(encoding_size, dtype=tf.float64),
        scale_diag=tf.ones(encoding_size, dtype=tf.float64),
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


def make_decoder(encoding, x_size, hidden_layer_size, multivariate_tril_decoder=False):
    hidden = tf.layers.dense(encoding, hidden_layer_size, tf.nn.relu)

    if multivariate_tril_decoder:
        return tfp.trainable_distributions.multivariate_normal_tril(
            x=hidden,
            dims=x_size,
            name='likelihood',
        )
    else:
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


####
# Training functions
###


class RunSummary(object):

    def __init__(self, iteration, save_path, best_elbo):
        self.iteration = iteration
        self.save_path = save_path
        self.best_elbo = best_elbo


def random_draw(x, y=None, batch_size=100):
    indices = np.random.randint(0, len(x), batch_size)
    if y is not None:
        return x[indices], y[indices]
    else:
        return x[indices]


def train_vae(
    vae_cls,
    x,
    x_test,
    learning_rate,
    n_epochs,
    batch_size,
    prev_run_summary=None,
    name='vae',
    log_every=1000,
    summary_log_path='./summary_log',
):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if prev_run_summary is not None:
            saver.restore(sess, prev_run_summary.save_path)
        else:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

        summary_writer = tf.summary.FileWriter(summary_log_path, sess.graph)

        save_path = prev_run_summary.save_path if prev_run_summary else None
        best_elbo = prev_run_summary.best_elbo if prev_run_summary else float('-inf')
        iteration = prev_run_summary.iteration if prev_run_summary else 0

        for i, epoch in enumerate(range(n_epochs)):
            iteration += i
            x_batch = random_draw(x, batch_size=batch_size)

            sess.run(
                vae_cls.train_op,
                feed_dict={
                    vae_cls.x: x_batch,
                    vae_cls.learning_rate: learning_rate,
                },
            )

            if i % log_every == 0:
                elbo_test, summary = sess.run(
                    [vae_cls.elbo_loss, vae_cls.summary],
                    feed_dict={vae_cls.x: x_test},
                )

                summary_writer.add_summary(summary, global_step=iteration)

                if not math.isnan(elbo_test) and elbo_test > best_elbo:
                    save_path = saver.save(sess, f'./{name}.ckpt')
                    best_elbo = elbo_test

                logger.info(f'{i} / {n_epochs}: latest = {elbo_test} | best = {best_elbo}')

        return RunSummary(iteration, save_path, best_elbo)


def encoding(vae, x, save_path):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        return sess.run(vae.encoding, {vae.x: x})


def generate(vae, num_samples, save_path):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
