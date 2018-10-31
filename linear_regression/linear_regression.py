import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
from sklearn import preprocessing
import tensorflow as tf
import tensorflow_probability as tfp


def noisy_line(a=2.0, b=3.0, sigma=0.25, size=100, support=(-1, 1)):
    """
    Generate 2D line with added normal error ~ N(0, sigma**2)
    """
    x = np.linspace(*support, num=size)
    errors = scipy.stats.norm.rvs(loc=0, scale=sigma, size=size)
    y = a * x + b + errors
    return x, y


class BayesianLinearRegression(object):

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None])
        self.y = tf.placeholder(tf.float32, [None])

        self.slope = tf.get_variable('slope', dtype=tf.float32, shape=())
        self.intercept = tf.get_variable('intercept', dtype=tf.float32, shape=())
        self.sigma = tf.get_variable('sigma', dtype=tf.float32, shape=())

        self.y_hat = tf.add(tf.multiply(self.slope, self.x), self.intercept)

        self.log_likelihood = - tf.log(self.sigma) - tf.divide(
            tf.square(self.y - self.y_hat), 2 * tf.square(self.sigma))

        # prior is implictly set to uniform with amplitude 1.0
        self.log_posterior = tf.reduce_sum(self.log_likelihood)

    def run_log_posterior(self, session, x, y, parameters):
        return session.run(
            self.log_posterior,
            feed_dict={
                self.x: x,
                self.y: y,
                self.slope: parameters[0],
                self.intercept: parameters[1],
                self.sigma: parameters[2],
            }
        )

    def metropolis_hasting(
        self,
        session,
        x,
        y,
        num_samples,
        initial_parameters,
        proposal_widths,
        bounds,
        burnin=0,
    ):
        if burnin < 0:
            raise ValueError('burnin must be greather than or equal to zero')

        current_parameters = initial_parameters
        draws = []
        current_log_post_cached_ = self.run_log_posterior(session, x, y, current_parameters)
        for i in range(num_samples + burnin):
            proposal_parameters = np.random.multivariate_normal(
                mean=current_parameters,
                cov=np.diag(proposal_widths),
            )

            for k, param_bounds in enumerate(bounds):
                min_, max_ = param_bounds[0], param_bounds[1]

                if proposal_parameters[k] < min_:
                    proposal_parameters[k] = min_
                elif proposal_parameters[k] > max_:
                    proposal_parameters[k] = max_

            log_post_current = current_log_post_cached_
            log_post_proposal = self.run_log_posterior(session, x, y, proposal_parameters)

            u = np.random.uniform(0.0, 1.0)

            p_accept =  log_post_proposal - log_post_current

            if p_accept > np.log(u):
                current_parameters = proposal_parameters
                current_log_post_cached_ = log_post_proposal

            draws.append((current_parameters, current_log_post_cached_))

        return draws[burnin:]


def plot_line(x, y=None, ys=None, y_hat=None, draws=None):
    f, ax = plt.subplots(1, 1, figsize=(15, 6))

    palette = sns.color_palette('colorblind')

    if ys is not None:
        ax.plot(x, ys, 'o', label='Noisy data', color=palette[0])

    if y is not None:
        ax.plot(x, y, label='Actual line', color=palette[0])

    if draws is not None:
        ax.plot(x, draws, 'o', label='Draws', color=palette[2])

    if y_hat is not None:
        ax.plot(x, y_hat, label='Fitted line', color=palette[2])

    ax.legend()
    return f, ax
