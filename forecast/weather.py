import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from .export import export_model
from .rnn.network import LSTMNetwork


WEATHER_DATA_PATH = '~/workspace/ds/data/AMPds/Climate_HourlyWeather.csv'
WEATHER_COLUMNS = ['Temp (C)', 'Dew Point Temp (C)', 'Visibility (km)', 'Stn Press (kPa)']
TIME_STEPS = 24
NUM_INPUTS = 4
NUM_OUTPUTS = 4
NUM_NEURONS = 128
LEARNING_RATE = 1e-2
BATCH_SIZE = 100

logger = logging.getLogger(__name__)


def train_weather_forecast_model(
        weather_data_path=WEATHER_DATA_PATH,
        model_dir='checkpoints',
        export_local=False,
):
    logger.info(f'Loading weather data from {weather_data_path}')
    X, scaler = get_weather_data(weather_data_path)
    X_train, X_test = get_training_test_sets(X)

    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_dir, 'weather')

    model = LSTMNetwork(NUM_INPUTS, NUM_OUTPUTS, TIME_STEPS, NUM_NEURONS, LEARNING_RATE)

    saver = tf.train.Saver()
    best_mse = float('Inf')

    logger.info('Training')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        summary_writer = tf.summary.FileWriter('forecast/summary_log/train', sess.graph)

        iteration = 0
        while 1:
            iteration += 1
            X_batch, y_batch = next_batch(X_train, BATCH_SIZE)
            sess.run(model.training_op, feed_dict={model.X: X_batch, model.Y: y_batch})

            mse, summary = sess.run(
                [model.loss, model.summary],
                feed_dict={model.X: X_batch, model.Y: y_batch})

            summary_writer.add_summary(summary, global_step=iteration)

            if iteration % 100 == 0:
                logger.info(f'{iteration}\tMSE: {mse}\tBest: {best_mse}')
                saver.save(sess, save_path, global_step=iteration)

            if export_local and best_mse > mse:
                best_mse = mse
                model_save_path = saver.save(sess, save_path, global_step=iteration)
                export_model(saver, model_save_path, 'weather', [model.rnn_outputs.name[:-2]], local=True)


def get_weather_data(weather_data_path):
    dt_column = 'Date/Time'
    weather_df = pd.read_csv(weather_data_path, parse_dates=[dt_column])
    weather_df.set_index(dt_column, inplace=True)
    weather_dedup_df = weather_df[~weather_df.index.duplicated(keep='first')]
    clean_df = weather_dedup_df[WEATHER_COLUMNS].dropna()
    X = clean_df.values
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def get_training_test_sets(X, ratio=0.3):
    last_slice = int(len(X) * ratio)
    return X[:-last_slice,:], X[-last_slice:,:]


def next_batch(X, batch_size):
    def get_X(m):
        return m[:-1, :]

    def get_y(m):
        return m[1:, :]

    X__, y__ = [], []
    for x in np.random.randint(len(X) - (TIME_STEPS + 1), size=batch_size):
        matrix = X[x:x+TIME_STEPS+1]
        X__.append(get_X(matrix))
        y__.append(get_y(matrix))

    return np.array(X__), np.array(y__)
