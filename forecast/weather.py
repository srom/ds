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
LEARNING_RATE = 1e-3
BATCH_SIZE = 50

logger = logging.getLogger(__name__)


def train_weather_forecast_model(
    weather_data_path=WEATHER_DATA_PATH,
    model_dir='checkpoints',
    export_local=False,
):
    logger.info(f'Loading weather data from {weather_data_path}')
    X, _ = get_weather_data(weather_data_path)
    X_train, X_test = get_train_test_sets(X)

    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_dir, 'weather')

    model = LSTMNetwork(NUM_INPUTS, NUM_OUTPUTS, TIME_STEPS, NUM_NEURONS, LEARNING_RATE, adam_epsilon=1.0)

    saver = tf.train.Saver()
    best_mse = float('Inf')

    logger.info('Training')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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


def forecast_weather(X):
    predictions = None
    with load_lstm_model() as lstm:
        for i, x in enumerate(iterate_over_window(X)):
            if len(x) != TIME_STEPS:
                break

            if i % 1000 == 0:
                print(f'Iteration {i}')
            x_forecast = lstm.evaluate([x])[0]

            if predictions is None:
                predictions = np.array([x_forecast[-1]])
            else:
                predictions = np.append(predictions, [x_forecast[-1]], axis=0)

    return predictions


def forecast_weather_one_window(x):
    with load_lstm_model() as lstm:
        return lstm.evaluate([x])[0]


def get_weather_data(weather_data_path):
    dt_column = 'Date/Time'
    weather_df = pd.read_csv(weather_data_path, parse_dates=[dt_column])
    weather_df.set_index(dt_column, inplace=True)
    weather_dedup_df = weather_df[~weather_df.index.duplicated(keep='first')]
    clean_df = weather_dedup_df[WEATHER_COLUMNS].dropna()
    X = clean_df.values
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def get_train_test_sets(X, ratio=0.3):
    last_slice = int(len(X) * ratio)
    return X[:-last_slice,:], X[-last_slice:,:]


def next_batch(X, batch_size, time_steps=TIME_STEPS):
    def get_X(m):
        return m[:-1, :]

    def get_y(m):
        return m[1:, :]

    X__, y__ = [], []
    for x in np.random.randint(0, len(X) - (time_steps + 1), size=batch_size):
        matrix = X[x:x+time_steps+1]
        X__.append(get_X(matrix))
        y__.append(get_y(matrix))

    return np.array(X__), np.array(y__)


def load_lstm_model():
    """
    Use as a context manager or explicitly close after use.
    """
    return load_model(
        path='forecast/model/weather.pb',
        input_name='weather/lstm/input/X:0',
        evaluator_name='weather/lstm/rnn/dynamic_rnn/transpose_1:0',
    )


class LSTMForecast(object):

    def __init__(self, session, x, f_x):
        self.session = session
        self.x = x
        self.f_x = f_x

    def evaluate(self, x):
        return self.session.run(self.f_x, feed_dict={self.x: x})

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_model(path, input_name, evaluator_name):
    """
    Use as a context manager or explicitly close after use.
    """
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='weather')
        session = tf.Session(graph=graph)

        x = graph.get_tensor_by_name(input_name)
        f_x = graph.get_tensor_by_name(evaluator_name)

        return LSTMForecast(session, x, f_x)


def iterate_over_window(X, time_steps=TIME_STEPS):
    for i in range(len(X)):
        yield X[i:time_steps + i,:]
