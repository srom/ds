import logging

import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing

from .vae_model import VariationalAutoEncoder, train_vae


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(levelname)s) %(message)s')

    dataset_path = 'vae/data/epileptic_seizure_dataset.csv'

    logger.info(f'Loading dataset {dataset_path}')

    dataset = pd.read_csv(dataset_path)

    X = dataset[dataset.columns[1:-1]].values
    y = dataset['y']

    scaler = sklearn.preprocessing.StandardScaler()
    X_norm = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X_norm, y, test_size=0.2, random_state=444)

    vae_tril = VariationalAutoEncoder(178, 10, multivariate_tril_decoder=True, name='vae_tril')

    save_path = train_vae(vae_tril, X_train, X_test, 0.001, 20000, 100, name='vae_tril')

    logger.info(f'Results saved to {save_path}')
    logger.info('Done')


if __name__ == '__main__':
    main()
