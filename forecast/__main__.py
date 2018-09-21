import argparse
import logging
import sys

from .weather import train_weather_forecast_model


logger = logging.getLogger(__name__)


def forecast_electricity():
    raise NotImplementedError


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser(description='Forecast weather and electricity consumption')
    parser.add_argument('forecast_type')
    parser.add_argument('--export-local', '--export_local', default=False, action='store_true')
    args = parser.parse_args()
    forecast_type = args.forecast_type
    export_local = args.export_local

    if forecast_type not in ['weather', 'electricity']:
        logger.error('Forecast type must be one of: weather, electricity')
        sys.exit(1)

    if forecast_type == 'weather':
        train_weather_forecast_model(export_local=export_local)
    else:
        forecast_electricity()


if __name__ == '__main__':
    main()
