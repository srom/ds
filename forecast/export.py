import logging

import boto3
import tensorflow as tf
from tensorflow.python.framework import graph_util


BUCKET_NAME = 'forecast'
OUTPUT_KEY_NAME = 'model/{name}.pb'
OUTPUT_KEY_NAME_BEST = 'model/{name}_best.pb'
LOCAL_EXPORT = 'forecast/model/{name}.pb'


logger = logging.getLogger(__name__)


def export_model(saver, model_save_path, name, output_node_names, local=False, best=False):
    if local:
        logger.info('Exporting model to %s', LOCAL_EXPORT.format(name=name))
    else:
        logger.info('Exporting model to S3')

    with tf.Session() as session:
        saver.restore(session, model_save_path)

        output_graph_def = graph_util.convert_variables_to_constants(
            session,
            session.graph.as_graph_def(),
            output_node_names,
        )

        if local:
            export_model_to_disk(output_graph_def, name)
        else:
            export_model_to_s3(output_graph_def, name, best=best)

    logger.info('Export completed')


def export_model_to_s3(output_graph_def, name, best=False):
    s3 = boto3.resource('s3', region_name='eu-west-1')
    bucket = s3.Bucket(BUCKET_NAME)
    key_name = (OUTPUT_KEY_NAME if not best else OUTPUT_KEY_NAME_BEST).format(name=name)
    bucket.put_object(Key=key_name, Body=output_graph_def.SerializeToString())
    logger.info('Model exported to s3://%s/%s', BUCKET_NAME, key_name)


def export_model_to_disk(output_graph_def,  name):
    with open(LOCAL_EXPORT.format(name=name), 'wb') as f:
        f.write(output_graph_def.SerializeToString())
