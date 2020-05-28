import tensorflow as tf
from preprocessing import load_and_process_data
import numpy as np
converter = tf.lite.TFLiteConverter.from_saved_model("models/full_dataset_nn")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
dataset, labels, attributes = load_and_process_data("Datasets/svelteSinkhole12combinedCoojasMitMTrainKDDTrain+_filtered.arff", do_normalize=True)
# dataset = np.column_stack((dataset, labels))
print(dataset.shape)
tf_dataset = tf.data.Dataset.from_tensor_slices((dataset.astype(np.float32))).batch(1)

def representative_data_gen():
    for input_value in tf_dataset.take(100):
        yield [input_value]


#representative_data_gen()
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
open("models/full_dataset_lite", "wb").write(tflite_model)
