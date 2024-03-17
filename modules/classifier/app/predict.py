from datetime import datetime
from urllib.request import urlopen

import numpy as np
import PIL.Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'

od_model = None
labels = None


class ObjectDetection:
    OUTPUT_TENSOR_NAMES = ['detected_boxes', 'detected_scores', 'detected_classes']

    def __init__(self, model_filename):
        self._interpreter = tflite.Interpreter(model_path=model_filename)
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        assert len(input_details) == 1
        self._input_index = input_details[0]['index']

        # Get dictionary with output details {name: index}
        output_name_index = {d['name']: d['index'] for d in output_details}
        self._output_indexes = [output_name_index[name] for name in self.OUTPUT_TENSOR_NAMES]

        self._input_size = int(input_details[0]['shape'][1])

    def predict_image(self, image):
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = image.resize((self._input_size, self._input_size))

        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        self._interpreter.set_tensor(self._input_index, input_array)
        self._interpreter.invoke()

        outputs = [self._interpreter.get_tensor(i) for i in self._output_indexes]
        return outputs


def initialize():
    global od_model
    od_model = ObjectDetection(MODEL_FILENAME)
    global labels
    with open(LABELS_FILENAME) as f:
        labels = [label.strip() for label in f.readlines()]


def predict_url(image_url):
    with urlopen(image_url) as binary:
        image = PIL.Image.open(binary)
        return predict_image(image)


def predict_image(image):
    assert od_model is not None
    assert labels is not None

    predictions = od_model.predict_image(image)

    predictions = [{'probability': round(float(p[1]), 8),
                    'tagId': int(p[2]),
                    'tagName': labels[p[2]],
                    'boundingBox': {
                        'left': round(float(p[0][0]), 8),
                        'top': round(float(p[0][1]), 8),
                        'width': round(float(p[0][2] - p[0][0]), 8),
                        'height': round(float(p[0][3] - p[0][1]), 8)
                        }
                    } for p in zip(*predictions)]

    response = {'id': '', 'project': '', 'iteration': '', 'created': datetime.utcnow().isoformat(),
                'predictions': predictions}

    print("Resuls: " + str(response))
    return response
