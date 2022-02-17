from turtle import width
import cv2
import tensorflow.lite as tflite
import numpy as np
import re

from PIL import Image

vid = cv2.VideoCapture(0)
interpreter = tflite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()
DETECTION_THRESHOLD = 0.97

input = interpreter.get_input_details()
out = interpreter.get_output_details()

input_shape = input[0]['shape']
height = input_shape[1]
width = input_shape[2]

input_index = input[0]['index']

cap = cv2.VideoCapture(0)


def load_labels(label_path):
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def process_image(interpreter, img, in_index):
    in_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(in_index, in_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()

    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    # this is an empty array to store results
    results = []

    print(np.average(scores))

    if len(positions) == 0:
        return results

    for idx, score in enumerate(positions):
        if score > DETECTION_THRESHOLD:
            results.append(
                {'pos': positions[idx], 'class': classes[idx], 'score': score})

    return results


while (True):
    ret, img = cap.read()
    if ret:
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = image.resize((height, width))

        results = process_image(interpreter, image, input_index)

        print("Found {}".format(results))

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv2.release()

cv2.destroyAllWindows()
