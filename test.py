import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

model_path = "./model/mobilenet.tflite"
label_path = "./model/labels.txt"
image_test_path = "./test_images/test.jpg"


def predict(image_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = cv2.imread(image_path)
    image = cv2.resize(
        image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_path)

    for i in top_k:
        print(labels[i], results[i])

    return labels[top_k[0]]

# load labels from label file


def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=image_test_path)
    args = parser.parse_args()
    predict(args.image_path)
    print("Done")

    # os.system("python3 test.py --image_path ./test_images/test.jpg")
