# face detection with mtcnn on a photograph
from mtcnn.mtcnn import MTCNN
import cv2
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

mtcnn_detector = MTCNN()

protextPath = "data/DNN/deploy.prototxt"
caffeModelPath = "data/DNN/res10_300x300_ssd_iter_140000.caffemodel"
dnn_detector = cv2.dnn.readNetFromCaffe(protextPath, caffeModelPath)

img = cv2.imread("data/Dummy.jpg")
mtcnn_detector.detect_faces(img)
print("Detector Initialize")