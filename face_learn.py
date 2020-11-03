import pickle
from os import listdir
from os.path import isdir

import cv2
from PIL import Image
from numpy import array, asarray, load
from numpy import expand_dims, savez_compressed
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import os
import time

import face_detect as fd
import face_recognition


def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    faces, h, w = fd.find_faces_dnn(pixels)
    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            dnn_box = faces[0, 0, i, 3:7] * array([w, h, w, h])
            sx, sy, ex, ey = dnn_box.astype("int")
            face = pixels[sy:ey, sx:ex]
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            return face_array
    return None


def load_faces(directory):
    faces = list()
    for i, filename in enumerate(listdir(directory)):
        print(f"Process in {directory}: {i+1}/{len(listdir(directory))}")
        path = directory + filename
        face = extract_face(path)
        if face is not None:
            faces.append(face)
    return faces


def load_dataset(directory):
    if len(listdir(directory)) < 2:
        print("2명 이상의 인물을 검색해주세요.")
        return None, None
    x, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue

        faces = load_faces(path)

        labels = [subdir for _ in range(len(faces))]

        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)


def face_learning():
    directory = 'data/IMG/'

    start = time.time()
    tTrainX, trainY = load_dataset(directory)
    if tTrainX is None:
        return
    trainX = list()
    for i, face_pixels in enumerate(tTrainX):
        print(f"Process in train : {i + 1}/{len(tTrainX)}")
        embedding = face_recognition.face_encodings(face_pixels, {(0, 159, 159, 0)})
        trainX.append(embedding)
    trainX = asarray(trainX)
    ttrainX = []
    for e in trainX:
        ttrainX.append(array(e).flatten())
    trainX = ttrainX
    savez_compressed('data/model.npz', trainX, trainY)

    learned_people = []
    for person in trainY:
        if person not in learned_people:
            learned_people.append(str(person))
            print(person)

    with open('data/modellist.bin', "wb") as f:
        pickle.dump(learned_people, f)
        f.close()

    end = time.time()
    print("Success make model", format(end - start, '.2f'), "s")


def make_model():
    path = 'data/model.npz'
    if not os.path.exists(path):
        print("Need Learning model")
        return None
    data = load('data/model.npz')

    trainX, trainY = data['arr_0'], data['arr_1']

    in_encoder = Normalizer(norm='max')
    trainX = in_encoder.transform(trainX)

    out_encoder = LabelEncoder()
    out_encoder.fit(trainY)
    trainY = out_encoder.transform(trainY)

    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainY)

    return model, in_encoder, out_encoder


def compare(model, in_encoder, out_encoder, face, names):
    face = cv2.resize(face, (160, 160))

    newtestX = list()
    embedding = face_recognition.face_encodings(face, {(0, 159, 159, 0)})
    newtestX.append(embedding)
    newtestX = asarray(newtestX)

    testX = []
    for t in newtestX:
        testX.append(array(t).flatten())

    testX = in_encoder.transform(testX)

    for i in range(testX.shape[0]):
        sample = expand_dims(testX[i], axis=0)
        yhat_class = model.predict(sample)
        yhat_prob = model.predict_proba(sample)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        print(f"Predict : {predict_names[0]}, {int(class_probability)}%")
        for name in names:
            if (class_probability >= 77) and (predict_names[0] == name):
                return name

    return None
