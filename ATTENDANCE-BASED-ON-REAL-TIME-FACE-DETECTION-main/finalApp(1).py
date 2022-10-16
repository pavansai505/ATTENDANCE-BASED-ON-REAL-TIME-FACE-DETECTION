import os
import pickle
import cv2
import time
from sklearn.svm import SVC
import face_recognition
import imutils
import datetime
import numpy as np
from imutils import paths
from imutils.video import FPS
from imutils.video import VideoStream
import dlib

dlib.DLIB_USE_CUDA = True

# generate faces from images paths source and destination
path_for_images = "D:/face-recognition-using-deep-learning-master/vediophotoextract"
path_to_generated_faces = "D:/face-recognition-using-deep-learning-master/dataset/"

# path for generating vectors from images
path_source_faces = "D:/face-recognition-using-deep-learning-master/dataset"
outputpath_for_generated_pickle = "D:/face-recognition-using-deep-learning-master/output"
name_of_pickle = "encode"

# paths for pickle files
using_encode_pickle = os.path.join(outputpath_for_generated_pickle + '/' + name_of_pickle + ".pickle")
generating_trainmodel = os.path.join(outputpath_for_generated_pickle + "/trainmodel.pickle")

# paths for live models


# getting model paths and datsets
Model_wieghts = "D:/face-recognition-using-deep-learning-master/face_detection_model/deploy.prototxt"
Model = "D:/face-recognition-using-deep-learning-master/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
Face_detector = cv2.dnn.readNetFromCaffe(Model_wieghts, Model)


# model for face extraction
def generating_faces_from_image(path_for_images, path_to_generated_faces, Face_detector):
    imagePaths = list(paths.list_images(path_for_images))
    total = 0
    for (i, imagePath) in enumerate(imagePaths):
        print("Processing image {}/{}".format(i, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        source = os.path.join(path_to_generated_faces + name)
        try:
            if not os.path.exists(source):
                os.makedirs(source)
        except OSError:
            pass
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                          swapRB=False, crop=False)
        Face_detector.setInput(imageBlob)
        detections = Face_detector.forward()
        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                try:
                    face = image[startY:endY, startX:endX]
                    cv2.imwrite(source + '/' + str(total) + ".jpg", face)
                except:
                    pass
                total += 1
    print("completed face extract")


# model for vectorization
def generate_vectors(path_source_faces, outputpath_for_generated_pickle, name_of_pickle):
    knownNames = []
    encodes = []
    total = 0
    outputsource = os.path.join(outputpath_for_generated_pickle + '/' + name_of_pickle + ".pickle")
    source_face = list(paths.list_images(path_source_faces))
    for (i, source_face) in enumerate(source_face):
        print("Processing image {}".format(i))
        name = source_face.split(os.path.sep)[-2]
        image = cv2.imread(source_face)
        gg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(gg)
        for e in encode:
            encodes.append(e)
            knownNames.append(name)
        total += 1
    data = {"encodings": encodes, "names": knownNames}
    oo = open(outputsource, "wb")
    oo.write(pickle.dumps(data))
    oo.close()
    print("completed")


# model for training
def train_model(using_encode_pickle, generator_trainmodel):
    data = pickle.loads(open(using_encode_pickle, "rb").read())
    recognizer = SVC(C=1.0, kernel='rbf', decision_function_shape='ovr')
    recognizer.fit(data["encodings"], data["names"])
    f = open(generator_trainmodel, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    print("completed")


# model for prediction
def live_camera_attendance(Face_detector, p, k):
    recognizer = pickle.loads(open(generating_trainmodel, "rb").read())
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                          swapRB=False, crop=False)
        Face_detector.setInput(imageBlob)
        detections = Face_detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                vec = face_recognition.face_encodings(face, num_jitters=1, model="small")
                for i in zip(vec):
                    preds = recognizer.predict(i)
                    p.append(preds[0])
                    text = "{}".format(preds[0])
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        fps.update()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    k.append(datetime.datetime.now())
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()
    print("completed")
    return p, k


# Driver program
if __name__ == "__main__":
    print("press: 1.Face extraction /2.Vectorization /3.Training /4.LiveCam")
    count = int(input())
    p = list()
    k = list()
    if count == 1:
        print("generating faces from images")
        generating_faces_from_image(path_for_images, path_to_generated_faces, Face_detector)
    elif count == 2:
        print("generating vectors for images")
        generate_vectors(path_source_faces, outputpath_for_generated_pickle, name_of_pickle)
    elif count == 3:
        print("training the model")
        train_model(using_encode_pickle, generating_trainmodel)
    else:
        print("starting live camera")
        p, k = live_camera_attendance(Face_detector, p, k)
    d = dict()
    for i in p:
        d[i] = d.get(i, 0) + 1
    print(d, k)
