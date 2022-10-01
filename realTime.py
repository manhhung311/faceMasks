from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os.path as path

def detect_and_predict_mask(frame, faceNet, maskNet, faceNet1):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds_mask = []
    preds_face = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            hello = cv2.imwrite('2.jpg',face)
            print(hello)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds_mask = maskNet.predict(faces)
        preds_face = faceNet1.predict(faces)
    return (locs, preds_mask, preds_face)

# load our serialized face detector model from disk
prototxtPath = path.join(path.abspath(path.dirname(__file__)),'./face_detector/deploy.prototxt')
weightsPath = path.join(path.abspath(path.dirname(__file__)),'./face_detector/res10_300x300_ssd_iter_140000.caffemodel')
print(prototxtPath)
print(weightsPath)
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model(path.join(path.abspath(path.dirname(__file__)), "./model_mask.h5"))
faceNet1 = load_model(path.join(path.abspath(path.dirname(__file__)), "./model_face.h5"))
label = []
with open('./public/label.txt', 'r') as f:
    label = f.readlines()
print(label[0])
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds_mask, preds_face) = detect_and_predict_mask(frame, faceNet, maskNet, faceNet1)

    for (box, pred_mask, pred_face) in zip(locs, preds_mask, preds_face):

        (startX, startY, endX, endY) = box
        print("print" ,preds_mask)
        print("face", pred_face)
        pred_f=np.argmax(pred_face)

        color = (0,0,0)
        
        if(pred_mask[0] <= 0.5):
            color = (0, 0, 255)
            cv2.putText(frame,"khong khau trang - "+ label[pred_f], (startX, startY - 10),
            
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,color,2)
        else:
            color = (0, 255, 0)
            cv2.putText(frame,"co khau trang \n"+ label[pred_f], (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,color,2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()