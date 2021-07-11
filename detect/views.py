from django.http.response import HttpResponseServerError, StreamingHttpResponse
from django.shortcuts import render
import cv2
import threading
from django.views.decorators import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent

# Create your views here.
def index_view(request):
    return render(request, "detect/index.html")

def about_view(request):
    return render(request, "detect/about.html")


def detect_mask(frame, face_nn, mask_nn):
    (h,w) = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(frame,1.0,(224,224), (104.0,177.0,123.0))
    face_nn.setInput(blob)
    detections = face_nn.forward()

    faces = []
    locations = []
    predictions = []

    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.25:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0,startX),max(0,startY)
            endX, endY = min(w-1, endX), min(h-1, endY)

            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.resize(face, (128,128))
                face = face[...,::-1].astype(np.float32) / 255.0
                face = np.reshape(face, (1,128,128,3))
                faces.append(face)
                locations.append((startX, startY, endX, endY))
                prediction = mask_nn.predict(face)
            except:
                continue
            
            predictions.append(prediction)
        
    return locations,predictions


def get_frame():
    mapping = {0:"mask worn incorrectly",1:"with mask",2:"without mask"}
    prototxtPath = str(BASE_DIR) +"/detect/face_detector/deploy.prototxt"
    weightsPath = str(BASE_DIR) + "/detect/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    face_nn = cv2.dnn.readNet(prototxtPath,weightsPath)
    mask_nn = load_model(str(BASE_DIR) + "/detect/model_nn")

    camera = cv2.VideoCapture(0)
    while True:
        _, img = camera.read()
        locations, predictions = detect_mask(img, face_nn, mask_nn)
        for box, prediction in zip(locations, predictions):
            startX, startY, endX, endY = box
            prediction = np.argmax(prediction,axis=-1)[0]
            if prediction == 0:
                color = (0,165,255)
            elif prediction == 1:
                color = (255,0,0)
            else:
                color = (0,0,255)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img, mapping[prediction],(startX, startY - 20),cv2.FONT_HERSHEY_COMPLEX, 0.55, color, 2)
        imgencode = cv2.imencode('.jpg',img)[1]

        stringData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del camera

@gzip.gzip_page
def dynamic_stream(request,stream_path="video"):
    try:
        return StreamingHttpResponse(get_frame(),content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        return "error"

def test_view(request):
    try:
        return render(request,"detect/test.html")
    except:
        print("error")

"""


@gzip.gzip_page
def test_view(request):
    mapping = {0:"mask worn incorrectly",1:"with mask",2:"without mask"}
    prototxtPath = str(BASE_DIR) +"/detect/face_detector/deploy.prototxt"
    weightsPath = str(BASE_DIR) + "/detect/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

    face_nn = cv2.dnn.readNet(prototxtPath,weightsPath)

    mask_nn = load_model( str(BASE_DIR) + "/detect/model_nn")
    try:
        cam = VideoCamera(face_nn,mask_nn,mapping)
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    del cam
    return render(request, "detect/about.html")

class VideoCamera(object):
    def __init__(self,face_nn,mask_nn,mapping):
        self.face_nn = face_nn
        self.mask_nn = mask_nn
        self.mapping = mapping
        self.video = cv2.VideoCapture(0)
        self.grabbed, self.frame = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
    def get_frame(self):
        image = self.frame
        locations, predictions = self.detect_mask(image,self.face_nn,self.mask_nn) 
        for box, prediction in zip(locations, predictions):
            startX, startY, endX, endY = box
            if prediction == 0:
                color = (0,165,255)
            elif prediction == 1:
                color = (255,0,0)
            else:
                color = (0,0,255)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            cv2.putText(image, self.mapping[prediction],(startX, startY - 20),cv2.FONT_HERSHEY_COMPLEX, 0.55, color, 2)
        _, jpg = cv2.imencode('.jpg', image)
        return jpg.tobytes()
    def update(self):
        while True:
            self.grabbed, self.frame = self.video.read()

    def detect_mask(self,frame, face_nn, mask_nn):
        (h,w) = frame.shape[:2]
        blob  = cv2.dnn.blobFromImage(frame,1.0,(224,224), (104.0,177.0,123.0))
        face_nn.setInput(blob)
        detections = face_nn.forward()

        faces = []
        locations = []
        predictions = []

        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.2:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                startX, startY, endX, endY = box.astype("int")
                startX, startY = max(0,startX),max(0,startY)
                endX, endY = min(w-1, endX), min(h-1, endY)

                face = frame[startY:endY, startX:endX]
                try:
                    face = cv2.resize(face, (128,128))
                    face = face[...,::-1].astype(np.float32) / 255.0
                    face = np.reshape(face, (1,128,128,3))
                    faces.append(face)
                    locations.append((startX, startY, endX, endY))
                    prediction = mask_nn.predict(face)
                    prediction = np.argmax(prediction, axis=-1)[0]
                    predictions.append(prediction)
                except:
                    print("huh")
            
        return (locations,predictions)

def gen(camera):
    seconds = 15
    timeout = time.time() + seconds
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
        )
        time.sleep(0.01)
        if time.time() > timeout:
            break
    camera.video.release()
    return {{{1:2}}}
"""