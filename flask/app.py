from io import BytesIO
from PIL import Image # type: ignore

import torch # type: ignore

import base64

import cv2
import time
import os
from werkzeug.utils import send_from_directory # type: ignore

from flask import Flask, render_template, request, redirect, send_file, url_for, Response

from ultralytics import YOLO # type: ignore

import pygame # type: ignore
import pygame.camera # type: ignore

from pathlib import Path
from openai import OpenAI


DEFAULT_IMAGE_FILE = "autoimage.jpg"

SIBI = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index6.html')

@app.route("/start-predict", methods=["POST"])
def startPredict():
    data = request.get_json()
    imageBase64 = data["imageBase64"]

    # Remove the "data:image/jpeg;base64," prefix
    header, encoded = imageBase64.split(",", 1)

    # Decode the Base64 string
    img_data = base64.b64decode(encoded)

    # Convert the byte data into an image
    image = Image.open(BytesIO(img_data))

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    print("start predicting")
    return predict_img(image)

def predict_img(image):
    yolo = YOLO('best.pt')
    print("before detect")
    detections = yolo.predict(image, save=True)
    print(len(detections))
    for result in detections:
        print("result is =: ", result)
        tensorFormat = result.boxes.cls
        convertToList = tensorFormat.tolist()
        print(convertToList)
        
        result = "nothing"
        if len(convertToList) > 0:
            result = SIBI[int(convertToList[0])]
    
        return result
    
def takeImage():
    pygame.init()
    print("initializing camera")
    # initializing  the camera
    pygame.camera.init()

    print("camera initialized")

    camlist = []

    while not camlist:
        camlist = pygame.camera.list_cameras()
        if not camlist:
            time.sleep(0.1)  # Wait before retrying
            retry_count -= 1

    print("camlist: ", camlist)
    
    # if camera is detected or not 
    if camlist: 
        # initializing the cam variable with default camera 
        cam = pygame.camera.Camera(camlist[0], (640, 480)) 

        print("cam: ", cam)
    
        # opening the camera 
        cam.start()

        print("camera has started")
    
        # capturing the single image 
        image = cam.get_image() 

        print("image", image)
    
        # saving the image 
        pygame.image.save(image, "uploads/" + DEFAULT_IMAGE_FILE)
        print("images saved successfully")
    
    # if camera is not detected the moving to else part 
    else: 
        print("No camera on current device") 