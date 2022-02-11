import streamlit as st
import cv2
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import tensorflow as tf 
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import base64

st.set_page_config(page_title="Intelligent Smart Parking Application", page_icon=":car:", layout="wide")
st.sidebar.title("Navigation")
select_page=st.sidebar.selectbox("Pages", ["Parking Spot Detection","Drowsiness","License Plate Recognition", "License Plate Number"])

if select_page == "Parking Spot Detection":
    st.title("Detect Parking Spot")
    def preprocess(img,input_size):
        nimg = img.convert('RGB').resize(input_size, resample= 0)
        img_arr = (np.array(nimg))/255
        return img_arr

    def reshape(imgs_arr):
        return np.stack(imgs_arr, axis=0)

    def predict_class(image):
        cnn_model = tf.keras.models.load_model('model4.h5')
        input_size = (48, 48)
        class_names = ['empty','occupied']
        X = preprocess(image,input_size)
        X = reshape([X])
        predictions = cnn_model.predict(X)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        result = "The Image Uploaded is: {}".format(image_class)
        
        return result


    file_uploaded = st.file_uploader("Choose the file", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)


if select_page == "Drowsiness":
    emotion_dict = {0:'Driver Ngantuk', 1 :'Normal', 2: 'Normal', 3:'Driver Ngantuk'}
    emotion_dict2 = {0:'Driver Ngantuk', 1 :'Normal'}
    # load json and create model
    json_file = open('modeln.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)

    # load weights into new model
    classifier.load_weights('modeln.h5')

    # load json and create model
    json_file2 = open('model_mata.json', 'r')
    loaded_model_jsone = json_file2.read()
    json_file2.close()
    classifiere = model_from_json(loaded_model_jsone)

    # load weights into new model
    classifiere.load_weights('model_mata.h5')

    #load face
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye= cv2.CascadeClassifier('haarcascade_eye.xml')
    except Exception:
        st.write("Error loading cascade classifiers")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class Faceemotion(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            #image gray
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                image=img_gray, scaleFactor=1.3, minNeighbors=5)
            eyes = eye.detectMultiScale(
                image=img_gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img=img, pt1=(x, y), pt2=(
                    x + w, y + h), color=(211, 211, 211), thickness=2)
                roi_gray = img_gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
                for (ex, ey, ew, eh) in eyes:
                    roi_gray = img_gray[ey:ey + eh, ex:ex + ew]
                    roi_gray = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        prediction2 = classifiere.predict(roi)[0]
                        maxindex2 = int(np.argmax(prediction2))
                        if maxindex2 == 0:
                            clr =(0, 0, 255)
                        else:
                            clr =(0, 255, 0)
                        finalout2 = emotion_dict2[maxindex2]
                        output2 = str(finalout2)
                    label_position = (ex, ey)
                    cv2.putText(img, output2, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = classifier.predict(roi)[0]
                    maxindex = int(np.argmax(prediction))
                    if maxindex == 0 or maxindex == 3:
                        clr1 =(0, 0, 255)
                    else:
                        clr1 =(0, 255, 0)
                    finalout = emotion_dict[maxindex]
                    output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1,clr1, 2)
            return img

    def main():
    
        st.title("Drowsiness Detection") 
        #st.header("Webcam Live Feed")
        st.write("This project can detect Drowsiness Driver")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=Faceemotion)
    if __name__ == "__main__":
        main()


if select_page == "License Plate Recognition":
    st.title("Detect Plate Number Position")
    dada = st.radio("Source", ("Image", "Video"))
    if dada == "Image":

            def TFOD(img):
                yolo = cv2.dnn.readNet("model.weights", "darknet-yolov3.cfg")
                classes = []

                with open("classes.names", "r") as file:
                    classes = [line.strip() for line in file.readlines()]
                layer_names = yolo.getLayerNames()
                output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

                height, width, channels = img.shape

                #Detecting objects
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                yolo.setInput(blob)
                outputs = yolo.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 3)
                        cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2)
                return img
            
            u = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])
            if u is not None:
                imag = Image.open(u)
                st.image(u)
                # image = cv2.imread(im)
                input_image = np.array(imag.convert('RGB'))
                image = TFOD(input_image)
                height, width = image.shape[:2]
                resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

                fig = plt.gcf()
                fig.set_size_inches(18, 10)
                plt.axis("off")
                # cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                st.image(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    
    if dada == "Video":
        video1 = open("output1.mp4", 'rb')
        st.video(video1)




if select_page == "License Plate Number":
        st.title("Extract Plate Number into Text")
        i = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])
        if i is not None:
            imag = Image.open(i)
            st.image(i)
            # image = cv2.imread(im)
            image = np.array(imag.convert('RGB'))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 11,90, 90)
            edges = cv2.Canny(blur, 30, 200)
            cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            image_copy = image.copy()
            _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
            image_copy = image.copy()
            _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
            plate = None
            for c in cnts:
                perimeter = cv2.arcLength(c, True)
                edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
                if len(edges_count) == 4:
                    x,y,w,h = cv2.boundingRect(c)
                    plate = image[y:y+h, x:x+w]
                    break
        
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            text = pytesseract.image_to_string(plate, lang='eng', config="--psm 7")
            st.write("Plat Nomor: ", text)
   



