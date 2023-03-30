import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from testProject import getTFminiData


CATEGORIES = ["10 speed limit sign", "15 speed limit sign", "25 speed limit sign", "30 speed limit sign", "35 speed limit sign", "45 speed limit sign", "Truck (Slow sign)", "Pedestrian (Stop sign)",
              "Slow Sign (Wrong Way)", "Stop Sign (Yield sign)", "Vehicle (Car)", "Wrong Way (Truck) ", " Yield Sign (Ped)"]


# load model
model = tf.keras.models.load_model("final20val.model")

# video
video = cv2.VideoCapture(1)


video.set(cv2.CAP_PROP_FPS, 10)

while True:
    ret, frame = video.read()
    #if (ret == False):
    #    print("No camera")
    #    break

    # convert image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)


    # resize for input to model 256x256
    im = im.resize((256, 256))
    img_array = np.array(im)

    # change input to input tensor
    img_array = np.expand_dims(img_array, axis=0)

    # predict method
    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction)
    print(f"class: {prediction_class}, score: {prediction[0][prediction_class]}")
    if prediction[0][prediction_class] > 0.65:
        p = CATEGORIES[prediction_class]
        s = prediction[0][prediction_class]
        print(p)
        print(prediction_class)
        distance = getTFminiData()
        distance = str(distance) + " cm"
        text = str(p) + str(s)
        org = (50,50)
        #sorg = (50, 100)
        dorg = (50, 100)
        cv2.putText(frame, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (250,0,0))
        #cv2.putText(frame, s, sorg, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (250,0,0))
        cv2.putText(frame, distance, dorg, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (250,0,0))


    # if no input
    #if prediction:
    #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()