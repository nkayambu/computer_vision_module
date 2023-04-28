import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


CATEGORIES = ["10 speed limit sign", "15 speed limit sign", "25 speed limit sign", "30 speed limit sign", "35 speed limit sign", "45 speed limit sign", "Stop Sign"]


# load model
model = tf.keras.models.load_model("7_class.model")

# video
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

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
    print (f"class: {prediction_class}, score: {prediction[0][prediction_class]}")
    if prediction[0][prediction_class] > 0.85:
        p = CATEGORIES[prediction_class]
        s = prediction[0][prediction_class]
        text = str(p) + str(s)
        print(p)
        print(prediction_class)
        org = (50,50)
        cv2.putText(frame, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (0,250,0))
        #cv2.putText(frame, s, org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(250, 0, 0))


    # if no input
    #if prediction:
    #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()