import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import time

# Load the saved model
from tensorflow.keras.models import load_model
model = load_model('my_model.hd5f')

def detect_points(face_img):
    me  = np.array(face_img) / 255
    x_test = np.expand_dims(me, axis = 0)
    x_test = np.expand_dims(x_test, axis = 3)

    y_test = model.predict(x_test)
    label_points = (np.squeeze(y_test) * 48) + 48 
    print(label_points)
    return label_points

# Load haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dimensions = (96, 96)

cap = cv2.VideoCapture(0)
ind = 0
while True:

    ret, frame = cap.read()
    default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    faces_img = np.copy(gray_img)
    plt.rcParams["axes.grid"] = False
    all_x_cords = []
    all_y_cords = []
    temp_xx = []
    temp_yy = []
    set_ = None
    for i, (x,y,w,h) in enumerate(faces):
        
        h += 10
        w += 10
        x -= 5
        y -= 5
        
        just_face = cv2.resize(gray_img[y:y+h,x:x+w], dimensions)
        cv2.rectangle(faces_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        scale_val_x = w / 96
        scale_val_y = h / 96
        
        label_point = detect_points(just_face)
        all_x_cords.append((label_point[::2] * scale_val_x) + x)
        all_y_cords.append((label_point[1::2] * scale_val_y) + y)

        for i in range(len(all_x_cords)):
            for j in range(len(all_x_cords[i])):
                set_ = (int(all_x_cords[i][j]), int(all_y_cords[i][j]))
                print(set_)
                default_img = cv2.circle(default_img, set_, 1, (0, 0, 255), 3)
    
    cv2.imwrite(f'{ind}.jpg', default_img)
    cv2.imshow('', default_img)
    ind += 1
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()