import numpy as np
import cv2
import pyautogui
from keras.models import load_model

model = load_model("my_model.h5")

game_map = {0: 'Run', 1: 'Jump'}
start = 0
cap = cv2.VideoCapture(0)

x = 10
y = 10
x2 = 200
y2 = 222

while True:
    _, im = cap.read()

    cv2.putText(im, "Press Esc to Close", (300, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 253, 0), 2, cv2.LINE_AA, False)
    cv2.putText(im, "Fist -> Run", (300, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 253, 0), 2, cv2.LINE_AA, False)
    cv2.putText(im, "Palm -> Jump", (300, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 253, 0), 2, cv2.LINE_AA, False)

    cv2.rectangle(im, (10, 10), (200, 222), (255, 0, 0), 2)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    hand_img = thresh[y:y2, x:x2]
    rerect_sized = cv2.resize(hand_img, (100, 89))

    img = np.expand_dims(rerect_sized, axis=0)
    img = np.reshape(img, (1, 89, 100, 1))

    result = model.predict(img)
    result1 = result[0][0]

    if result1 >= 0.6:
        label = "Jump"
        pyautogui.press('space')

    else:
        label = "Run"
    print(label)
    cv2.putText(im, label, (10, 222), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 230), 2, cv2.LINE_AA, False)
    cv2.imshow('img', im)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
