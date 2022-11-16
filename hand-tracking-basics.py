import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# define model
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
pTime=0
cTime=0


while True:
    succ, img = cap.read()
    # send RGP to the model
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # make detection
    results = hands.process(imgRGB)
    # extrect hands
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            # get landmarks and id numbers
            for id,lm in enumerate(handlms.landmark):

                h, w, c = img.shape
                cx, cy =int(lm.x*w), int(lm.y*h)
                print(id,cx, cy)
                if id==4:
                    cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHand.HAND_CONNECTIONS)
    # frame rate
    cTime= time.time()
    fps= 1/(cTime-pTime)
    pTime= cTime
    # put text=> image, text, position, font family, scale, color, thickness
    cv2.putText(img, str(int(fps)),(10,90), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)


    cv2.imshow('output ',img)
    cv2.waitKey(1)