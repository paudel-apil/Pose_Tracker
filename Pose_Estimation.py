import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS,landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255)),connection_drawing_spec = mpDraw.DrawingSpec(color = (0,255,0)))
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id,lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
    cTime = time.time()
    if cTime != pTime:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0  # or set to some default value
    pTime = cTime

    cv2.putText(img, str(int(fps)), (40, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
