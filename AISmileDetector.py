import cv2
import time
from random import randrange

def saveFrame(frame, count):   
    cv2.imshow("cam-test",frame)
    cv2.waitKey(1000)
    cv2.destroyWindow("cam-test")
    # If multiple saves are needed...
    # imgName = "smile" + str(count) + ".jpg"
    imgName = "smile.jpg"
    cv2.imwrite(str(imgName),frame)

trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trainedSmileData = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)                                                    #videos work too
count = 0
smileState = False
print("Smile To Click a Picture!\nPress 'Q' to Quit")
while True:
    successfulFrameRead, frame = webcam.read()
    if not successfulFrameRead:
        break
    if smileState:
        saveFrame(frame, count)
        smileState = False
    successfulFrameRead, frame = webcam.read()
    if not successfulFrameRead:
        break    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCoordinates = trainedFaceData.detectMultiScale(grayFrame,6,7)    
    for (x,y,w,h) in faceCoordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 4)                 #(src, coordinate, colour, thickness)        
        theFace = frame[y:y+h, x:x+w]       
        grayFace = cv2.cvtColor(theFace, cv2.COLOR_BGR2GRAY)
        smileCoordinates = trainedSmileData.detectMultiScale(grayFace, 1.7, 20)
        for (x2, y2, w2, h2) in smileCoordinates:
            cv2.rectangle(theFace, (x2,y2), (x2+w2, y2+h2), (255, 0, 255), 4)   #(src, coordinate, colour, thickness)
        
        if len(smileCoordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), 2, cv2.FONT_HERSHEY_PLAIN, (255, 0, 0))
            time.sleep(0.3)
            if count < 10:
                count += 1 
                smileState = True
        
    cv2.imshow('Face Detector', frame)
    if cv2.waitKey(1) == 113 or cv2.waitKey(1) == 81:
        break

webcam.release()
cv2.destroyAllWindows()

print('Code Completed')
