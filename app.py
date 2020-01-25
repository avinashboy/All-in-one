import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
count = 0

framed = False
eye = False
smile = False
blurs = False
capture = False

while True:
    count +=1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=4)
    for (x,y,w,h) in faces:

        if framed:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        if capture:
            cv2.imwrite("face/{}.jpg".format(count),frame[y:y+h,x:x+w])    
        if blurs:
            blur_area =  frame[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(blur_area, (101,101), 0)
            frame[y:y+h, x:x+w] = blur

        ri_gray = gray[y:y+h, x:x+w]
        ri_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(ri_gray,scaleFactor=1.1, minNeighbors=5)
        for(ex,ey,ew,eh) in eyes:
            if eye:
                cv2.rectangle(ri_color, (ex,ey), (ex+ew, ey+eh),(0,0,255), 2)

        rh_gray = gray[y:y+h, x:x+w]
        rh_color = frame[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(ri_gray,scaleFactor=1.8, minNeighbors=20)
        for(sx,sy,sw,sh) in smiles:
            if smile:
                cv2.rectangle(ri_color, (sx,sy), (sx+sw, sy+sh),(255,255,255), 2)

    cv2.imshow("video", frame)

    ch = cv2.waitKey(1) & 0xFF
    if ch == ord("f"):
        framed = not framed
    if ch == ord("e"):
        eye = not eye
    if ch == ord("s"):
        smile = not smile
    if ch == ord("b"):
        blurs = not blurs
    if ch == ord("c"):
        capture = not capture
    if ch == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
