import cv2

camera = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("FDM/haarcascade_frontalface_default.xml")

while True:
    retval, pict = camera.read()
    faces = cascade.detectMultiScale(pict,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(pict, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", pict)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
