import cv2


cascade = cv2.CascadeClassifier("FDM/haarcascade_frontalface_default.xml")

# get the grey image
image = cv2.imread("FDM/abba.png", cv2.COLOR_RGB2GRAY)

faces = cascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)

print("found faces", format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)