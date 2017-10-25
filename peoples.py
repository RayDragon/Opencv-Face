import cv2
import numpy as np
import os
import time

face_cascade = cv2.CascadeClassifier("MM/haar_facedetect.xml")
names = os.listdir("MM/train/")
recognizer = cv2.face.createLBPHFaceRecognizer()

def face_get(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(face) == 0:
        return None, None
    (x, y, w, h) = face[0]
    return gray[y:y+w, x:x+h], face[0], (x, y, w, h)


def get_user_pics():
    cam = cv2.VideoCapture(0)
    i = 0
    frames = []
    while i<20:
        retval, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pic = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in pic:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            frames.append(frame)
            i += 1
            print(i)
            if i >= 20:
                break
    cam.release()
    return frames


def add_user_data(name):
    frames = get_user_pics()
    i = 0
    if not os.path.exists("MM/train/"+name):
        os.mkdir("MM/train/"+name)

    for frame in frames:
        img, rect, (x,y,w,h) = face_get(frame)
        i+=1
        cv2.imwrite("MM/train/"+name+"/"+str(i)+".jpeg", img)
    train()


def train():
    faces = []
    labels = []
    i=0
    for name in names:
        i += 1
        print("training from folder"+name)
        data = os.listdir("MM/train/"+name+"/")
        for imname in data:
            img = cv2.imread("MM/train/"+name+"/"+imname, cv2.COLOR_BGR2GRAY)
            if not img is None:
                labels.append(i)
                faces.append(img)
                # print(labels[:])
                # print(imname)
    # recognizer = cv2.face.createLBPHFaceRecognizer()
    recognizer.train(faces, np.array(labels))
    # recognizer.save("recog.xml")


def recog_camera():
    cam = cv2.VideoCapture(0)
    time.sleep(1)

    while True:
        try:
            rectval, pict = cam.read()

            try:
                rectval, pict = cam.read()
                face, rect, (x,y,w,h) = face_get(pict)
                cv2.rectangle(pict, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label, conf = recognizer.predict(face)
                cv2.putText(pict, names[label-1]+str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            except ValueError:
                print(ValueError)

            cv2.imshow("d", pict)
        except ValueError:
            print(ValueError)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break


def train_from_folder(fname, name):
    if not os.path.exists("MM/train/"+name):
        os.mkdir("MM/train/"+name)
    fnames = os.listdir(fname)
    i=1
    for fn in fnames:
        img = cv2.imread(fname+"/"+fn)
        frame, rect, (x, y, w, h) = face_get(img)
        if not frame is None:
            cv2.imwrite("MM/train/"+name+"/"+str(i)+".jpeg", frame)
            print(i, end=" ")
            i += 1
    train()


def recog_file(path):
    img = cv2.imread(path)
    face, rect, (x,y,w,h) = face_get(img)
    recogniser = cv2.face.createLBPHFaceRecognizer()
    recogniser.load("recog.xml")

    try:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label, conf = recogniser.predict(face)
        cv2.putText(img, names[label - 1]+str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    except ValueError:
        print(ValueError)
    cv2.imshow("d", img)
    cv2.waitKey(0)

q=2
train()
while q != 0:
    print("What would you like to do")
    print("1 : for existing user")
    print("2 : create new user")
    print("3 : let me guess who you are")
    print("4 : train from images folder")
    print("5 : predict existing image")

    q = int(input())

    if q == 2:
        print("Name:")
        name = str(input())
        add_user_data(name)

    elif q == 3:
        recog_camera()
    elif q == 4:
        print("Folder address:", end=" ")
        dest = str(input())
        print("Name:")
        name = str(input())
        train_from_folder(dest, name)
    elif q == 5:
        print("Image path:", end=" ")
        path = input()
        recog_file(path)

    cv2.destroyAllWindows()
