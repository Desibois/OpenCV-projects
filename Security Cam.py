import cv2
import time
import datetime

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
record = False
start_time = 0
stop_after_time = 5
size = (int(cap.get(3)), int(cap.get(4)))
code = cv2.VideoWriter_fourcc(*"mp4v")
output = None
last_detection_time = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    body = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(body) > 0:
        last_detection_time = time.time()

        if not record:
            record = True
            now = datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")
            output = cv2.VideoWriter(f"{now}.mp4", code, 20, size)
            print("Started Recording!!")
            start_time = time.time()
    elif record and time.time() - last_detection_time > stop_after_time:
        record = False
        output.release()
        print("Stopped Recording")

    if record:
        output.write(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("j"):
        break

cap.release()
if output:
    output.release()

cv2.destroyAllWindows()
