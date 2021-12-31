
import cv2
from random import randrange 

#making a classifier
#using openCV's existing forntal face classifier algo on pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#pass the image of the video stream or picture into this aforementioned classifier
#img = cv2.imread('RDJ.jpg')
#img2 = cv2.imread('friends.jpeg')

#webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

webcam = cv2.VideoCapture(0)

#iterate over frames from the webcam
if not webcam.isOpened():
    raise IOError("Cannot open webcam")

while True:

	## read the current frame from the webcame
	successful_frame_read_bool, frame = webcam.read()
	#convert the frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	face_coordinates = trained_face_data.detectMultiScale(gray)
	for (x,y,w,h) in face_coordinates:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(128,256)), 5)

	cv2.imshow("Hifza\'s Webcam Face Detector", frame)
	key = cv2.waitKey(1)

	#Pressing upper/lowercase Q quits the program
	if key == 81 or key ==113:
		break

# Release videocapture object
webcam.release()
webcam.destroyAllWindows()


print("Code completed!")