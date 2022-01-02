
import cv2
from random import randrange 

#making a classifier
#using openCV's existing forntal face classifier algo on pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture('baby.mp4')


while True:

	## read the current frame from the webcame
	ret, frame = video.read()
	#convert the frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	face_coordinates = trained_face_data.detectMultiScale(gray)
	for (x,y,w,h) in face_coordinates:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(128,256)), 5)

	cv2.imshow("Hifza\'s Video Face Detector", frame)
	key = cv2.waitKey(1)

	#Pressing upper/lowercase Q quits the program
	if key == 81 or key ==113:
		break

# Release videocapture object
video.release()
video.destroyAllWindows()


print("Code completed!")