
import cv2
from random import randrange 

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')
trained_eye_data = cv2.CascadeClassifier('haarcascade_eye.xml')


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
		cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 2)

		the_face = frame[y:y+h, x:x+w]
		face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

		smile_coordinates = trained_smile_data.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

		eye_coordinates = trained_eye_data.detectMultiScale(face_grayscale)

		if len(smile_coordinates) > 0:
			cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
		
		#iterate over every smile in the graysclae face 
		for (x_,y_,w_,h_) in smile_coordinates:
			cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 2)

		#iterate over every smile in the graysclae face 
		for (x__,y__,w__,h__) in eye_coordinates:
			cv2.rectangle(the_face, (x__,y__), (x__+w__, y__+h__), (255,255,255), 2)

	#show the current frame 		
	cv2.imshow("Smile Detector", frame)
	key = cv2.waitKey(1)

	#Pressing upper/lowercase Q quits the program
	if key == 81 or key == 113:
		break

# Release videocapture object
webcam.release()
webcam.destroyAllWindows()


print("Code completed!")