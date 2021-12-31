

import cv2
from random import randrange 

#making a classifier
#using openCV's existing forntal face classifier algo on pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#pass the image of the video stream or picture into this aforementioned classifier
img = cv2.imread('RDJ.jpg')
img2 = cv2.imread('friends.jpeg')


#make the img greyScale 
#grey_img = cv2.imread('RDJ.jpg',cv2.IMREAD_GRAYSCALE)

#alternative way to do the above is to use the bottom command
#in openCV, the RGB color scheme is backwards to be BGR instead hence the comamnd

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
friends_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Detect Faces 
#Findings objects and returning coordinates of a rectangular box to be drawn around the image
#the returned coordinates represented a dot of the top half left corner and rect dimensions
#returns 4 coordinates which is the top left corner and the width(x) and height(x)
face_coordinates = trained_face_data.detectMultiScale(gray)
#print(face_coordinates)

#now we draw the rectangles on those coordinates
#we are taking the two tuples representing the top left and bottom right corner

#for (x,y,w,h) in face_coordinates:

#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
#2 represents the thickness of the drawn rectangle, green is the rect color
#cv2.rectangle(img, (87,114),(87+361,114+361), (0,255,0), 2)

#img2 is being changed with the rectangles drawn on them 
multi_face_coordinates = trained_face_data.detectMultiScale(friends_gray)
#looping through all faces and then do the pictures
for (x,y,w,h) in multi_face_coordinates:
	#cv2.rectangle(img2, (x,y), (x+w, y+h), (0,255,0), 5)
	#alternatively, randomly select the color of the drawn rectangle 
	cv2.rectangle(img2, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(128,256)), 5)
cv2.imshow("Hifza\'s Multi Face Detector", img2)
cv2.waitKey()

print("Code completed!")