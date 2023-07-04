from ObjectDetectorModule import *


cap = cv2.VideoCapture(0)
cap.set(3,1280)   #width
cap.set(4,720)    #height
# cap.set(10,150)  #for brightness
while True:
    success,img = cap.read()
    result,objectInfo = getObjects(img,0.45,0.2)
    cv2.imshow("Output",img) #displaying image
    cv2.waitKey(1) #waitkey() function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed.