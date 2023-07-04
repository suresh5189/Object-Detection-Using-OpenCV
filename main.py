import cv2
import numpy as np

# img = cv2.imread('./data/lena.png')  #importing image

thres = 0.45 #threshold to detect object

nms_threshold = 0.2

# Non Maximum Suppression is a computer vision method that selects a single entity out of many overlapping entities (for example bounding boxes in object detection).

cap = cv2.VideoCapture(0)
cap.set(3,1280)   #width
cap.set(4,720)   #height
cap.set(10,150)  #for brightness
 

classNames = []
classFile = './data/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# We use the open() function to open a file in Python. It takes the filename as its first input argument and the python literals “r”, “w”, “r+”, etc as its second input argument to specify the mode in which the file is opened.

    
    
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPaths = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPaths,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    # print(classIds,bbox)
    
    # confidence threshold is the minimum score that the model will consider the prediction to be a true prediction (otherwise it will ignore this prediction entirely).
     
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    
    for i in indices:
        box = bbox[i]
        startX,startY,endX,endY = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(startX,startY),(endX+startX,endY+startY),color=(0,255,0),thickness=2)
        cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # flatten() method in Python is used to return a copy of a given array in such a way that it is collapsed into one dimension.
    
    # if len(classIds) != 0:
    #     for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    #         cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #         cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
    cv2.imshow("Output",img) #displaying image

    cv2.waitKey(1) #waitkey() function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed.