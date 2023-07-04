import cv2

# img = cv2.imread('./data/lena.png')  #importing image

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

def getObjects(img,thres,nms,draw=True,objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    # print(classIds,bbox)
    
    # confidence threshold is the minimum score that the model will consider the prediction to be a true prediction (otherwise it will ignore this prediction entirely).

    # flatten() method in Python is used to return a copy of a given array in such a way that it is collapsed into one dimension.
    
    if len(objects) == 0: objects = classNames
    
    objectInfo = []
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId-1]
            if className in objects:
                objectInfo.append([box,className])
                if(draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img,objectInfo
        
    
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)   #width
    cap.set(4,720)    #height
    # cap.set(10,150)  #for brightness
    while True:
        success,img = cap.read()
        result,objectInfo = getObjects(img,0.45,0.2,objects=['laptop'])
        cv2.imshow("Output",img) #displaying image
        cv2.waitKey(1) #waitkey() function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed.
    