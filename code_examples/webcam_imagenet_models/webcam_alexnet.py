import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import class_file


cap = cv2.VideoCapture(0) #first webcam in the system

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
net = torchvision.models.alexnet(pretrained=True, progress=True)                                 

while(True):
                                                                     # Capture frame-by-frame
    ret, frame = cap.read()  #returned in the frame, return part is for getting true or false  
    img = frame
    frame = cv2.resize(frame,(228,228))     #adjusting the size of frame for neural network
    frame_tp = np.transpose(frame, (2,0,1))  # changing the order of variables [batch-channel-height-width] 
    frame_torch = torch.from_numpy(frame_tp).float().cpu() #converting frame numpy to torch
    output = net(frame_torch.unsqueeze(0))     #to expand the dimensions of the frame 3 to 4
    a = torch.nn.functional.softmax(output[0], dim=0)
    values, indices = a.max(0)
    print(indices)
                          

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break


# When everything done, release the capture
cap.release()   #camera will be released, if we try to use a camera that's in use, we'll get an error
                                        #we can think this as a modifying a file while it is opened
cv2.destroyAllWindows()