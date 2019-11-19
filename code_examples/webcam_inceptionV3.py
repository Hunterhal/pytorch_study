import numpy as np
import cv2
import torch
import torch.nn
import torch.nn.functional
import torchvision


cap = cv2.VideoCapture(0)  # first webcam in the system

# aux_logits = False, to disable inception_v3 model's auxiliary output since we have 1 input image at a time (batch size = 1)
net = torchvision.models.inception_v3(pretrained=True, progress=True, aux_logits=False)

while(True):
                             # capture frame-by-frame
    ret, frame = cap.read()  # returned in the frame, return part is for getting true or false  
    img = frame
    # important: In contrast to the other models the inception_v3 expects tensors with a size of 299 x 299
    frame = cv2.resize(frame, (299, 299))  # adjusting the size of frame for neural network
    frame_tp = np.transpose(frame, (2, 0, 1))  # changing the order of variables [batch-channel-height-width] 
    frame_torch = torch.from_numpy(frame_tp).float().cpu()  # converting frame numpy to torch
    output = net(frame_torch.unsqueeze(0))  # to expand the dimensions of the frame 3 to 4
    a = torch.nn.functional.softmax(output[0], dim=0)
    values, indices = a.max(0)
    print(indices)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break

# when everything done, release the capture
cap.release()  # camera will be released, if we try to use a camera that's in use, we'll get an error
               # we can think this as a modifying a file while it is opened
cv2.destroyAllWindows()
