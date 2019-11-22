import numpy as np
import cv2
import torch
import torch.nn
import torch.nn.functional
import torchvision
from torchvision import transforms
import class_file
from timeit import default_timer as timer


cap = cv2.VideoCapture(0)  # first webcam in the system

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# aux_logits = False, to disable inception_v3 model's auxiliary output since we have 1 input image at a time (batch size = 1)
net = torchvision.models.inception_v3(pretrained=True, progress=True, aux_logits=False).to(device)

start = timer()  # start timer

while(True):
                             # capture frame-by-frame
    ret, frame = cap.read()  # returned in the frame, return part is for getting true or false 
    img = frame
    # important: In contrast to the other models the inception_v3 expects tensors with a size of 299 x 299
    frame = cv2.resize(frame, (299, 299))  # adjusting the size of frame for neural network
    frame_tp = np.transpose(frame, (2, 0, 1))  # changing the order of variables [batch-channel-height-width] 
    frame_torch = torch.from_numpy(frame_tp).float().cpu()  # converting frame numpy to torch
    frame_norm = normalize(frame_torch)  # normalize frame
    frame_norm = frame_norm.to(device)  # send to GPU
    output = net(frame_norm.unsqueeze(0))  # to expand the dimensions of the frame 3 to 4
    a = torch.nn.functional.softmax(output[0], dim=0)
    values, indices = a.max(0)
    print(class_file.class_dict[indices.item()])

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break

end = timer()  # end timer
elapsed_time = format((end - start)/60, '.3f')  # calculate elapsed time
print('Elapsed time: ', elapsed_time, ' mins')

# when everything done, release the capture
cap.release()  # camera will be released, if we try to use a camera that's in use, we'll get an error
               # we can think this as a modifying a file while it is opened
cv2.destroyAllWindows()
