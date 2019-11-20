
from c_neural_network_file import*                       #imports nn
net.load_state_dict(torch.load('mnist_net.pth'))       #loads our saved trained model

while(True):
    ret, frame = cap.read()                            #reads image
    frame2 = cv2.resize(frame,(28,28))                 #adjusts the size of frame for nn
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)    #rgb to grayscale
    frame_torch = torch.from_numpy(gray).float().cpu() #converts frame numpy to torch
    frame_torch=frame_torch.unsqueeze(0)               #expands the dimensions of the frame 2 to 3
    output = net(frame_torch.unsqueeze(0))             #expands the dimensions of the frame 3 to 4
    a = torch.nn.functional.softmax(output[0], dim=0)  
    values, indices = a.max(0)
    print(indices)                                     #prints output, nn inference
    
                           

    # Displays the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   #to get out of the infinite loop 
        break
# When everything done, release the capture
cap.release()       
cv2.destroyAllWindows()
