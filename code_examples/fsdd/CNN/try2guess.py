from learn_spec_img import *
import numpy as np


directory = "/home/mehmet/Desktop/CNN/tinypics"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.png")))
i = 0
for filename in files:
    net.load_state_dict(torch.load("./33x33_saved_models/net_epoch_199.pth"))
    img = cv2.imread(filename, 0)  # 0 is put because gray scale images will be loaded
    img = torch.from_numpy(img)  # transform numpy to tensor
    pred = net(img.unsqueeze(0).unsqueeze(0).to(device))  # unsqueeze is used to to make our input as (1 x 1 x imgHeight x imgWidth)
    _, predicted = torch.max(pred.data, 1)
    if int(filename.split("/")[-1].split("_")[0]) == int(predicted.item()):
        i += 1
    print('Ground Truth =', filename.split("/")[-1].split("_")[0], '      Prediction = ', predicted.item())
print(i,'/',len(files))
