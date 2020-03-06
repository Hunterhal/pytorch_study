from learn_spec_img import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.load_state_dict(torch.load("./saved_models/net_epoch_16.pth"))
img = cv2.imread('0_mehmet_0.png')
print(img.shape)
img = np.transpose(img, (2, 0, 1))
print(img.shape)
img = torch.from_numpy(img)
pred = net(img[0, :, :].unsqueeze(0).unsqueeze(0).to(device))
print(pred.size())
_, predicted = torch.max(pred.data, 1)

print(predicted)
