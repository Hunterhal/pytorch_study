from learn_spec_img import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#directory = "/home/mehmet/Desktop/bitirme/codes/fsdd/CNN/try2guess_test_files"
directory = "/home/mehmet/Desktop/bitirme/codes/fsdd/CNN/tinypics"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.png")))
i = 0
for filename in files:
    net.load_state_dict(torch.load("./saved_models/net_epoch_205.pth"))
    img = cv2.imread(filename)
    #print(img.shape)
    img = np.transpose(img, (2, 0, 1))
    #print(img.shape)
    img = torch.from_numpy(img)
    pred = net(img[0, :, :].unsqueeze(0).unsqueeze(0).to(device))
    #print(pred.size())
    _, predicted = torch.max(pred.data, 1)
    if int(filename.split("/")[-1].split("_")[0]) == int(predicted.item()):
        i += 1
    print('Ground Truth =', filename.split("/")[-1].split("_")[0], '      Prediction = ', predicted.item())
print(i,'/',len(files))
