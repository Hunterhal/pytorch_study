from learn_spec_img import *
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
directory = "./saved_models"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.pth")))
files.sort(key=lambda x: os.path.getmtime(x))
acc_list = []
epoch_list = []
for filename in files:
    net.load_state_dict(torch.load(filename))
    correct = 0
    total = 0
    print(filename)
    with torch.no_grad():
        for data in testloader:
            spec = np.transpose(data[0], (0, 3, 1, 2))
            spec = spec.to(device)
            label = data[1].to(device)
            output = net(spec[:, 0, :, :].unsqueeze(1))

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    acc_list.append(100*(correct/total))
    epoch_list.append(int(filename.split("_")[-1].split(".")[0]))
    
fig = plt.figure()
plt.plot(epoch_list, acc_list)
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
fig.savefig("TrainingHistory.jpg")
plt.show()
