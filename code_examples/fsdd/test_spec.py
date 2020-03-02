from learn_spec import *


directory = "./saved_models"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.pth")))
files.sort(key=lambda x: os.path.getmtime(x))
acc_list = []
epoch_list = []
for filename in files:
    net.load_state_dict(torch.load(filename))
    print(filename)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            spec = data[0][0].squeeze(0).squeeze(0).to(device)
            label = data[1].to(device)
            seq_length = data[3].to(device)
            h1 = torch.zeros(1, L1, requires_grad=True, device=device)
            h2 = torch.zeros(1, L2, requires_grad=True, device=device)

            c1 = torch.zeros(1, L1, requires_grad=True, device=device)
            c2 = torch.zeros(1, L2, requires_grad=True, device=device)
            for i in range(seq_length):
                state = [h1, c1 ,h2, c2]
                output, last_state = net(spec[:, i].unsqueeze(0), state)

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