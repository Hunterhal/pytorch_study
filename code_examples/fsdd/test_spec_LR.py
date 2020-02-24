from learn_spec_LR import *

net.load_state_dict(torch.load(netname))

correct = 0
total = 0
net = net.float()
with torch.no_grad():
    for data in testloader:
        spec = data[0][0].squeeze(0).squeeze(0).to(device)
        
        label = data[1].to(device)
        h1 = torch.zeros(1, L1, requires_grad = True, device = device)
        h2 = torch.zeros(1, L2, requires_grad = True, device = device)

        c1 = torch.zeros(1, L1, requires_grad = True, device = device)
        c2 = torch.zeros(1, L2, requires_grad = True, device = device)
        
        for i in range(spec.size()[1]):
            h1 = h1.float()
            h2 = h2.float()
            c1 = c1.float()
            c2 = c2.float()
            state = [h1, c1 ,h2, c2]
            output, last_state = net(spec[:, i].unsqueeze(0).float(), state)

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
print("Accuracy over %d wav files is %d %%" %(len(testdataset), 100*(correct/total)))
