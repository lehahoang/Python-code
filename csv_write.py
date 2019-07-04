'''
  Writing the acc to a csv file
'''

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()# sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    
    with open('./temp/'+args.saved_file+ args.fault_rate+'.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow([acc])

Plz copy and paste thie code section right after the parser code.

    if not os.path.isfile('./temp/'+args.saved_file+ args.fault_rate+'.csv'):
        with open('./temp/'+args.saved_file+ args.fault_rate+'.csv', 'wb') as csvfile: #args.saved_file="1/data"
            filewriter = csv.writer(csvfile, delimiter=',')
