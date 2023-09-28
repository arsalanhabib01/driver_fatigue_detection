# encoding=utf-8
import torch
import torch.nn as nn
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def Post_Training_Dynamic_Quantization(model):
    return torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    '''
    early_stopper = EarlyStopper(patience=3, min_delta=10)
for epoch in np.arange(n_epochs):
    train_loss = train_one_epoch(model, train_loader)
    validation_loss = validate_one_epoch(model, validation_loader)
    if early_stopper.early_stop(validation_loss):             
        break
    '''


class FatigueNet(nn.Module):
    def __init__(self):
        super(FatigueNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 197, out_features=512),
            nn.ReLU()
        )
        #self.bn = nn.BatchNorm1d(512)#nn.Dropout(p=0.3)#BatchNorm1d(512)

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=2)
        )
        #self.logSoftmax = LogSoftmax(dim=1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        #print(out.shape)
        #out = out.reshape(-1, out.shape[1] * out.shape[2])
        out = flatten(out, 1)
        out = self.fc1(out)
        #out = self.bn(out)
        #out = self.fc2(out)
        out = self.fc3(out)
        #outsoftmax = self.logSoftmax(out)
        return out

class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t
    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target
    def __len__(self):
        return len(self.samples)


def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.01
    return base_lr/(1+factor*epoch)


def train(model, optimizer, train_loader, valid_loader, test_loader, scheduler=None):
    criterion = nn.CrossEntropyLoss()
    epoch = 100
    result=[]
    for e in range(epoch):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        for sample, target in train_loader:
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE)
            #sample = sample.view(-1, 4, 4000)
            output = model(sample)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)
        if scheduler is not None:
            scheduler.step()
        if e > 1 and e % 2:
            pathmodel = './cnn1d_fatigue_epoch'+str(e)+'.pt'
            print('saving chkpt:',pathmodel)
            print('performance on test:', valid(model, test_loader))
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, pathmodel)

        # Testing
        acc_test = valid(model, valid_loader)
        print(f'Epoch: [{e}/{epoch}], loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, test_acc: {float(correct) * 100 / total:.2f}')
        result.append([acc_train, acc_test])
    return result
        #result.append([acc_train, acc_test])
        #result_np = np.array(result, dtype=float)
        #np.savetxt('result.csv', result_np, fmt='%.2f', delimiter=',')


def valid(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        for sample, target in test_loader:
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE)
            sample = sample.view(-1, 4, 800)
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
    acc_test = float(correct) * 100 / total
    return acc_test


def plot(data):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Test Accuracy', fontsize=16)
    plt.savefig('plot.png')

def eval_models(model, test_loader):
    chps=[]
    for a in range(3, 16, 2):
        chps.append("cnn1d_fatigue_epoch" + str(a) + ".pt")
    for c in chps:
        checkpoint = torch.load(c)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(valid(model, test_loader), 'for checkpint:', c)


if __name__ == '__main__':
    if 1:
        Xt, yt, gt = load_pickle(os.path.join(
            "/Volumes/Elements/dais/data/uta_rldd/processed/test_participant0.pkl"))
        print("[INFO] evaluating network...")
        xdnnt = np.reshape(Xt, (-1, 4000, 4))
        xdnnt = torch.from_numpy(xdnnt[:, ::5])
        xdnnt = torch.transpose(xdnnt, 2, 1)
        # X_train, X_test, y_train, y_test = train_test_split(xdnn, y, test_size=0.33, random_state=42)
        test_set = data_loader(xdnnt, yt, None)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        # trainSteps = len(train_loader.dataset) // BATCH_SIZE
        # valSteps = len(test_loader.dataset) // BATCH_SIZE
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        print('preprocessing done')
        model = FatigueNet().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        checkpoint = torch.load("./cnn1d_fatigue_epoch15.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print_model_size(model)

        model_dynamic_quantized = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )

        print_model_size(model_dynamic_quantized)

        backend = "qnnpack"
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        model_static_quantized = torch.quantization.prepare(model, inplace=False)
        model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

        print_model_size(model_static_quantized)


        # turn off autograd for testing evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model_dynamic_quantized.eval()
            # initialize a list to store our predictions
            preds = []
            # loop over the test set
            correct, total = 0, 0
            for (x, y) in test_loader:
                # send the input to the device
                x = x.to(DEVICE).float()
                y = y.to(DEVICE)
                # make the predictions and add them to the list
                pred = model_dynamic_quantized(x)
                _, predicted = torch.max(pred.data, 1)
                preds.extend(predicted.cpu().numpy())
                total += y.size(0)
                correct += (predicted == y).sum()
            acc_test = float(correct) * 100 / total
        from sklearn.metrics import classification_report

        print(classification_report(yt,
                                    np.array(preds), target_names=['awake', 'sleep']))
        quit()


    participant_test=0
    print('loading the training set')
    X, y, g = load_pickle(os.path.join("/Volumes/Elements/dais/data/uta_rldd/processed/train_participant"+str(participant_test)+".pkl"))
    print('loading the test set')
    Xt, yt, gt = load_pickle(os.path.join("/Volumes/Elements/dais/data/uta_rldd/processed/test_participant"+str(participant_test)+".pkl"))

    print('trainingset:',Xt.shape,' participants:', len(np.unique(g)))
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.model_selection import train_test_split
    import torch.optim.lr_scheduler as lr_scheduler

    INIT_LR = 1e-3
    BATCH_SIZE = 16
    EPOCHS = 40
    # define the train and val splits
    TRAIN_SPLIT = 0.9
    VAL_SPLIT = 1 - TRAIN_SPLIT
    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SPLIT, random_state=42)
    for i, (train_index, valid_index) in enumerate(gss.split(X, y, g)):
        # reshaping train and test
        xdnn = np.reshape(X, (-1, 4000, 4))
        # keeping each 10th value in the timeseries
        xdnn = torch.from_numpy(xdnn[:,::5])
        xdnn = torch.transpose(xdnn, 2, 1)
        xdnnt = np.reshape(Xt, (-1, 4000, 4))
        xdnnt = torch.from_numpy(xdnnt[:, ::5])
        xdnnt = torch.transpose(xdnnt, 2, 1)
        print('After preprocessing:', xdnn.shape)
        #X_train, X_test, y_train, y_test = train_test_split(xdnn, y, test_size=0.33, random_state=42)
        train_set = data_loader(xdnn[train_index], y[train_index], None)
        valid_set = data_loader(xdnn[valid_index], y[valid_index], None)
        test_set = data_loader(xdnnt, yt, None)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        #trainSteps = len(train_loader.dataset) // BATCH_SIZE
        #valSteps = len(test_loader.dataset) // BATCH_SIZE
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        print('preprocessing done')
        model = FatigueNet().to(DEVICE)

        import torch.optim as optim
        import time
        #optimizer = optim.SGD(params=model.parameters(
        #), lr=0.01, momentum=0.9)

        # initialize our optimizer and loss function
        #optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
        optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)#lr_scheduler.LambdaLR(optimizer, lr_lambda)
        #lossFn = nn.NLLLoss() # in case of the softmax func
        #scheduler = None
        # measure how long training is going to take
        print("[INFO] training the network...")
        startTime = time.time()

        result = train(model, optimizer, train_loader, valid_loader, test_loader, scheduler=scheduler)
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))
        #result = np.array(result, dtype=float)
        #np.savetxt('result.csv', result, fmt='%.2f', delimiter=',')
        #plot(result)
        print("[INFO] evaluating network...")
        # turn off autograd for testing evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # initialize a list to store our predictions
            preds = []
            # loop over the test set
            correct, total = 0, 0
            for (x, y) in test_loader:
                # send the input to the device
                x = x.to(DEVICE).float()
                y = y.to(DEVICE)
                # make the predictions and add them to the list
                pred = model(x)
                _, predicted = torch.max(pred.data, 1)
                preds.extend(predicted.cpu().numpy())
                total += y.size(0)
                correct += (predicted == y).sum()
            acc_test = float(correct) * 100 / total
            print('Same acc?',acc_test)
        from sklearn.metrics import classification_report
        print(classification_report(yt,
                                    np.array(preds), target_names=['awake', 'sleep']))