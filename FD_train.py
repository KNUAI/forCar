import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from data_process.data_loader import read_data, read_all_data
from models.model import CRAE2

import argparse

parser = argparse.ArgumentParser(description='AWFD')

parser.add_argument('--seed', type=int, default=117, help='seed')
parser.add_argument('--data', type=str, default='D1.csv', help='data:: D1, D2')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--fold', type=int, default=5, help='5-fold:: 1, 2, 3, 4, 5')
parser.add_argument('--latent_size', type=int, default=128, help='dimension of latent vector')
parser.add_argument('--threshold_rate', type=float, default=5, help='threshold_rate')
parser.add_argument('--n_layer', type=int, default=1, help='n_layers of rnn model')
parser.add_argument('--epoch', type=int, default=200, help='epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--r_model', type=str, default='LSTM', help='rnn model')
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--patience', type=int, default=3, help='patience of early_stopping')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')

args = parser.parse_args()

#seed
if args.seed is not None:
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Let's use", torch.cuda.device_count(), "GPUs!")
print('device:', device)

data_path = f'./datasets/Anomaly_Detection/{args.data}'
#data_split to train/valid/test
train_loader, valid_loader, test_loader, input_size, max_len, scaler = read_data(data_path, args.batch_size, args.fold, args.cols)

#model
model = CRAE2(input_size, args.latent_size, max_len, args.n_layer, args.r_model)
model.to(device)

#loss, optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#train
total_loss = 0
train_loss = []
i = 1
stop_loss = np.inf
count = 0
for epoch in range(args.epoch):
    #train
    model.train()
    for data, target in train_loader:
        data = data.float().to(device)
        target = target.float().to(device)

        output = model(data)
        loss = criterion(output, target)

        total_loss += loss
        train_loss.append(total_loss/i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch: {epoch+1}\t Train Step: {i:3d}\t Loss: {loss:.4f}')
        i += 1
    print(f'Epoch: {epoch+1} finished')

    #validation
    with torch.no_grad():
        model.eval()
        valid_loss = []
        for data, target in valid_loader:
            data = data.float().to(device)
            target = target.float().to(device)
    
            output = model(data)
            loss = criterion(output, target)
            valid_loss.append(loss.detach().cpu().numpy())
    
    #early_stopping
    if not os.path.exists('./path/Anomaly_Detection'):
        os.makedirs('./path/Anomaly_Detection')
    if np.mean(valid_loss) < stop_loss:
        stop_loss = np.mean(valid_loss)
        print('best_loss:: {:.4f}'.format(stop_loss))
        torch.save(model.state_dict(), f'./path/Anomaly_Detection/{args.data}_fold_{args.fold}_latent_{args.latent_size}_th_rate_{args.threshold_rate}_batch_{args.batch_size}_lr_{args.lr}.pth')
        count = 0
    else:
        count += 1
        print(f'EarlyStopping counter: {count} out of {args.patience}')
        print(f'best_loss:: {stop_loss:.4f}\t valid_loss:: {np.mean(valid_loss):.4f}' )
        if count >= args.patience:
            print('Ealry stopping')
            break

model.load_state_dict(torch.load(f'./path/Anomaly_Detection/{args.data}_fold_{args.fold}_latent_{args.latent_size}_th_rate_{args.threshold_rate}_batch_{args.batch_size}_lr_{args.lr}.pth'))


