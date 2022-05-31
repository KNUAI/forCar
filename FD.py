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

if args.evaluate:
    model.load_state_dict(torch.load(f'./path/Anomaly_Detection/{args.data}_fold_{args.fold}_latent_{args.latent_size}_th_rate_{args.threshold_rate}_batch_{args.batch_size}_lr_{args.lr}.pth'))
    
    #setting_threshold
    loss_list = []
    with torch.no_grad():
        model.eval()
        for data, target in valid_loader:
            data = data.float().to(device)
            target = target.float().to(device)
    
            output = model(data)
            loss = criterion(output, target)
            loss_list.append(loss)
    
    threshold = max(loss_list) * args.threshold_rate
    print('threshold:: ', threshold)
    
    #test
    test_acc_sum = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for i, (data, target, label) in enumerate(test_loader):
            data = data.float().to(device)
            target = target.float().to(device)
    
            output = model(data)
            loss = criterion(output, target)
    
            #print(f'\nTest set: Loss: {loss:.4f}')
    
            if loss > threshold:
                logits = 1
                #print(i, 'th wafer is defective.')
            else:
                logits = 0
                #print(i, 'th wafer is OK')
                if label == 1:
                    print(i, 'th wafer is originally defective!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('loss:: ', loss)
            
            y_true.append(label)
            y_pred.append(logits)
            
            test_acc_sum += np.equal(label, logits).sum()            
        print(f'  Test Accuracy: {100 * test_acc_sum / len(test_loader.dataset):.4f}')
        tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
        print('confusion_matrix:: ', tn, fp, tp, fn)

    #all test
    all_loader = read_all_data(data_path, max_len, scaler, args.cols)
    all_acc_sum = 0
    a_true = []
    a_pred = []
    tp_loss = []
    tn_loss = []
    fp_loss = []
    fn_loss = []
    with torch.no_grad():
        model.eval()
        for i, (data, target, label) in enumerate(all_loader):
            data = data.float().to(device)
            target = target.float().to(device)
    
            output = model(data)
            loss = criterion(output, target)
    
            #print(f'\nTest set: Loss: {loss:.4f}')
    
            if loss > threshold:
                logits = 1
                #print(i, 'th wafer is defective.')
                if label == 1:
                    tn_loss.append(loss.detach().cpu().numpy())
                else: fn_loss.append(loss.detach().cpu().numpy())
            else:
                logits = 0
                #print(i, 'th wafer is OK')
                if label == 1:
                    fp_loss.append(loss.detach().cpu().numpy())
                    print(i, 'th wafer is originally defective!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('loss:: ', loss)
                else:
                    tp_loss.append(loss.detach().cpu().numpy())

            a_true.append(label)
            a_pred.append(logits)
            
            all_acc_sum += np.equal(label, logits).sum()
        print(f'  all Accuracy: {100 * all_acc_sum / len(all_loader.dataset):.4f}')
        tp, fn, fp, tn = confusion_matrix(a_true, a_pred).ravel()
        print('confusion_matrix:: ', tn, fp, tp, fn)

    #figure
    if not os.path.exists('./picture/Anomaly_Detection/'):
        os.makedirs('./picture/Anomaly_Detection/')
    fig, ax = plot_confusion_matrix(np.array([[tp, fn],
                                              [fp, tn]]))

    plt.savefig(f'./picture/Anomaly_Detection/confusion_matrix_{args.data}_fold_{args.fold}_latent_{args.latent_size}_th_rate_{args.threshold_rate}_batch_{args.batch_size}_lr_{args.lr}.png')
    plt.close()

    tp_loss = np.array(tp_loss)
    tn_loss = np.array(tn_loss)
    fp_loss = np.array(fp_loss)
    fn_loss = np.array(fn_loss)

    plt.hist(tp_loss, bins = 10000, color='b', histtype='step', label = 'true_positive')
    plt.hist(tn_loss, bins = 10000, color='y', histtype='step', label = 'true_negative')
    plt.hist(fp_loss, bins = 10000, color='r', histtype='step', label = 'false_positive')
    plt.hist(fn_loss, bins = 10000, color='g', histtype='step', label = 'false_negative')
    plt.legend()

    plt.savefig(f'./picture/Anomaly_Detection/hist_{args.data}_fold_{args.fold}_latent_{args.latent_size}_th_rate_{args.threshold_rate}_batch_{args.batch_size}_lr_{args.lr}.png')
    plt.close()

else:
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
    
    #setting_threshold
    loss_list = []
    with torch.no_grad():
        model.eval()
        for data, target in valid_loader:
            data = data.float().to(device)
            target = target.float().to(device)
    
            output = model(data)
            loss = criterion(output, target)
            loss_list.append(loss)
    
    threshold = max(loss_list) * args.threshold_rate
    print('threshold:: ', threshold)
    
    #test
    test_acc_sum = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for i, (data, target, label) in enumerate(test_loader):
            data = data.float().to(device)
            target = target.float().to(device)
    
            output = model(data)
            loss = criterion(output, target)
    
            #print(f'\nTest set: Loss: {loss:.4f}')
    
            if loss > threshold:
                logits = 1
                #print(i, 'th wafer is defective.')
            else:
                logits = 0
                #print(i, 'th wafer is OK')
                if label == 1:
                    print(i, 'th wafer is originally defective!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('loss:: ', loss)
            
            y_true.append(label)
            y_pred.append(logits)
            
            test_acc_sum += np.equal(label, logits).sum()
        print(f'  Test Accuracy: {100 * test_acc_sum / len(test_loader.dataset):.4f}')
        tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
        print('confusion_matrix:: ', tn, fp, tp, fn)

    #all test
    all_loader = read_all_data(data_path, max_len, scaler, args.cols)
    all_acc_sum = 0
    a_true = []
    a_pred = []
    tp_loss = []
    tn_loss = []
    fp_loss = []
    fn_loss = []
    with torch.no_grad():
        model.eval()
        for i, (data, target, label) in enumerate(all_loader):
            data = data.float().to(device)
            target = target.float().to(device)
    
            output = model(data)
            loss = criterion(output, target)
    
            #print(f'\nTest set: Loss: {loss:.4f}')
    
            if loss > threshold:
                logits = 1
                #print(i, 'th wafer is defective.')
                if label == 1:
                    tn_loss.append(loss.detach().cpu().numpy())
                else: fn_loss.append(loss.detach().cpu().numpy())
            else:
                logits = 0
                #print(i, 'th wafer is OK')
                if label == 1:
                    fp_loss.append(loss.detach().cpu().numpy())
                    print(i, 'th wafer is originally defective!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('loss:: ', loss)
                else:
                    tp_loss.append(loss.detach().cpu().numpy())

            a_true.append(label)
            a_pred.append(logits)
            
            all_acc_sum += np.equal(label, logits).sum()
        print(f'  all Accuracy: {100 * all_acc_sum / len(all_loader.dataset):.4f}')
        tp, fn, fp, tn = confusion_matrix(a_true, a_pred).ravel()
        print('confusion_matrix:: ', tn, fp, tp, fn)

    #figure
    if not os.path.exists('./picture/Anomaly_Detection/'):
        os.makedirs('./picture/Anomaly_Detection/')
    fig, ax = plot_confusion_matrix(np.array([[tp, fn],
                                              [fp, tn]]))

    plt.savefig(f'./picture/Anomaly_Detection/confusion_matrix_{args.data}_fold_{args.fold}_latent_{args.latent_size}_th_rate_{args.threshold_rate}_batch_{args.batch_size}_lr_{args.lr}.png')
    plt.close()

    tp_loss = np.array(tp_loss)
    tn_loss = np.array(tn_loss)
    fp_loss = np.array(fp_loss)
    fn_loss = np.array(fn_loss)

    plt.hist(tp_loss, bins = 10000, color='b', histtype='step', label = 'true_positive')
    plt.hist(tn_loss, bins = 10000, color='y', histtype='step', label = 'true_negative')
    plt.hist(fp_loss, bins = 10000, color='r', histtype='step', label = 'false_positive')
    plt.hist(fn_loss, bins = 10000, color='g', histtype='step', label = 'false_negative')
    plt.legend()

    plt.savefig(f'./picture/Anomaly_Detection/hist_{args.data}_fold_{args.fold}_latent_{args.latent_size}_th_rate_{args.threshold_rate}_batch_{args.batch_size}_lr_{args.lr}.png')
    plt.close()




