import os
import argparse
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data_process.etth_data_loader import Dataset_Custom, Dataset_Pred
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from models.SCINet import SCINet
from models.LSTM import LSTM
from models.Informer import Informer

parser = argparse.ArgumentParser(description='SCINet on ETT dataset')

parser.add_argument('--model', type=str, default='SCINet',help='model of experiment, options: [SCINet, Informer, LSTM]')
### -------  dataset settings --------------
parser.add_argument('--root_path', type=str, default='./datasets/forecasting/', help='root path of the data file')
parser.add_argument('--data', type=str, required=True, default='ETTh1.csv', help='location of the data file')
parser.add_argument('--checkpoints', type=str, default='./path/forecasting/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =True, help='denorm the output data')

### -------  device settings --------------
parser.add_argument('--gpu', type=str, default='0', help='gpu')
                                                                                  
### -------  input/output length settings --------------                                                                            
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of SCINet encoder, look back window')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length, horizon')
parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0)
                                                              
### -------  training settings --------------  
parser.add_argument('--cols', type=str, required=True, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae',help='loss function')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--save', type=bool, default =False, help='save the output results')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', action='store_true', default=False)

### -------  model settings --------------  
parser.add_argument('--hidden_size', default=4, type=float, help='hidden channel of module')
parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--positionalEcoding', type=bool, default=False)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--num_decoder_layer', type=int, default=1)
parser.add_argument('--RIN', type=bool, default=False)

args = parser.parse_args()

#seed
torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

#GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Let's use", torch.cuda.device_count(), "GPUs!")
print('device:', device)

if args.model == 'LSTM':
    args.seq_len = args.pred_len

print('Args in experiment:')
print(args)

class Exp():
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.model = self._build_model().to(device)

    def _build_model(self):

        in_dim = len(self.args.cols)

        if self.args.model == 'SCINet':
            model = SCINet(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim= in_dim,
                hid_size = self.args.hidden_size,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                groups = self.args.groups,
                kernel = self.args.kernel,
                dropout = self.args.dropout,
                single_step_output_One = self.args.single_step_output_One,
                positionalE = self.args.positionalEcoding,
                modified = True,
                RIN=self.args.RIN)
        elif self.args.model == 'Informer':
            model = Informer(
                enc_in=in_dim,
                dec_in=in_dim,
                c_out=in_dim,
                seq_len=self.args.seq_len,
                label_len=0,
                out_len=self.args.pred_len)
        elif self.args.model == 'LSTM':
            model = LSTM(input_dim= in_dim,
                num_layers=self.args.num_decoder_layer)
        else:
            print('wrong model!')
            exit()
        print(model)
        return model

    def _get_data(self, flag):
        args = self.args

        Data = Dataset_Custom

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            inverse=args.inverse,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, (batch_x, batch_y) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, batch_x, batch_y)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

            total_loss.append(loss)

        total_loss = np.average(total_loss)

        preds = np.array(preds)
        trues = np.array(trues)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y)

                loss = criterion(pred, true)

                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                                        
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)

        save_model(epoch, lr, self.model, path, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x,batch_y) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, batch_x, batch_y)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())


        preds = np.array(preds)
        trues = np.array(trues)

        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('TTTT denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

        if not os.path.exists('./picture/forecasting/'):
            os.makedirs('./picture/forecasting/')

        variable = len(self.args.cols)

        for place in range(1):
            for i in range(variable):
                plt.plot(range(len(pred_scales[0])), pred_scales[self.args.pred_len*place,:,i], color = 'r', label = 'preds')
                plt.plot(range(len(true_scales[0])), true_scales[self.args.pred_len*place,:,i], color = 'b', label = 'trues')
                plt.xlabel('time')
                plt.ylabel('value')
                plt.title('forecasting')
                plt.legend()
                #plt.show()
                plt.savefig(f'./picture/forecasting/{args.data}_test_place_{place}_{self.args.cols[i]}_{mse:.4f}.png')
                plt.close()
        return mae, maes, mse, mses

    def predict(self, setting, evaluate=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        pred_scales = []
        
        for i, (batch_x,batch_y) in enumerate(pred_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                pred_data, batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())

        preds = np.array(preds)
        pred_scales = np.array(pred_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        if not os.path.exists('./picture/forecasting/'):
            os.makedirs('./picture/forecasting/')

        variable = len(self.args.cols)


        for i in range(variable):
            plt.plot(range(len(pred_scales[0])), pred_scales[-1,:,i], color = 'r', label = 'preds')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.title('forecasting')
            plt.legend()
            #plt.show()
            plt.savefig(f'./picture/forecasting/{args.data}_predict_{self.args.cols[i]}.png')
            plt.close()

        return

    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()

        if self.args.model == 'Informer':
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(device)
            outputs = self.model(batch_x, dec_inp)
        else:
            outputs = self.model(batch_x)

        #if self.args.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)

        batch_y = batch_y[:,-self.args.pred_len:,:].to(device)
        batch_y_scaled = dataset_object.inverse_transform(batch_y)

        return outputs, outputs_scaled, 0,0, batch_y, batch_y_scaled

mae_ = []
maes_ = []
mse_ = []
mses_ = []

if args.evaluate:
    setting = 'sl{}_pl{}_lr{}_bs{}_hid{}_l{}_dp{}_inv{}'.format(args.seq_len, args.pred_len,args.lr,args.batch_size,args.hidden_size, args.levels,args.dropout,args.inverse)
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting, evaluate=True)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
    exp.predict(setting, evaluate=True)

else:
    setting = 'sl{}_pl{}_lr{}_bs{}_hid{}_l{}_dp{}_inv{}'.format(args.seq_len, args.pred_len,args.lr,args.batch_size,args.hidden_size, args.levels,args.dropout,args.inverse)
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
    exp.predict(setting)



