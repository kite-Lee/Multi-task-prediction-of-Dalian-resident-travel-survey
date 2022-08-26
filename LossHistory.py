from keras import callbacks
import pandas as pd

class LossHistory(callbacks.Callback):
    def get_csv_name(self,csvname):
        self.csvname = csvname

    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.mode_loss = {'batch': [], 'epoch': []}
        self.val_mode_loss = {'batch': [], 'epoch': []}
        self.purpose_loss = {'batch': [], 'epoch': []}
        self.val_purpose_loss = {'batch': [], 'epoch': []}
        self.mode_acc = {'batch': [], 'epoch': []}
        self.val_mode_acc = {'batch': [], 'epoch': []}
        self.purpose_acc = {'batch': [], 'epoch': []}
        self.val_purpose_acc = {'batch': [], 'epoch': []}
        self.mode_ce = {'batch': [], 'epoch': []}
        self.val_mode_ce = {'batch': [], 'epoch': []}
        self.purpose_ce = {'batch': [], 'epoch': []}
        self.val_purpose_ce = {'batch': [], 'epoch': []}
        self.iter = {'batch': [], 'epoch': []}
        self.mode_coef = {'batch': [], 'epoch': []}

        self.H = {}

    def on_epoch_end(self, epoch, logs={}):

        # 每一个epoch完成后向容器里面追加loss，acc



        self.val_mode_acc['epoch'].append(logs.get('val_mode_acc'))
        self.val_purpose_acc['epoch'].append(logs.get('val_purpose_acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_mode_loss['epoch'].append(logs.get('val_mode_loss'))
        self.val_purpose_loss['epoch'].append(logs.get('val_purpose_loss'))
        self.val_mode_ce['epoch'].append(logs.get('val_mode_ce'))
        self.val_purpose_ce['epoch'].append(logs.get('val_purpose_ce'))

        self.mode_acc['epoch'].append(logs.get('mode_acc'))
        self.purpose_acc['epoch'].append(logs.get('purpose_acc'))
        self.loss['epoch'].append(logs.get('loss'))
        self.mode_loss['epoch'].append(logs.get('mode_loss'))
        self.purpose_loss['epoch'].append(logs.get('purpose_loss'))
        self.mode_ce['epoch'].append(logs.get('mode_ce'))
        self.purpose_ce['epoch'].append(logs.get('purpose_ce'))

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        data = pd.DataFrame(self.H)
        data.to_csv(self.csvname,index=False)