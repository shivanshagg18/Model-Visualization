import pandas as pd
import numpy as np
from Dataset import MyDataset
import torch


class ADNIDataset():
    def __init__(self, data_path="./akshay_data", meta_path="./metadata.csv", npy_column="npy_path", group_column="Group"):
        metadata = pd.read_csv(meta_path)
        ad_scans_path = list(metadata[metadata[group_column] == "AD"][npy_column])
        cn_scans_path = list(metadata[metadata[group_column] == "CN"][npy_column])
        
        print("AD MRI scans: " + str(len(ad_scans_path)))
        print("CN MRI scans: " + str(len(cn_scans_path)))
        
        ad_scans = np.array([np.load(path) for path in ad_scans_path])
        cn_scans = np.array([np.load(path) for path in cn_scans_path])
        ad_labels = np.array([1 for _ in range(len(ad_scans))])
        cn_labels = np.array([0 for _ in range(len(cn_scans))])
        
        X = np.concatenate((ad_scans, cn_scans))
        X_mean = np.mean(X)
        X_std = np.std(X)
        X = (X - X_mean) / X_std
        
        ad_scans = np.array([np.expand_dims(X[i], axis=0) for i in range(len(ad_scans))])
        cn_scans = np.array([np.expand_dims(X[i], axis=0) for i in range(len(ad_scans), len(ad_scans) + len(cn_scans))])
        
        x_train = np.concatenate((ad_scans[:144], cn_scans[:144]), axis=0)
        y_train = np.concatenate((ad_labels[:144], cn_labels[:144]), axis=0)
        x_val = np.concatenate((ad_scans[144:164], cn_scans[144:164]), axis=0)
        y_val = np.concatenate((ad_labels[144:164], cn_labels[144:164]), axis=0)

        x_test = np.concatenate((ad_scans[164:184], cn_scans[164:184]), axis=0)
        y_test = np.concatenate((ad_labels[164:184], cn_labels[164:184]), axis=0)
        print(
            "Number of samples in train, validation, and test are %d, %d, and %d"
            % (x_train.shape[0], x_val.shape[0], x_test.shape[0])
        )
        
        self.ad_scans = ad_scans
        self.cn_scans = cn_scans
        
        training = MyDataset(x_train, y_train)
        validation = MyDataset(x_val, y_val)

        self.train_loader = torch.utils.data.DataLoader(
            training,
            batch_size=4,
            shuffle=True,
            num_workers=4,
        )
        self.val_loader = torch.utils.data.DataLoader(
            validation,
            batch_size=4,
            num_workers=4,
        )
        
        testing = MyDataset(x_test, y_test)

        self.test_loader = torch.utils.data.DataLoader(
            testing,
            batch_size=2,
            num_workers=4,
        )
        
    def train_loader(self):
        return self.train_loader
    
    def val_loader(self):
        return self.val_loader
    
    def test_loader(self):
        return self.test_loader
        
        