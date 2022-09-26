import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]