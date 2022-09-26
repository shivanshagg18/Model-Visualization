from torch import autograd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class heatmap():
    def __init__(self, model, scans, index=0):
        self.pred = model(torch.tensor(np.expand_dims(scans[index], axis=0), dtype=torch.float32))
        self.last_layer_output = model.last_layer_output
        
        print("Prediction: ", self.pred)
        
    def GradCAM(self):
        grads = autograd.grad(self.pred[:, self.pred.argmax().item()], self.last_layer_output)
        last_layer_output = np.squeeze(self.last_layer_output)
        last_layer_output = last_layer_output.detach().numpy()
        grads = grads[0][0].mean((1,2,3))
        grads = grads.detach().numpy()
        
        for i in range(last_layer_output.shape[0]):
            last_layer_output[i,:,:,:] *= grads[i]

        heatmap = np.sum(last_layer_output, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap)
        return heatmap
    
    def HiResCAM(self):
        grads = autograd.grad(self.pred[:, self.pred.argmax().item()], self.last_layer_output)
        last_layer_output = np.squeeze(self.last_layer_output)
        last_layer_output = last_layer_output.detach().numpy()
        grads = grads[0][0]
        grads = grads.detach().numpy()
        
        for i in range(last_layer_output.shape[0]):
            last_layer_output[i,:,:,:] *= grads[i,:,:,:]

        heatmap = np.sum(last_layer_output, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap)
        return heatmap