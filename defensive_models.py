import torch
from torch import nn
import torch.functional as F

class DenoisingAutoEncoder_1():
    def __init__(self,img_shape = (1,28,28)):
        self.img_shape = img_shape
    
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.AvgPool2d(2),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3,padding=1),
                nn.Sigmoid())
    
    def train(self,data,save_path="./saved_model/model1.pth",v_noise=0,num_epochs=100, batch_size=256,if_save=True):
        optimizer = torch.optim.Adam(self.model.parameters())
        log_interval = 10
        for epoch in range(num_epochs):
            self.model.train()
            for batch_i,(data_train,data_label) in enumerate(data):
                noise = v_noise * torch.randn_like(data_train)
                noisy_data = torch.clamp(data_train+noise,min=0,max=1 )
                
                optimizer.zero_grad()
                output = self.model(noisy_data)
                loss = F.mse_loss(output,data_train)
                loss.backward()
                optimizer.step()
                if batch_i % log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_i}/{len(data)} \t Loss: {loss.item()}]')
        
        torch.save(self.model.state_dict(), save_path)
        
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
            
        


class DenoisingAutoEncoder_2():
    def __init__(self,img_shape = (1,28,28)):
        self.img_shape = img_shape
    
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3,padding=1),
                nn.Sigmoid())
        
    def train(self,data,save_path="./saved_model/model1.pth",v_noise=0,num_epochs=100, batch_size=256,if_save=True):
        optimizer = torch.optim.Adam(self.model.parameters())
        log_interval = 10
        for epoch in range(num_epochs):
            self.model.train()
            for batch_i,(data_train,data_label) in enumerate(data):
                noise = v_noise * torch.randn_like(data_train)
                noisy_data = torch.clamp(data_train+noise,min=0,max=1 )
                
                optimizer.zero_grad()
                output = self.model(noisy_data)
                loss = F.mse_loss(output,data_train)
                loss.backward()
                optimizer.step()
                if batch_i % log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_i}/{len(data)} \t Loss: {loss.item()}]')
        
        torch.save(self.model.state_dict(), save_path)
        
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

