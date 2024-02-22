import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data_handling import *
import torchvision.models.resnet
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch import nn
from torchvision.transforms import Lambda, ToTensor
from scipy.signal import fftconvolve
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #lead_poll = self.img_labels.iloc[idx, 2]
        #diffuse_radiation = self.img_labels.iloc[idx, 8]
        humidity = self.img_labels.iloc[idx, 12]
        wind = self.img_labels.iloc[idx, 14]
        #image = image.transpose(0, 2).transpose(0, 1)
        return image, label, humidity, wind
    
image_transforms = transforms.Compose([
    
    Lambda(lambda x: x[:3]), #normalize bands (RGB)
    #transforms.RandomCrop(224),
    transforms.ToPILImage(),
    ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

learning_rate = .0001
batch_size = 64
epochs = 40
train_labels = "tlabels.csv"
validate_labels = "vlabels.csv"
train_folder = "train_data"
validate_folder = "validate_data"

#normalization stuff
tf = pd.read_csv(train_labels, sep=',', names=['File', 'AQI', "Pollutant","Amount", "Lat", "Long", "Time", "Temp", "Rad", "R24", "R8", "R1", "Humidity", "Pressure", "Wind"])
vf = pd.read_csv(validate_labels, sep=',', names=['File', 'AQI', "Pollutant","Amount", "Lat", "Long", "Time", "Temp", "Rad", "R24", "R8", "R1", "Humidity", "Pressure", "Wind"])
s = [tf, vf]
df = pd.concat(s)
h_min = df['Humidity'].min() #radiation min
h_max = df['Humidity'].max() #radiation max
w_min = df['Wind'].min() #temp min
w_max = df['Wind'].max() #temp max

training_data = CustomImageDataset(train_labels, train_folder, transform=image_transforms)
validation_data = CustomImageDataset(validate_labels, validate_folder, transform=image_transforms)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(2)


class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.features = torchvision.models.resnet.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.features.fc = nn.Identity()
        
        self.relu = nn.SiLU()#nn.ReLU(inplace=True)

        self.latent_fc1 = nn.Linear(2048, 1024) #512
        self.latent_fc2 = nn.Linear(1024, 512)

        self.latent_fc3 = nn.Linear(512, 256)
        self.latent_fc4 = nn.Linear(256, 32)

        self.wind_fc1 = nn.Linear(1, 32)
        self.hum_fc1 = nn.Linear(1, 32)
        self.hum_wind_fc = nn.Linear(32 + 32, 32)
        self.fin_fc1 = nn.Linear(64, 32)
        self.fin_fc2 = nn.Linear(32, 1)

        #more layers (2-3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hum, wind):
        h = self.hum_fc1(hum)
        h = self.relu(h)
        w = self.relu(self.wind_fc1(wind))
        c = torch.cat([h, w], dim=1)
        c = self.hum_wind_fc(c)
        logits = self.features.forward(x)
        logits = self.sigmoid(self.latent_fc1(logits))
        logits = self.latent_fc2(logits)
        logits = self.relu(logits)
        logits = self.latent_fc3(logits)
        logits = self.relu(logits)
        logits = self.latent_fc4(logits) #32        
        fin = torch.cat((logits.view(logits.size(0), -1), c.view(c.size(0), -1)), dim=1)
        fin = self.relu(fin)
        fin = self.fin_fc1(fin)
        fin = self.relu(fin)
        fin = self.fin_fc2(fin)
        return fin
        
        

cnn = ConvNeuralNet(num_classes=500)
cnn = torch.nn.DataParallel(cnn, device_ids=[2, 3, 4, 5]) # no of gpu must be a factor of batch size
cnn.to(device)

class WeightedMSELoss(nn.Module):
    def __init__(self, kernel_size, sigma, frequencies):
        super(WeightedMSELoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.weights = list(fftconvolve(frequencies, self.get_gaussian(kernel_size, sigma)))[4:][:-5]

    def get_gaussian(self, kernel_size, sigma):
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        kernel = np.exp(-0.5 * (x / sigma)**2)
        gaussian = kernel / np.sum(kernel)  # Normalize the kernel to sum to 1
        return gaussian
    
    def forward(self, prediction, truth):

        total_loss = 0
        
        i = 0
        for p,t in zip(prediction, truth):
            index = int(t*500)
            if index > len(self.weights)-1:
                index = len(self.weights)-1
            elif index < 0:
                index = 0
            weight = self.weights[index]
            if weight < 1e-5:
                weight = 1
            else:
                weight = 1 / weight
            i += 1
            diff = abs(p-t)
            total_loss += weight * diff**2 * 100

        total_loss = total_loss / i
        return total_loss

#Handling of frequencies
tdf = pd.read_csv('tlabels.csv', sep=',', names=['File', 'AQI', "Pollutant","Amount", "Lat", "Long", "Time", "Temp", "Rad", "R24", "R8", "R1", "Humidity", "Pressure", "Wind"])
hlist = [0] * 230 #histogram frequencies
for i in tdf['AQI']:
    val = int(round(i))
    hlist[val] += 1
kernel_size = 10
sigma = 2

def get_patches(hlist, kernel_size, sigma):
    # Find the minimum and maximum values in hlist
    lo = min(hlist)
    hi = max(hlist)
    b = 1
    t = 50
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    gaussian = kernel / np.sum(kernel)
    convolution_result = list(fftconvolve(hlist, gaussian))
    convolution_result = [round(i+1) for i in convolution_result]
    f = [1 / i for i in convolution_result]
    lo = min(f)
    lo = min(f)
    f = [round(i/lo) for i in f]
    return f[4:-5]
    
patches = get_patches(hlist, kernel_size, sigma)

def generate_noise(size, range_min, range_max):
    half_size = size // 2
    positive_values = np.random.uniform(range_min, range_max, half_size)
    
    negative_values = -positive_values
    
    if size % 2 != 0:
        positive_values = np.append(positive_values, np.random.uniform(range_min, range_max))
    
    noise = np.concatenate((positive_values, negative_values))
    
    np.random.shuffle(noise)
    
    return noise

loss_fn = WeightedMSELoss(10, 2, hlist)
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
l = []

def get_image_patches(img_tensor, num_patches):
    crop_size = 224
    transform = transforms.RandomCrop(crop_size)
    img_list = []
    for i in range(num_patches):
        pil_img = transforms.ToPILImage()(img_tensor)
        crop_pil = transform(pil_img)
        tensor_img = transforms.ToTensor()(crop_pil)

        #fft stuff
        tran = transforms.Compose([
            transforms.Grayscale()
        ])
        img = tran(tensor_img)
        f_transform = torch.fft.fftn(img)

        f_transform_shifted = torch.fft.fftshift(f_transform)

        magnitude_spectrum = torch.abs(f_transform_shifted)
        full_img = torch.cat((tensor_img, magnitude_spectrum), 0)

        img_list.append(full_img)
    stack = torch.stack(img_list)
    return stack
	
def getF(img_tensor):
    pil_img = transforms.ToPILImage()(img_tensor)
    tensor_img = transforms.ToTensor()(pil_img)
    tran = transforms.Compose([
        transforms.Grayscale()
    ])
    img = tran(tensor_img)
    f_transform = torch.fft.fftn(img)
    f_transform_shifted = torch.fft.fftshift(f_transform)
    magnitude_spectrum = torch.abs(f_transform_shifted)
    full_img = torch.cat((tensor_img, magnitude_spectrum), 0)
    #print(full_img.shape)
    return full_img


torch.autograd.set_detect_anomaly(True)
l = []
def train_loop(dataloader, model, loss_fn, optimizer):
    
    size = len(dataloader.dataset)
    
    model.train()
    for batch, (X, y, h, w) in enumerate(dataloader):
        w=w.to(device)
        h=h.to(device)
        size = list(y.shape)[0]
        X=X.to(device)
        i = 0
        for a,b,hum,wind in zip(X,y,h,w): #a and b are both TENSORS one with SINGULAR image, and the other single GT AQI
            #tensor(106.8352, dtype=torch.float64) <- b
            i+=1
            #print("GT", b)
            gt = b.item()
            
            num_patches = patches[round(gt)]
            
            humidity = ((hum.item() - h_min) / (h_max - h_min))
            windspeed = ((wind.item() - w_min) / (w_max - w_min))


            hum_tensor = torch.full((num_patches,1), humidity)
            wind_tensor = torch.full((num_patches,1), windspeed)
           
            variance = torch.Tensor(generate_noise(num_patches, -5, 5))
            crops = get_image_patches(a, num_patches)
            
            prediction = model(crops, hum_tensor, wind_tensor)
            
           
            noisy_truth = variance + gt
            noisy_truth = noisy_truth.unsqueeze(1).to(device).to(torch.float32)
            noisy_truth = torch.div(noisy_truth, 500)
           
            loss = loss_fn(prediction, noisy_truth)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        loss, current = loss.item(), (batch + 1) * len(X)
        if current % batch_size == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            l.append(loss)
       

acc = []
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    accuracy = 0
    sumerr = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        correct = 0
        total = 0
        a=0
        for batch, (X, y, h, w) in enumerate(dataloader):
            h=h.to(device)
            w=w.to(device)
            X=X.to(device)
            img_list = []
            for a,b in zip(X,y):
                k = getF(a)
                #print(type(k))
                

                img_list.append(k)
            X_n = torch.stack(img_list)

            h = ((h - h_min) / (h_max - h_min)).to(torch.float32)
            w = ((w - w_min) / (w_max - w_min)).to(torch.float32)
            h = h.unsqueeze(1)
            w = w.unsqueeze(1)
            pred = model(X_n, h, w)
            y = y.unsqueeze(1).to(device)
            y = y.to(torch.float32)
            y = torch.div(y, 500)
            
            
            test_loss += loss_fn(pred, y).item()
           
            for prediction,ground_truth in zip(pred, y):
                p = prediction.item() * 500
                gt = ground_truth.item() * 500
                sumerr += abs(gt-p)
                if p <= 50 and gt <= 50:
                    correct += 1
                elif p <= 100 and p > 50 and gt <= 100 and gt > 50:
                    correct += 1
                elif p <= 150 and p > 100 and gt <= 150 and gt > 100:
                    correct += 1
                elif p <= 200 and p > 150 and gt <= 200 and gt > 150:
                    correct += 1
                elif p <= 300 and p > 200 and gt <= 300 and gt > 200:
                    correct += 1
                elif p <= 500 and p > 300 and gt <= 500 and gt > 300:
                    correct += 1
            total += torch.numel(y)
  
    err = correct / total #a / total * 500
    mae = sumerr / total
    print(f"Classification accuracy of {err} for epoch")
    print(f"MAE of {mae} AQI")
    print()
    acc.append(err)


epochs = 100
if __name__ == '__main__':
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, cnn, loss_fn, optimizer)
        test_loop(validation_dataloader, cnn, loss_fn)
    torch.save(cnn.state_dict(), "model1.pt")

