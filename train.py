import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random

from mydatasets import get_datasets
from models import get_model
from tools import myaccn
from config import *

# config od data amnd model
data_config = {}
data_config["path"] = 'datas/data_240119.csv'
data_config["data_size"] = 5000
data_config["offset"] = 0
data_config["data_type"] = "Picture"
data_config["data_source"] = "pros"
data_config["num_moves"] = 240

model_config = {}
model_config["data_type"] = "ResNet"
model_config["hidden_size"] = HIDDEN_SIZE
model_config["bert_layers"] = BERT_LAYERS
model_config["res_channel"] = RES_CHANNELS
model_config["res_layers"] = RES_LAYERS

device = "cuda:0"
pathb = "models/BERTex/mid_s63_5000.pt"
pathr = "models/ResNet/mid_s57_5000.pt"
path1 = path2 = None
model = get_model(model_config, pathr=pathr, pathb=pathb).to(device)
if model_config["data_type"] == "Combine":
    model.mb = model.mb.to(device)
    model.mr = model.mr.to(device)

#model.load_state_dict(torch.load(f'models/BERT/model2.pt'))

# config of training
batch_size = 64
num_epochs = 50
lr = 5e-4
save = True
random_seed = random.randint(0,100)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
print(f'rand_seed:{random_seed}')

trainData, testData = get_datasets(data_config)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fct = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testData, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    losses = []
    for datas in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        if data_config["data_type"] == "Word":
            x, m, t, y = (d.to(device) for d in datas)
            pred = model(x, m, t)
        elif data_config["data_type"] == "Combine":
            xp, xw, m, t, y = (d.to(device) for d in datas)
            pred = model(xp, xw, m, t)
        elif data_config["data_type"] == "Picture":
            x, y = (d.to(device) for d in datas)
            pred = model(x)
        loss = loss_fct(pred,y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    print(f'epoch {epoch+1}/{num_epochs},  train_loss = {sum(losses)/len(losses):.4f}')

    model.eval()
    pred_label = []
    pred_logit = []
    true = []
    with torch.no_grad():
        for datas in tqdm(test_loader, leave=False):
            if data_config["data_type"] == "Word":
                x, m, t, y = (d.to(device) for d in datas)
                pred = model(x, m, t)
            elif data_config["data_type"] == "Combine":
                xp, xw, m, t, y = (d.to(device) for d in datas)
                pred = model(xw, m, t, xp)
            elif data_config["data_type"] == "Picture":
                x, y = (d.to(device) for d in datas)
                pred = model(x)
            ans = torch.max(pred,1).indices
            pred_logit.extend(pred.cpu().numpy())
            pred_label.extend(ans.cpu().numpy())
            true.extend(y.cpu().numpy())

    print(f'val_loss:{loss_fct(pred, y):.4f}')
    print(f'accuracy5:{myaccn(pred_logit, true, 5)}')
    print(f'accuracy:{accuracy_score(pred_label, true)}')
    if save:
        torch.save(model.state_dict(), f'/home/F74106165/Language_Go/tmpmodels1/model{epoch+1}.pt')