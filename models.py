from transformers import BertModel, BertConfig
import torch.nn as nn
import torch
import torch.nn.functional as F

class myBert(nn.Module):
    def __init__(self, config, num_labels=361, p_model = None):
        super(myBert, self).__init__()
        if p_model:
            self.bert = p_model
        else:
            self.bert = BertModel(config)
        self.linear1 = nn.Linear(config.hidden_size, 512)
        self.linear2 = nn.Linear(512, num_labels)
    def forward(self, x, m, t):
        output = self.bert(input_ids=x, attention_mask=m, token_type_ids=t)["last_hidden_state"]
        logits = torch.mean(output, dim=1)
        logits = self.linear1(logits)
        logits = self.linear2(logits)
        return logits

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, padding=int((kernal_size-1)/2))
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  
        nn.init.kaiming_normal_(self.cnn.weight, mode="fan_out", nonlinearity="relu")
    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.cnn1 = ConvBlock(in_channels, out_channels, 3)
        self.cnn2 = ConvBlock(out_channels, out_channels, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        out = F.relu(self.cnn1(x), inplace=True)
        out = self.cnn2(out)
        out += identity
        return F.relu(out, inplace=True)

class myResNet(nn.Module):
    def __init__(self, in_channels, res_channels, res_layers):
        super(myResNet, self).__init__()
        self.cnn_input = ConvBlock(in_channels, res_channels, 3)
        self.residual_tower = nn.Sequential(
            *[ResBlock(res_channels, res_channels) for _ in range(res_layers)]
        )
        self.policy_cnn = ConvBlock(res_channels, 2, 1)
        self.policy_fc = nn.Linear(2 * 19 * 19, 19 * 19)
    def forward(self, planes):
        x = self.cnn_input(planes)
        x = self.residual_tower(x)
        pol = self.policy_cnn(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))
        return pol

class Combine(nn.Module):
    def __init__(self, modelB, modelR):
        super(Combine, self).__init__()
        self.m1 = modelB
        self.m2 = modelR
        for param in self.m1.parameters():
            param.requires_grad = False
        for param in self.m2.parameters():
            param.requires_grad = False
        self.linear1 = nn.Linear(722, 512)
        self.linear2 = nn.Linear(512, 361)
    def forward(self, xp, xw, m, tt):
        yp = self.m2(xp)
        yw = self.m1(xw, m, tt)
        yw = nn.functional.softmax(yw, dim=-1)
        yp = nn.functional.softmax(yp, dim=-1)
        logits = torch.cat((yp, yw), dim=-1)
        logits = self.linear1(logits)
        logits = self.linear2(logits)
        return logits

def get_model(model_config, device, path_r=None, path_b=None):
    model = None
    if model_config["data_type"] == 'Word':
        config = BertConfig()
        config.type_vocab_size = 7
        config.hidden_size = model_config["hidden_size"]
        config.num_hidden_layers = model_config["bert_layers"]
        config.vocab_size = 363
        config.num_attention_heads = 1
        config.intermediate_size = config.hidden_size * 4
        config.position_embedding_type = "relative_key"
        model = myBert(config, 361)

    elif model_config["data_type"] == 'Picture':
        res_channel = model_config["res_channel"]
        layers = model_config["res_layers"]
        in_channel = 4
        model = myResNet(in_channel, res_channel, layers)
    
    elif model_config["data_type"] == 'Combine':
        model_config["data_type"] = 'Word'
        modelb = get_model(model_config, device)
        model_config["data_type"] = 'Picture'
        modelr = get_model(model_config, device)
        model_config["data_type"] = 'Combine'
        if path_b and path_r:
            modelb.load_state_dict(torch.load(path_b, map_location=device, weights_only=True))
            modelr.load_state_dict(torch.load(path_r, map_location=device, weights_only=True))
        model = Combine(modelR=modelr, modelB=modelb)
        

    return model.to(device)

def load_models(paths, data_types, model_config, device):
    models = []
    for i, data_type in enumerate(data_types):
        model_config["data_type"] = data_type
        state = torch.load(paths[i], map_location=device, weights_only=True)
        model = get_model(model_config, device)
        model.load_state_dict(state)
        models.append(model)

    return models

if __name__ == "__main__":
    model_config = {}
    model_config["data_type"] = "BERT"
    model = get_model(model_config, "cpu")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    print(model)
