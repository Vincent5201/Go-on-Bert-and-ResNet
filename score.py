from mydatasets import get_datasets
from models import load_models
from application import prediction
from sklearn.metrics import f1_score, accuracy_score
from tools import myaccn
from torch.utils.data import DataLoader
from config import *

def scores(data_config, model_config, device, path):

    _, test_set = get_datasets(data_config, train=False)
    model = load_models([path], [model_config["data_type"]], model_config, device)[0]
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    pred_logits, pred_labels = prediction(data_config["data_type"], model, device, test_loader)

    print(f'accuracy_socre: {accuracy_score(test_set.y, pred_labels)}')
    print(f'accuracy5_socre: {myaccn(pred_logits, test_set.y, 5)}')
    print(f'accuracy10_socre: {myaccn(pred_logits, test_set.y, 10)}')
    print(f'f1_micro: {f1_score(test_set.y, pred_labels, average="micro")}')
    print(f'f1_macro: {f1_score(test_set.y, pred_labels, average="macro")}')

if __name__ == "__main__":

    model_config = {}
    model_config["hidden_size"] = HIDDEN_SIZE
    model_config["bert_layers"] = BERT_LAYERS
    model_config["res_channel"] = RES_CHANNELS
    model_config["res_layers"] = RES_LAYERS
    model_config["data_type"] = "Combine"
    data_config = {}
    data_config["path"] = 'datas/data_Foxwq_9d.csv'
    data_config["data_size"] = 20
    data_config["offset"] = 1000
    data_config["data_type"] = "Combine"
    data_config["data_source"] = "foxwq"
    data_config["num_moves"] = 240

    #paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//BERT//mid_s27_30000.pt"]
    #paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//ResNet//mid_s65_30000.pt"]
    paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//Combine//B20000_R20000.pt"]
    device = "cpu"
    models = load_models(paths, [model_config["data_type"]], model_config, device)
    num_moves = 240
    scores(data_config, model_config, device, paths[0])