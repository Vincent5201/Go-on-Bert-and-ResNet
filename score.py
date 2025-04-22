from mydatasets import get_datasets
from models import load_models
from appilication import prediction
from sklearn.metrics import f1_score, accuracy_score
from tools import myaccn
from torch.utils.data import DataLoader

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

    