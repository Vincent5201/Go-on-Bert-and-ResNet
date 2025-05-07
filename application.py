from tools import *
from gen_board import gen_one_board
from mydatasets import gen_token_type
from config import *
from models import load_models
import torch

def prediction(data_type, model, device, test_loader):
    model.eval()
    pred_logits = []
    pred_labels = []
    with torch.no_grad():
        for datas in tqdm(test_loader, leave=False):
            if data_type == "Word":
                x, m, t, _ = (d.to(device) for d in datas)
                pred = model(x, m, t)
            elif data_type == "Combine":
                xp, xw, m, t, _ = (d.to(device) for d in datas)
                pred = model(xp, xw, m, t)
            elif data_type == "Picture":
                x, _ = (d.to(device) for d in datas)
                pred = model(x)
            ans = torch.max(pred,1).indices
            pred_logits.extend(pred.cpu().numpy())
            pred_labels.extend(ans.cpu().numpy())

    return pred_logits, pred_labels

def next_move(data_type, model, device, board=None, seq=None):
    # board ans seq are np.array
    if not board is None:
        board = board.reshape(1, *board.shape)
    if not seq is None:
        seq = seq.reshape(1, *seq.shape)

    model.eval()
    if data_type == "Word":
        token_types = gen_token_type(seq, board)
        x = torch.tensor(seq, dtype=torch.long).to(device)
        mask = (x != 361).detach().long().to(device)
        t = torch.tensor(token_types, dtype=torch.long).to(device)
        
        with torch.no_grad():
            pred = model(x, mask, t)[0]

    elif data_type == "Picture":
        x = torch.tensor(board, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(x)[0]
    
    elif data_type == 'Combine':
        token_types = gen_token_type(seq, board)
        xw = torch.tensor(seq, dtype=torch.long).to(device)
        mask = (xw != 361).detach().long().to(device)
        t = torch.tensor(token_types, dtype=torch.long).to(device)
        xp = torch.tensor(board, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(xp, xw, mask, t)[0]

    pred = torch.nn.functional.softmax(pred, dim=-1).cpu().numpy()
    return pred

def vote_next_move(data_types, models, device, board=None, seq=None):
    probs = np.zeros([BOARD_SIZE * BOARD_SIZE])
    for i, model in enumerate(models):
        pred = next_move(data_types[i], model, device, board, seq)
        probs += pred
    return np.argsort(-probs), probs

def get_next_move(game, data_types, models, num_moves, device):
    board, seq = gen_one_board(game, num_moves)
    poses, probs = vote_next_move(data_types, models, device, board, seq)

    return poses, probs

if __name__ == "__main__":
    data_types = ["Combine"]
    model_config = {}
    model_config["hidden_size"] = HIDDEN_SIZE
    model_config["bert_layers"] = BERT_LAYERS
    model_config["res_channel"] = RES_CHANNELS
    model_config["res_layers"] = RES_LAYERS
    #paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//BERTex//mid_s27_30000.pt"]
    #paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//ResNet//mid_s65_30000.pt"]
    paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//Combine//B20000_R20000.pt"]
    device = "cpu"
    models = load_models(paths, data_types, model_config, device)
    num_moves = 240
    
    game = ['dq','dd','pp','pc','qe','co','od','oc','nd','nc','md','lc','mc','mb','cp','do','ld',
              'kc','kd','jc','jd','ic','bo','bn','bp','cm','qc','pd','qd','pe','pf','qf','qg',
              'rf','rg','of','pg','oe','id','hd','he','ge','gd','hc','fd','hf','ie','gf','pb',
              'ob','ee','cf','de','ce','eg','gh','cd','cc','bd','bc','dc','be','ed','ad','qb',
              'jg','dd','dh','eh','di','ei','lg','dj','cj','ck','dk','ej','bk','ci','cl','dg',
              'ch','cg','bh','bg','bi','qq','cb','db','da','ab','ac','af','ae','ea','ca','fb',
              'gb','gc','hb','og','ng','nf','mf','ne','gj','nh','mg','lb','na','df','bb','aa',
              'eq','ep','fq','fp','gp','gq','gr','hq','dr','dp','hr','iq','ir','jq','cr','la',
              'ka','go','jr','kq','kr','lr','lq','mr','lp','mh','nq','nr','oq','or','io','hp',
              'ko','pa','oa','lh','kh','ki','ji','kj','jj','mq','mp','kk','oo','kf','kg','if',
              'ig','qm','pm','ql']
    game = [transfer(step) for step in game]
    poses, probs = get_next_move(game, data_types, models, num_moves, device)
    print(poses[0])
