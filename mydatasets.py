import pandas as pd
import gc
import torch
from torch.utils.data import Dataset
from gen_board import gen_all_boards
from tools import *

class ResNetDataset(Dataset):
    # data loading
    def __init__(self, boards, labels):
        
        self.x = torch.tensor(boards, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.n_samples = boards.shape[0]
        gc.collect()
        
    def __getitem__(self, index):  
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def gen_token_type(seqs, boards):
    token_types = np.zeros(seqs.shape)
    for i, game in enumerate(seqs):
        for j, move in enumerate(game):
            if move == 361:
                break
            token_types[i][j] = boards[i][3][move // BOARD_SIZE][move % BOARD_SIZE]
    return token_types


class BERTDataset(Dataset):
    # data loading
    def __init__(self, boards, seqs, labels):
        
        token_types = gen_token_type(seqs, boards)

        self.x = torch.tensor(seqs, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.mask = (self.x != 361).detach().long()
        self.token_types = torch.tensor(token_types, dtype=torch.long)
        self.n_samples = self.y.shape[0]
        gc.collect()

    def __getitem__(self, index):  
        return self.x[index], self.mask[index], self.token_types[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
class CombineDataset(Dataset):
    # data loading
    def __init__(self, boards, seqs, labels):
        
        token_types = np.zeros(seqs.shape)
        for i, game in enumerate(seqs):
            for j, move in enumerate(game):
                if move == 361:
                    break
                token_types[i][j] = boards[i][3][move // BOARD_SIZE][move % BOARD_SIZE]

        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.boards = torch.tensor(boards, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.mask = (self.x != 361).detach().long()
        self.token_types = torch.tensor(token_types, dtype=torch.long)
        self.n_samples = self.y.shape[0]
        gc.collect()

    def __getitem__(self, index):  
        return self.boards[index], self.seqs[index], self.mask[index], self.token_types[index], self.y[index]

    def __len__(self):
        return self.n_samples

def get_datasets(data_config, split_rate=0.1, train=True): 
    df = pd.read_csv(data_config["path"], encoding="ISO-8859-1", on_bad_lines='skip')
    df = df.sample(frac=1,replace=False,random_state=8596).reset_index(drop=True)\
        .to_numpy()[data_config["offset"]:data_config["offset"] + data_config["data_size"]]
    # games contains [["pq"...], ["dd"...], ...], len(game) > num_moves
    games = [game for game in df if check(game, data_config["data_source"], data_config["num_moves"])]
    print(f'valid_rate:{len(games)/len(df)}')
    print(f'has {len(games)} games')
    
    # transfer to 0~360, and pad 361
    games = [[transfer(step) for step in game[:data_config["num_moves"]]] for game in games]
    print("transfer finish")

    boards, seqs, labels = gen_all_boards(games, data_config["num_moves"])
    split = int(len(games) * split_rate)
    train_dataset = None
    eval_dataset = None

    if data_config["data_type"] == 'Word':
        if train:
            train_dataset = BERTDataset(boards[split:], seqs[split:], labels[split:])
            eval_dataset = BERTDataset(boards[:split], seqs[:split], labels[:split])
            print(f'trainDatab shape:{train_dataset.x.shape}')
        else:
            eval_dataset = BERTDataset(boards, seqs, labels)
        
    elif data_config["data_type"] == 'Picture':
        if train:
            train_dataset = ResNetDataset(boards[split:], labels[split:])
            eval_dataset = ResNetDataset(boards[:split], labels[:split])
            print(f'trainDatap shape:{train_dataset.x.shape}')
        else:
            eval_dataset = ResNetDataset(boards, labels)
    elif data_config["data_type"] == "Combine":
        if train:
            train_dataset = CombineDataset(boards[split:], seqs[split:], labels[split:])
            eval_dataset = CombineDataset(boards[:split], seqs[:split], labels[:split])
            print(f'trainDatab shape:{train_dataset.seqs.shape}')
            print(f'trainDatap shape:{train_dataset.boards.shape}')
        else:
            eval_dataset = CombineDataset(boards, seqs, labels)

    gc.collect()
    return train_dataset, eval_dataset