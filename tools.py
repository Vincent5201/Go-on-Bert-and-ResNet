import numpy as np
from tqdm import tqdm
from config import *

def check(game, data_source, num_moves):
    
    if isinstance(game, np.ndarray):
        game = game.tolist()
    if data_source == "foxwq":
        game = game[1:]
    if len(game) < num_moves:
        return False
    for i, step in enumerate(game):
        if i == 0 and not (step in FIRST_STEPS):
            return False
        if isinstance(step, float):
            break
        if(len(step) != 2 or step[0]<'a' or step[0]>'s' or step[1]<'a' or step[1]>'s'):
            return False
        
    return True

def transfer(step):
    if isinstance(step, float):
       return 361
    return (ord(step[0])-97) * 19 + (ord(step[1])-97) 

def split_move(move):
    return int(move // BOARD_SIZE), int(move % BOARD_SIZE)

def valid_pos(dx, dy):
    return dx >= 0 and dx < BOARD_SIZE and dy >= 0 and dy < BOARD_SIZE

def myaccn(pred_logits, true, n):
    total = len(true)
    correct = 0
    for i, p in tqdm(enumerate(pred_logits), total=len(pred_logits), leave=False):
        sorted_indices = (-p).argsort()
        top_k_indices = sorted_indices[:n]  
        if true[i] in top_k_indices:
            correct += 1
    return correct / total

def transfer_back(step):
    return chr((step//19)+97)+chr((step%19)+97) 

