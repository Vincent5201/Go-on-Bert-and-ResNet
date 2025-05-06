from tools import *
from gen_board import *
from application import *
from cpptools import value_board

"""
def value_board(board):
    def neighbor_liberty(board, p, x, y):
        pp = 0 if p else 1
        stack = [(x, y)]
        counted = set()
        liberty = 361

        while stack:
            cx, cy = stack.pop()
            counted.add((cx, cy))
            for dx, dy in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if valid_pos(dx, dy) and (dx, dy) not in counted:
                    if board[p][dx][dy]:
                        stack.append((dx, dy))
                    elif board[pp][dx][dy]:
                        liberty = min(liberty, board[3][dx][dy])
        return liberty
    
    def del_die(board, x, y, p):
        board[p][x][y] = 0
        board[3][x][y] = 0
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if valid_pos(dx, dy) and board[p][dx][dy]:
                del_die(board, dx, dy, p)
        return

    def count_neighbor(board, x, y):
        counted = set()
        def next(x, y, dist):
            if dist == 0:
                return 0, 0
            counted.add((x, y))
            p0 = 0
            p1 = 0
            direcs = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
            for dx, dy in direcs:
                if valid_pos(dx, dy) and not ((dx, dy) in counted):
                    if board[0][dx][dy] == 1:
                        p0 += 1
                    elif board[1][dx][dy] == 1:
                        p1 += 1
                    else:
                        t0, t1 = next(dx, dy, dist-1)
                        p0 += t0
                        p1 += t1

            return p0, p1
        return next(x, y, 10)


    board2 = np.array(board, copy=True)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board2[3][i][j] == 1:
                p = 1 if board2[1][i][j] else 0
                if neighbor_liberty(board2, p, i, j) > 1:
                    del_die(board2, i, j, p)
    p0 = 0
    p1 = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board2[0][i][j] == 0 and board2[1][i][j] == 0:
                t0, t1 = count_neighbor(board2, i, j)
                p0 += t0
                p1 += t1
    return p1 > p0 + 5
"""

def get_UCB(node: "MCTSnode", offset):
    if node.n == 0:
        return 9223372036854775807
    if node.parent is None:
        return node.w / node.n + sqrt(2 * np.log(node.n) / node.n)
    return node.w / node.n + sqrt(2 * np.log(node.parent.n + offset) / node.n)

class MCTSnode():
    def __init__(self, board, seq, length, parent: "MCTSnode" = None):
        self.w = 0
        self.n = 0
        self.children = []
        self.expands = []
        self.board = board
        self.seq = seq
        self.length = length
        self.parent = parent
        self.nch = 5
        self.ucb = 9223372036854775807
    
    def expand(self, data_types, models, device):
        if len(self.children) > 0:
            print("expand error")
            return
        poses, _ = vote_next_move(data_types, models, device, self.board, self.seq)
        for i in range(self.nch):
            board2 = np.array(self.board, copy=True)
            seq2 = np.array(self.seq, copy=True)
            seq2[self.length] = poses[i]
            self.expands.append(poses[i])
            x, y = split_move(poses[i])
            channel_01(board2, x, y, self.length + 1)
            channel_2(board2, self.length)
            channel_3(board2, x, y, self.length + 1)
            self.children.append(MCTSnode(board2, seq2, self.length + 1, self))
    
    def select_child(self):
        if len(self.children) == 0:
            return None
        maxucb = self.children[0].ucb
        maxidx = 0
        for i in range(1, self.nch):
            if self.children[i].ucb > maxucb:
                maxucb = self.children[i].ucb
                maxidx = i
        return self.children[maxidx]
    
    def rollout(self, data_types, models, num_moves, device):
        if self.n > 0:
            print("rollout error")
            return

        board2 = np.array(self.board, copy=True)
        seq2 = np.array(self.seq, copy=True)
        move_count = self.length
        while move_count < num_moves:
            move_count += 1
            poses, _ = vote_next_move(data_types, models, device, board2, seq2)
            pose = poses[0]
            x, y = split_move(pose)
            
            channel_01(board2, x, y, move_count)
            channel_2(board2, move_count + 1)
            channel_3(board2, x, y, move_count)
            seq2[move_count-1] = pose
       
        bwin = value_board(board2)

        if bwin:
            return 1
        return 0
    

def MCTS(data_types, models, device, board, seq, length, num_moves, iters):
    root = MCTSnode(board, seq, length)
    iter = 0
    root.expand(data_types, models, device)
    pbar = tqdm(total=iters)
    def next(node: "MCTSnode"):
        nonlocal iter
        if len(node.children) == 0:
            if node.n == 0:
                bwin = node.rollout(data_types, models, num_moves, device)
                iter += 1
                pbar.update(1)
            else:
                node.expand(data_types, models, device)
                bwin = next(node.select_child())
        else:
            bwin = next(node.select_child())

        node.n += 1
        node.w += bwin
        node.ucb = get_UCB(node, 1)
        return bwin

    while iter < iters:
        next(root)
    pbar.close()

    bwinrate = root.children[0].w / root.children[0].n
    idx = 0
    for i in range(1, root.nch):
        r = root.children[i].w / root.children[i].n
        if length % 2:
            if r < bwinrate:
                bwinrate = r
                idx = i
        else:
            if r > bwinrate:
                bwinrate = r
                idx = i
            
    return root.expands[idx]


if __name__ == "__main__":
    data_types = ["Picture"]
    model_config = {}
    model_config["hidden_size"] = HIDDEN_SIZE
    model_config["bert_layers"] = BERT_LAYERS
    model_config["res_channel"] = RES_CHANNELS
    model_config["res_layers"] = RES_LAYERS
    paths = []
    #paths.append("D://codes//python//.vscode//Go_on_Bert_Resnet//models//BERT//mid_s27_30000.pt")
    paths.append("D://codes//python//.vscode//Go_on_Bert_Resnet//models//ResNet//mid_s65_30000.pt")
    #paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//Combine//B20000_R20000.pt"]
    device = "cpu"
    models = load_models(paths, data_types, model_config, device)
    
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
              'ig','qm','pm']
    game = [transfer(step) for step in game]
    board, seq = gen_one_board(game, NUM_MOVES)
    print("start MCTS")
    pose = MCTS(data_types, models, device, board, seq, len(game), max(151, len(game) + 20), 20)
    
    print(transfer_back(pose))