from tools import *
from gen_board import *
from application import *

# 對每個節點，設定好10種下一步，然後都用模型的一選跑到240步，再用 value func. 去判斷這場誰贏
# 一氣的算死棋刪掉，但要檢查一氣的對方鄰居是不是也一氣
# 對於無人的空地，看接觸誰的數量多，就算誰的地

def value_board(board):
    def neighbor_liberty(board, p, x, y):
        pp = 0 if p else 1
        counted = set()
        def next(x, y):
            counted.add((x, y))
            liberty = 361
            direcs = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
            for dx, dy in direcs:
                if valid_pos(dx, dy) and not ((dx, dy) in counted):
                    if board[p][dx][dy]:
                        next(dx, dy)
                    elif board[pp][dx][dy]:
                        liberty = min(liberty, board[3][dx][dy])

            return liberty
        return next(x, y)
    
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
                if valid_pos(dx, dy) and board[0][dx][dy] != -1 and not ((dx, dy) in counted):
                    if board[0][dx][dy] == 1:
                        p0 += 1
                    elif board[1][dx][dy] == 1:
                        p1 += 1
                    else:
                        board[0][dx][dy] = -1
                        t0, t1 = next(dx, dy, dist-1)
                        p0 += t0
                        p1 += t1

            return p0, p1
        return next(x, y, 10)


    board2 = np.zeros_like(board)
    np.copyto(board2, board)
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

def get_UCB(node: "MCTSnode"):
    if node.n == 0:
        return 9223372036854775807
    if node.parent is None:
        return node.w / node.n + sqrt(2 * np.log(node.n) / node.n)
    return node.w / node.n + sqrt(2 * np.log(node.parent.n) / node.n)

class MCTSnode():
    def __init__(self, game=None, parent: "MCTSnode" = None):
        self.w = 0
        self.n = 0
        self.parent = parent
        self.children = []
        self.game = game if game is not None else []
        self.nch = 10
    
    def expand(self, data_types, models, num_moves, device):
        if len(self.children) > 0:
            print("expand error")
            return
        poses, _ = get_next_move(self.game, data_types, models, num_moves, device)
        for i in range(self.nch):
            self.children.append(MCTSnode(self.game + [poses[i]], self))
    
    def select_child(self):
        if len(self.children) == 0:
            return None
        maxucb = get_UCB(self.children[0])
        maxidx = 0
        for i in range(1, self.nch):
            ucb = get_UCB(self.children[i])
            if ucb > maxucb:
                maxucb = ucb
                maxidx = i
        return self.children[maxidx]
    
    def rollout(self, data_types, models, num_moves, device):
        if self.n > 0:
            print("rollout error")
            return
    
        move_count = len(self.game)
        board, seq = gen_one_board(game, num_moves)
        while move_count < num_moves:
            move_count += 1
            poses, _ = vote_next_move(data_types, models, device, board, seq)
            pose = poses[0]
            x = pose // BOARD_SIZE
            y = pose % BOARD_SIZE
            channel_01(board, x, y, move_count)
            channel_2(board, move_count + 1)
            channel_3(board, x, y, move_count)
            seq[move_count-1] = pose
        
        bwin = value_board(board)
        if bwin:
            return 1
        return 0
        


def MCTS(data_types, models, device, game, num_moves, iters):
    root = MCTSnode(game)
    iter = 0
    root.expand(data_types, models, num_moves, device)
    pbar = tqdm(total=iters)
    def next(node: "MCTSnode"):
        nonlocal iter
        if len(node.children) == 0:
            if node.n == 0:
                bwin = node.rollout(data_types, models, num_moves, device)
                iter += 1
                pbar.update(1)
            else:
                node.expand(data_types, models, num_moves, device)
                bwin = next(node.select_child())
        else:
            bwin = next(node.select_child())

        node.n += 1
        node.w += bwin
        return bwin

    while iter < iters:
        next(root)
    pbar.close()

    print(root.w)
    print(root.n)

    





if __name__ == "__main__":
    data_types = ["Word", "Picture"]
    model_config = {}
    model_config["hidden_size"] = HIDDEN_SIZE
    model_config["bert_layers"] = BERT_LAYERS
    model_config["res_channel"] = RES_CHANNELS
    model_config["res_layers"] = RES_LAYERS
    paths = []
    paths.append("D://codes//python//.vscode//Go_on_Bert_Resnet//models//BERT//mid_s27_30000.pt")
    paths.append("D://codes//python//.vscode//Go_on_Bert_Resnet//models//ResNet//mid_s65_30000.pt")
    #paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//Combine//B20000_R20000.pt"]
    device = "cpu"
    models = load_models(paths, data_types, model_config, device)
    
    game = ['dq','dd','pp','pc','qe','co','od','oc','nd','nc','md','lc','mc','mb','cp','do','ld',
              'kc','kd','jc','jd','ic','bo','bn','bp','cm','qc','pd','qd','pe','pf','qf','qg',
              'rf','rg','of','pg','oe','id','hd','he','ge','gd','hc','fd','hf','ie','gf','pb',
              'ob','ee','cf','de','ce','eg','gh','cd','cc','bd','bc','dc','be','ed','ad','qb',
              'jg','dd']
    game = [transfer(step) for step in game]
    print("start MCTS")
    MCTS(data_types, models, device, game, 100, 100)