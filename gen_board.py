from tools import *
from config import *

def channel_01(board, x, y, turn):
    #plain1 is black
    #plain0 is white
    board[turn%2][x][y] = 1
    live = set()
    died = set()
    def checkDie(x, y, p):
        ans = True
        pp = 0 if p else 1
        if (x, y) in live:
            return False
        if (x, y) in died:
            return True
        died.add((x, y))
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if valid_pos(dx, dy):
                if board[p][dx][dy] == 0 and board[pp][dx][dy] == 0:
                    #neighbor is empty, alive
                    live.add((x, y))
                    return False
                if board[p][dx][dy] == 1:
                    #neighbor is same, check neighbor is alive or not
                    #if one neighbor is alive, itself is alive 
                    ans = ans & checkDie(dx, dy, p)
        if not ans:
            died.remove((x, y))
            live.add((x, y))
        return ans
    
    def del_die(x, y, p):
        board[p][x][y] = 0
        board[3][x][y] = 0
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if valid_pos(dx, dy) and board[p][dx][dy]:
                del_die(dx,dy,p)
        return
    
    directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    for (dx, dy) in directions:
        if valid_pos(dx, dy):
            if turn % 2 == 1 and board[0][dx][dy] and checkDie(dx, dy, 0):
                del_die(dx, dy, 0)
            elif turn % 2 == 0 and board[1][dx][dy] and checkDie(dx, dy, 1):
                del_die(dx, dy, 1)
    return

def channel_2(board, turn):
    #next turn (all 1/0)
    if turn % 2 == 0:
        board[2].fill(1)
    return

def channel_3(board, x, y, turn):
    counted_empty = set()
    counted_pos = set()
    def check_liberty(x, y, p):
        liberty = 0
        pp = 0 if p else 1
        board[p][x][y] = 2
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if valid_pos(dx, dy):
                if board[pp][dx][dy] == 0 and board[p][dx][dy] == 0:
                    if not ((dx, dy) in counted_empty):
                        liberty += 1
                        counted_empty.add((dx,dy))
                elif board[p][dx][dy] == 1:
                    liberty += check_liberty(dx, dy, p)
       
        board[p][x][y] = 1
        counted_pos.add((x, y))    
        return liberty
    
    def set_liberty(x, y, p, liberty):
        board[p][x][y] = 2
        board[3][x][y] = min(6, liberty)
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if valid_pos(dx, dy) and board[p][dx][dy] == 1:
                set_liberty(dx, dy, p, liberty)
        board[p][x][y] = 1
        return
    
    if board[0][x][y] == 0 and board[1][x][y] == 0:
        return
    
    set_liberty(x, y, turn%2, check_liberty(x, y, turn%2))

    pp = 0 if turn%2 else 1
    directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    for (dx, dy) in directions:
        counted_empty.clear()
        if valid_pos(dx, dy) and board[pp][dx][dy] and not ((dx, dy) in counted_pos):
            set_liberty(dx, dy, pp, check_liberty(dx, dy, pp))
    return

def gen_all_boards(games, num_moves):
    total_moves = len(games) * num_moves
    boards = np.zeros([total_moves, CHANNEL_SIZE, BOARD_SIZE, BOARD_SIZE],  dtype=np.float32)
    seqs = np.full([total_moves, num_moves], fill_value=361, dtype=np.int64)
    labels = np.zeros(total_moves)

    game_count = 0
    for _, game in tqdm(enumerate(games), total=len(games), leave=False):
        for j, move in enumerate(game):
            labels[game_count] = move
            if j == 0:
                boards[game_count][2].fill(1)
            else:
                last_move = labels[game_count-1]
                x = last_move // BOARD_SIZE
                y = last_move % BOARD_SIZE
                np.copyto(boards[game_count], boards[game_count - 1])
                np.copyto(seqs[game_count], seqs[game_count - 1])
                seqs[game_count][j] = last_move
                channel_01(boards[game_count], x, y, j)
                channel_2(boards[game_count], j)
                channel_3(boards[game_count], x, y, j)
            game_count += 1

    return boards, seqs, labels

def gen_one_board(game, num_moves):
    board = np.zeros([CHANNEL_SIZE, BOARD_SIZE, BOARD_SIZE],  dtype=np.float32)
    seq = np.full([num_moves], fill_value=361, dtype=np.int64)
    if len(game) % 2:
        board[2].fill(1)
    for j, move in enumerate(game):
        x = move // BOARD_SIZE
        y = move % BOARD_SIZE
        seq[j] = move
        channel_01(board, x, y, j + 1)
        channel_3(board, x, y, j + 1)

    return board, seq