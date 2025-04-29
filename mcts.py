from tools import *
from gen_board import gen_one_board
from application import vote_next_move

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

    return p1 > p0 + 5, p1, p0





if __name__ == "__main__":
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
    board, _ = gen_one_board(game, 240)
    win, b, w = value_board(board)
    print(win)
    print(b)
    print(w)