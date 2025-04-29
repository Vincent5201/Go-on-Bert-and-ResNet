import pygame
import sys
import time
import copy
import random

from application import *
from gen_board import *

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (165, 42, 42)
TRANSPARENT_BLACK = (0, 0, 0, 20)  
TRANSPARENT_WHITE = (255, 255, 255, 20) 
BACKGROUND_COLOR = (173, 216, 230)
BUTTON_COLOR = (100, 100, 100)

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 660
GRID_SIZE = BOARD_SIZE - 1
GRID_WIDTH = 650 // GRID_SIZE
GRID_HEIGHT = 650 // GRID_SIZE
LEFT_TOP = 10

INFO_BOX_WIDTH = 150
INFO_BOX_HEIGHT = 50
INFO_BOX_X = 700
INFO_BOX_Y = 130

BUTTON_WIDTH = 80
BUTTON_HEIGHT = 30
BUTTON_X = INFO_BOX_X
BUTTON_Y = INFO_BOX_Y + INFO_BOX_HEIGHT + 20
BUTTON_COOLDOWN = 0.5
RADIUS = 10

running = True
board = np.zeros([CHANNEL_SIZE, BOARD_SIZE, BOARD_SIZE],  dtype=np.float32)
seq = np.full([NUM_MOVES], fill_value=361, dtype=np.int64)
board_history = []
text = ""
games = []
last_move = [-1, -1]
mode = "standby"
button_cool = True
cool_time = 0
turn = 1
playing = False
computer_turn = 0

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("圍棋")

def draw_board(board):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(LEFT_TOP + x * GRID_WIDTH, LEFT_TOP + y * GRID_HEIGHT,
                                GRID_WIDTH, GRID_HEIGHT)
            pygame.draw.rect(screen, BROWN, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE + 1):
            if board[0][i][j]:
                pygame.draw.circle(screen, TRANSPARENT_WHITE,
                                   (LEFT_TOP + j * GRID_WIDTH, LEFT_TOP + i * GRID_HEIGHT), RADIUS)
            if board[1][i][j]:
                pygame.draw.circle(screen, TRANSPARENT_BLACK,
                                   (LEFT_TOP + j * GRID_WIDTH, LEFT_TOP + i * GRID_HEIGHT), RADIUS)

def get_board_position(mouse_pos):
    grid_y = (mouse_pos[0] - LEFT_TOP + GRID_WIDTH // 2) // GRID_WIDTH
    grid_x = (mouse_pos[1] - LEFT_TOP + GRID_WIDTH // 2) // GRID_HEIGHT 
    return grid_x, grid_y

def draw_info_box(coordinates, x, y, width, height):
    info_box_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, WHITE, info_box_rect)
    pygame.draw.rect(screen, BLACK, info_box_rect, 2)

    font = pygame.font.SysFont(None, 24)
    text = font.render(coordinates, True, BLACK)
    text_rect = text.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text, text_rect)

def draw_button(text, x, y, width, height, action):
    button_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)

    font = pygame.font.SysFont(None, 24)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

    mouse_pos = pygame.mouse.get_pos()
    if button_rect.collidepoint(mouse_pos):
        if pygame.mouse.get_pressed()[0]:
            action()

def reset_game():
    global button_cool
    if button_cool:
        global mode, turn, board, text, games, board_history, cool_time, playing
        button_cool = False
        cool_time = time.time()
        mode = "standby"
        turn = 1
        board = np.zeros([CHANNEL_SIZE, BOARD_SIZE, BOARD_SIZE],  dtype=np.float32)
        text = ""
        games = []
        board_history = []
        playing = False

def back():
    global button_cool
    if button_cool:
        global games, turn, board, board_history, cool_time
        button_cool = False
        cool_time = time.time()
        board = board_history[-2]
        board_history = board_history[:-1]
        turn = 0 if turn else 1
        games = games[:-1]
        
def start():
    global button_cool
    if button_cool:
        global cool_time
        button_cool = False
        cool_time = time.time()
        global text, computer_turn, turn, board, text, games, board_history, playing, mode
        if len(games):
            text = "reset first"
            return
        mode = "playing"
        computer_turn = random.randint(0,1)
        playing = True

    
def quit_game():
    pygame.quit()
    sys.exit()

data_types = ["Picture", "Word"]
paths = ["D://codes//python//.vscode//Go_on_Bert_Resnet//models//ResNet//mid_s65_30000.pt",
         "D://codes//python//.vscode//Go_on_Bert_Resnet//models//BERT//mid_s27_30000.pt"]
device = "cpu"
model_config = {}
model_config["hidden_size"] = HIDDEN_SIZE
model_config["bert_layers"] = BERT_LAYERS
model_config["res_channel"] = RES_CHANNELS
model_config["res_layers"] = RES_LAYERS
models = load_models(paths, data_types, model_config, device)

while running:
    screen.fill(BACKGROUND_COLOR)
    if time.time() - cool_time > BUTTON_COOLDOWN:
        button_cool = True

    if playing and computer_turn == turn:
        poses, probs = vote_next_move(data_types, models, device, board, seq)
        tgt = 0
        while poses[tgt] == last_move[turn] or \
                            board[0][poses[tgt] // BOARD_SIZE][poses[tgt] % BOARD_SIZE] or \
                            board[1][poses[tgt] // BOARD_SIZE][poses[tgt] % BOARD_SIZE]:
            tgt += 1
        result = poses[tgt]
        last_move[turn] = result
        x = result // BOARD_SIZE
        y = result % BOARD_SIZE
        channel_01(board, x, y, turn)
        channel_2(board, turn + 1)
        channel_3(board, x, y, turn)
        
        move = transfer_back(result)
        text = move
        games.append(move)
        board_history.append(copy.deepcopy(board))
        turn = 0 if turn else 1
        
    for event in pygame.event.get():
        mouse_pos = pygame.mouse.get_pos()
        grid_x, grid_y = get_board_position(mouse_pos)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if valid_pos(grid_x, grid_y) and \
               board[0][grid_x][grid_y] == 0 and board[1][grid_x][grid_y] == 0:
                
                channel_01(board, grid_x, grid_y, turn)
                channel_2(board, turn + 1)
                channel_3(board, grid_x, grid_y, turn)
                move = transfer_back(grid_x * 19 + grid_y)
                text = move
                games.append(move)
                board_history.append(copy.deepcopy(board))
                turn = 0 if turn else 1
                
    draw_board(board)
    
    draw_info_box(text, INFO_BOX_X, INFO_BOX_Y, INFO_BOX_WIDTH, INFO_BOX_HEIGHT)
    draw_info_box(mode, INFO_BOX_X, INFO_BOX_Y - INFO_BOX_HEIGHT - 10, INFO_BOX_WIDTH, INFO_BOX_HEIGHT)
    
    draw_button("Reset", BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT, reset_game)
    draw_button("Back", BUTTON_X, BUTTON_Y + BUTTON_HEIGHT + 10, BUTTON_WIDTH, BUTTON_HEIGHT, back)
    draw_button("Quit", BUTTON_X, BUTTON_Y + 2 * (BUTTON_HEIGHT + 10), BUTTON_WIDTH, BUTTON_HEIGHT, quit_game)
    draw_button("Start", BUTTON_X, BUTTON_Y + 3 * (BUTTON_HEIGHT + 10), BUTTON_WIDTH, BUTTON_HEIGHT, start)
    
    if valid_pos(grid_x, grid_y):
        if turn:
            pygame.draw.circle(screen, TRANSPARENT_BLACK, mouse_pos, RADIUS)
        else:
            pygame.draw.circle(screen, TRANSPARENT_WHITE, mouse_pos, RADIUS)
    pygame.display.flip()
quit_game()
