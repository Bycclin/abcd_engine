'''
problem:
the computer is too slow,
suposed solution:
use fast make and unmake moves
problems with solution:
for some mysterious reason, the slow
computer searches more and is smarter
than the fast computer.
'''
import random
import pickle
import time
import math
import copy
from piece import *
openings = {
    "e2 e4": ["c7 c5", "c7 c6", "e7 e6", "e7 e5"],
    "d2 d4": ["g8 f6", "f7 f5", "d7 d5"],
    "g1 f3": ["d7 d5", "c7 c5", "g8 f6", "f7 f5"],
    "c2 c4": ["g8 f6", "c7 c5", "f7 f5", "e7 e5"],
}
max_depth = 2
sim_time = 100 # ply's
pondering_move = None
pondering_thread = None
predict_opponent_move = None
# idea:
# Is an added up some of ascii binary values of rnbkq
# as pieces and 0's as empty spaces.
# bin_startpos = 2496
try:
    with open("data.dat", "rb") as f:
        if __name__ == "__main__":
            print("loading...")
        q_table = pickle.load(f)
except FileNotFoundError:
    q_table = {}
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.2
def parse_move(move):
    move = move.replace(',', '')
    parts = move.split()
    if len(parts) != 2 or not all(len(part) == 2 for part in parts):
        return None
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    try:
        start_col, start_row = col_map[parts[0][0]], int(parts[0][1]) - 1
        end_col, end_row = col_map[parts[1][0]], int(parts[1][1]) - 1
        return ((7 - start_row, start_col), (7 - end_row, end_col))
    except (KeyError, ValueError):
        return None
def handle_promotion(board, x, y, piece, computer_promotion=False):
    if (piece == 'P' and x == 0) or (piece == 'p' and x == 7):
        if not computer_promotion:
            while True:
                promotion = input("Promote to (q/r/b/n): ").lower()
                if promotion in ['q', 'r', 'b', 'n']:
                    break
                print("Invalid choice. Choose q, r, b, or n.")
            board[x][y] = promotion.upper() if piece.isupper() else promotion
        else:
            board[x][y] = "Q" if piece.isupper() else "q"
def save_q_table(q_table):
    with open("data.dat", "wb") as file:
        pickle.dump(q_table, file)
def get_q_value(state, action):
    state_tuple = tuple(tuple(row) for row in state)
    return q_table.get((state_tuple, action), 0)
def update_q_table(state, action, reward, next_state):
    state_tuple = tuple(tuple(row) for row in state)
    next_state_tuple = tuple(tuple(row) for row in next_state)
    current_q = get_q_value(state, action)
    next_q_values = [get_q_value(next_state, a) for a in next_state.genMoves("white")]
    max_next_q = max(next_q_values, default=0)
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[(state_tuple, action)] = new_q
def negamax(board, depth, alpha, beta, color, last_move, t_table):
    if depth == 0:# or is_game_over(board):
        #start = time.time()
        values = quiescence_search(board, alpha, beta, color, last_move, t_table)
        #print("Quiesce time:", time.time()-start)
        return values
    best_move = None
    total_nodes = 0
    alpha_original = alpha
    player_color = 'white' if color == 1 else 'black'
    if depth >= 3 and not board.is_check(): #and not is_endgame(board):
        #start = time.time()
        score, best_move, nodes = negamax(board, depth - 3, -beta, -beta + 1, -color, last_move, t_table)
        #print("No-branches search:", time.time()-start)
        total_nodes += nodes
        if score >= beta:
            return beta, best_move, total_nodes
    moves = board.genMoves(player_color)
    moves = order_moves(board, moves, player_color, last_move, depth)
    state_tuple = tuple(tuple(row) for row in board)
    for move in moves:
        if (state_tuple, move) in q_table:
            if q_table[(state_tuple, move)] >= beta:
                return color * q_table[(state_tuple, move)], move, total_nodes
        temp_board = copy.deepcopy(board)
        temp_board[move[1][0]][move[1][1]] = board[move[0][0]][move[0][1]]
        temp_board[move[0][0]][move[0][1]] = " "
        total_nodes += 1
        previous_last_move = last_move
        last_move = move
        is_in_check = board.is_check(temp_board)
        is_capture = board[move[1][0]][move[1][1]] != " "
        is_promotion = board[move[0][0]][move[0][1]].lower() == "p" and move[1][0] == 0 or move[1][0] == 7
        gives_check = False
        if depth == 1 and not is_in_check and not is_capture and not is_promotion and not gives_check:
            static_eval = color * evaluate(temp_board)
            if static_eval + 100 <= alpha:
                total_nodes += 1
                continue
        #start = time.time()
        score, _, nodes = negamax(temp_board, depth - 1, -beta, -alpha, -color, last_move, t_table)
        #print("Reg search:", time.time()-start)
        score += get_q_value(temp_board, move)
        total_nodes += nodes
        last_move = previous_last_move
        score = -score
        last_move = previous_last_move
        if score >= beta:
            #print("beta-cutoff")
            return beta, move, total_nodes
        if score > alpha:
            best_move = move
            alpha = score
    return alpha, best_move, total_nodes
def quiescence_search(board, alpha, beta, color, last_move, t_table, depth=0, max_q_depth=3):
    key = board.hash()
    if key in t_table:
        stand_pat = color * t_table[key][1]
    else:
        stand_pat = evaluate(board)
        t_table[key] = (board, stand_pat)
        stand_pat *= color
    if depth >= max_q_depth: #or is_game_over(board):
        return stand_pat, None, 1
    if stand_pat >= beta:
        return beta, None, 1
    if alpha < stand_pat:
        alpha = stand_pat
    total_nodes = 1
    player_color = 'white' if color == 1 else 'black'
    if not board.is_check(board): #and not is_endgame(board):
        score = stand_pat
        if score >= beta:
            return beta, None, None
    allmoves = board.genMoves(player_color)
    moves = []
    for move in allmoves:
        end_pos = board[move[1][0]][move[1][1]]
        if end_pos != " ":
            moves.append(move)
    moves = order_moves(board, moves, player_color, last_move, depth)
    best_move = None
    for move in moves:
        #temp_board = copy.deepcopy(board)
        #temp_board[move[1][0]][move[1][1]] = temp_board[move[0][0]][move[0][1]]
        #temp_board[move[0][0]][move[0][1]] = " "
        capture = board[move[1][0]][move[1][1]]
        board.move_piece(move[0], move[1])
        previous_last_move = last_move
        last_move = move
        score, _, nodes = quiescence_search(board, -beta, -alpha, -color, last_move, t_table, depth + 1, max_q_depth)
        board.undo_move()
        total_nodes += nodes
        last_move = previous_last_move
        score = -score
        if score >= beta:
            return beta, move, total_nodes
        if score > alpha:
            alpha = score
            best_move = move
    return alpha, best_move, total_nodes
def search(board, t_table, computer_color="black", depth=max_depth, last_move=None):
    best_move = None
    color = -1 if computer_color == "black" else 1
    start_time = time.time()
    alpha = float('-inf')
    beta = float('inf')
    move = None
    eval_score, best_move, total_nodes = negamax(board, depth, alpha, beta, color, last_move, t_table)
    # Castling stuff
    piece = board[best_move[0][0]][best_move[0][1]]
    if piece.isupper():
        if piece == "K":
            white_king_moved = True
        elif piece == "R":
            white_rook_moved = True
    else:
        if piece == "k":
            black_king_moved = True
        elif piece == "r":
            black_rook_moved = True
    return best_move, depth, total_nodes
def order_moves(board, moves, player_color, last_move=None, depth=0):
    opponent_color = 'white' if player_color == 'black' else 'black'
    move_scores = []
    for move in moves:
        score = 0
        piece = board.piece_square(move[0], player_color)
        piece_val = piece_values[piece.upper()]
        board.move_piece(move[0], move[1])
        if player_color == "black":
            score += -evaluate(board)
        else:
            score += evaluate(board)
        board.undo_move()
        player_squares = board.genMoves(player_color)
        opponent_squares = board.genMoves(opponent_color)
        if move[1] in opponent_squares and move[1] not in player_squares:
            score -= math.factorial(piece_val) + 10000
        q_value = get_q_value(board, move)
        score += q_value
        move_scores.append((score, move))
    move_scores.sort(reverse=True, key=lambda x: x[0])
    ordered_moves = [move for score, move in move_scores]
    return ordered_moves
def play_chess():
    board = Position()
    player_color = 'white'
    last_move = None
    t_table = {}
    mode = input("Do you want to play against a person or the computer? ").strip().lower()
    if mode in ["person", "play against a person"]:
        while True:
            board.print_board()
            move = input(f"{player_color}'s turn. Enter your move (e.g., e2 e4): ").strip()
            if move.lower() == 'quit':
                save_q_table(q_table)
                return
            parsed_move = parse_move(move)
            if parsed_move is None:
                print("Invalid input. Please enter your move in the format 'e2 e4'.")
                continue
            if parsed_move not in board.genMoves(player_color):
                valid_move, last_move = board.move_piece(board, start_pos, end_pos, last_move)
                if valid_move:
                    handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                    board.rotate()
                    if board.is_checkmate():
                        print(f"Checkmate! {player_color} wins!")
                        save_q_table(q_table)
                        return
                    board.rotate()
                    if board.is_draw(color for color in ['white', 'black']):
                        print("It's a draw.")
                        save_q_table(q_table)
                        return
                    player_color = 'black' if player_color == 'white' else 'white'
                else:
                    print("Invalid move. Try again.")
            else:
                print("Invalid move. Try again.")
    elif mode in ["computer", "play against the computer"]:
        board.print_board()
        first_open = True
        while True:
            move_valid = False
            while not move_valid:
                move = input("white's move (e.g., e2 e4): ").strip()
                if move.lower() == 'exit':
                    save_q_table(q_table)
                    return
                parsed_move = parse_move(move)
                last_move = parsed_move
                start_pos, end_pos = parsed_move
                if parsed_move is None:
                    print("Invalid input, please enter a move in the correct format (e.g., e2 e4).")
                    continue
                move_result = move not in board.genMoves("white")
                if move_result:
                    _ = board.move_piece(parsed_move[0], parsed_move[1])
                    handle_promotion(board, end_pos[0], end_pos[1], board.piece_square(end_pos, "white"))
                    board.print_board()
                    board.rotate()
                    if board.is_checkmate():
                        print("Checkmate! white wins!")
                        save_q_table(q_table)
                        return
                    board.rotate()
                    if board.is_draw(color for color in ['white', 'black']):
                        print("Stalemate! It's a draw.")
                        save_q_table(q_table)
                    move_valid = True
                else:
                    print("Invalid move, try again.")
            if first_open:
                white_move = ' '.join([f'{chr(start_pos[1] + 97)}{8 - start_pos[0]}' for start_pos in last_move])
                if white_move in openings:
                    computer_move = random.choice(openings[white_move])
                    start_pos, end_pos = parse_move(computer_move)
                    _ = board.move_piece(start_pos, end_pos)
                    board.print_board()
                    first_open = False
                    continue
            state = copy.deepcopy(board)
            stime = time.time()
            action, depth, nodes = search(state, t_table)
            etime = time.time()
            ttime = etime - stime
            print(f"time: {ttime} nodes: {nodes} depth: {depth} score: {evaluate(board)}")
            start_pos, end_pos = action
            _ = board.move_piece(start_pos, end_pos)
            handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
            board.print_board()
            if board.is_checkmate():
                print("Checkmate! black wins!")
                save_q_table(q_table)
                return
            if board.is_draw(color for color in ['white', 'black']):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            reward = 0
            board.rotate()
            if board.is_check():
                reward = 1
            board.rotate()
            reward += evaluate(board) * -1
            next_state = copy.deepcopy(board)
            update_q_table(state, action, reward, next_state)
            first_open = False
    else:
        print("Invalid choice.")
        play_chess()
def sim_chess():
    board = Position()
    last_move = None
    first_open = True
    hist = []
    t_table = {}
    board.print_board()
    while True:
        state = Position()
        stime = time.time()
        white_move, depth, nodes = search(board, t_table, computer_color="white")
        #depth, nodes = "null", "null"
        #white_move = takespiece(board)
        hist.append(white_move)
        if len(hist) >= sim_time:
            save_q_table(q_table)
            return
        etime = time.time()
        ttime = etime - stime
        print(f"time: {ttime} nodes: {nodes} depth: {depth} score: {evaluate(board)}")
        start_pos, end_pos = white_move
        valid_move, last_move = board.move_piece(start_pos, end_pos)
        if valid_move:
            handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
            board.print_board()
            board.roate()
            if board.is_checkmate():
                print("Checkmate! white wins!")
                save_q_table(q_table)
                return
            board.roate()
            if board.is_draw(color for color in ['white', 'black']):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            white_reward = 0
            board.rotate()
            if board.is_check():
                white_reward = 1
            board.rotate()
            white_reward += evaluate(board)
            next_state = copy.deepcopy(board)
            update_q_table(state, white_move, white_reward, next_state)
        if first_open:
            white_move = ' '.join([f'{chr(start_pos[1] + 97)}{8 - start_pos[0]}' for start_pos in last_move])
            if white_move in openings:
                computer_move = random.choice(openings[white_move])
                start_pos, end_pos = parse_move(computer_move)
                valid_move, last_move = board.move_piece(start_pos, end_pos)
                board.print_board()
                first_open = False
                continue
        state = copy.deepcopy(board)
        stime = time.time()
        black_move, depth, nodes = search(board, t_table)
        #depth, nodes = "null", "null"
        #black_move = choose_action(state)
        hist.append(black_move)
        if len(hist) >= sim_time:
            save_q_table(q_table)
            return
        etime = time.time()
        ttime = etime - stime
        print(f"time: {ttime} nodes: {nodes} depth: {depth} score: {evaluate(board)}")
        start_pos, end_pos = black_move
        valid_move, last_move = board.move_piece(start_pos, end_pos)
        if valid_move:
            handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
            board.print_board()
            if board.is_checkmate():
                print("Checkmate! black wins!")
                save_q_table(q_table)
                return
            if board.is_draw(color for color in ['white', 'black']):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            reward = 0
            board.rotate()
            if board.is_check():
                reward = 1
            board.rotate()
            reward += evaluate(board) * -1
            next_state = copy.deepcopy(board)
            update_q_table(state, black_move, reward, next_state)
def main():
    instructions = ''
    play_again = ''
    print("This is chess.")
    print("If you do not know how to play chess, please consult someone who does.")
    while instructions not in "y n t r".split():
        instructions = input("Do you know how this program works? y/n ").strip().lower()
        if instructions == "n":
            print("You enter the starting algebraic coordinate and the ending algebraic coordinate in the input.")
            print("For promotion, you answer the question to promote.")
            print("q for queen, r for rook, n for knight, and b for bishop.")
            play_chess()
        elif instructions == "t":
            for game in range(1000):
                print("game #", game + 1)
                sim_chess()
        else:
            play_chess()
    while play_again not in "y n t r".split():
        play_again = input("Play again? y/n ").strip().lower()
        if play_again == "y":
            play_chess()
            print("Good game.")
            quit()
        elif play_again == "t":
            sim_chess()
            print("Good game.")
            quit()
        #elif instructions == "r":
        #    sim_chess1()
        #    print("Good game.")
        #    quit()
        else:
            print("Awww.")
            quit()
if __name__ == "__main__":
    main()
