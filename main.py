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
def parse_move(uci_move, invert=False):
    parts = uci_move.split()
    if len(parts) != 2:
        return None
    indices = []
    for square_str in parts:
        if len(square_str) != 2:
            return None
        file_char, rank_char = square_str[0].lower(), square_str[1]
        if file_char < 'a' or file_char > 'h' or rank_char < '1' or rank_char > '8':
            return None
        f_idx = ord(file_char) - ord('a')
        r_idx_input_perspective = int(rank_char) - 1
        final_r_idx = 0
        if invert:
            final_r_idx = 7 - r_idx_input_perspective
        else:
            final_r_idx = r_idx_input_perspective
        indices.append(f_idx + 8 * final_r_idx) # Corrected for a1=0 mapping
    return (indices[0], indices[1], "")
def handle_promotion(board, parsed):
    end_sq = parsed[1]
    moving_piece = board.piece_square(parsed[0], 'black').upper()
    is_promotion = (moving_piece == 'P' and (56 <= end_sq <= 63))
    if is_promotion:
        while True:
            choice = input("Promote pawn to (q)ueen, (r)ook, k(n)ight, or (b)ishop: ").strip().lower()
            if choice in ['q', 'r', 'n', 'b']:
                promoted_piece = choice.upper()
                break
            else:
                print("Invalid choice.")
        new_bitboards = copy.deepcopy(board.board)
        pawn_index = piece_to_index['P']
        promo_index = piece_to_index[promoted_piece]
        new_bitboards['white'][pawn_index] &= ~(1 << end_sq)
        new_bitboards['white'][promo_index] |= (1 << end_sq)
        return Position(new_bitboards, board.score, board.wc, board.bc, board.ep, board.kp)
    return board
def save_q_table(q_table):
    with open("data.dat", "wb") as file:
        pickle.dump(q_table, file)
def get_q_value(state, action):
    state_tuple = tuple(tuple(row) for row in state)
    return q_table.get((state_tuple, action), 0)
def update_q_table(state, action, reward, next_state):
    state_tuple = tuple(tuple(row) for row in state)
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
    #alpha_original = alpha
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
    #start_time = time.time()
    alpha = float('-inf')
    beta = float('inf')
    _, best_move, total_nodes = negamax(board, depth, alpha, beta, color, last_move, t_table)
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
    board = Position(initial, 0, (True, True), (True, True), 9, 4)
    player_color = 'white'
    t_table = {}
    mode = input("Do you want to play against a person or the computer? ").strip().lower()
    if mode in ["person", "play against a person"]:
        while True:
            board.print_board((player_color=='white'))
            mv_str = input(f"{player_color}'s turn. Enter your move (e.g., e2 e4): ").strip()
            if mv_str.lower() == 'quit':
                save_q_table(q_table)
                return
            parsed = parse_move(mv_str, (player_color=='black'))
            if not parsed:
                print("Invalid input. Use format 'e2 e4'.")
                continue
            legal = list(board.genMoves())
            if parsed in legal:
                board = board.move_piece(parsed)
                board = handle_promotion(board, parsed)
                temp = board.rotate()
                if temp.is_checkmate():
                    board.print_board(is_white=(player_color=='white'))
                    print(f"Checkmate! {player_color} wins!")
                    save_q_table(q_table)
                    return
                if temp.is_draw():
                    board.print_board(is_white=(player_color=='white'))
                    print("It's a draw.")
                    save_q_table(q_table)
                    return
                player_color = 'black' if player_color=='white' else 'white'
            else:
                print("Invalid move. Try again.")
    elif mode in ["computer", "play against the computer"]:
        board.print_board()
        first_open = True
        while True:
            # --- WHITE (human) ---
            mv_valid = False
            while not mv_valid:
                mv_str = input("white's move (e.g., e2 e4): ").strip()
                if mv_str.lower() == 'exit':
                    save_q_table(q_table)
                    return
                parsed = parse_move(mv_str, invert=False)
                if not parsed:
                    print("Invalid input. Use format 'e2 e4'.")
                    continue
                if parsed in board.genMoves():
                    board = board.move_piece(*parsed)
                    handle_promotion(board, parsed[1], board.piece_square(parsed[1], 'white'))
                    board.print_board()
                    mv_valid = True
                    last = parsed
                else:
                    print("Invalid move, try again.")
            if first_open:
                frm = f"{chr(last[0]%8+97)}{last[0]//8+1}"
                to  = f"{chr(last[1]%8+97)}{last[1]//8+1}"
                key = f"{frm} {to}"
                if key in openings:
                    comp = random.choice(openings[key])
                    sp, ep = parse_move(comp)
                    board = board.move_piece(sp, ep)
                    board.print_board()
                    first_open = False
                    continue
            # --- BLACK (computer) ---
            state = copy.deepcopy(board)
            sp, ep, = *search(state, t_table, computer_color="black"), 
            move, depth, nodes = sp
            start = time.time()
            mv, depth, nodes = search(state, t_table, computer_color="black")
            end = time.time()
            print(f"time: {end-start:.2f}s nodes: {nodes} depth: {depth} score: {evaluate(board)}")
            board = board.move_piece(*mv)
            board.print_board()
            if board.is_checkmate():
                print("Checkmate! black wins!")
                save_q_table(q_table)
                return
            if board.is_draw(to_move=None):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            # Q-learn
            reward = 1 if board.rotate().is_check() else 0
            reward += -evaluate(board)
            update_q_table(state, mv, reward, board)
            first_open = False
    else:
        print("Invalid choice.")
        play_chess()
def sim_chess():
    board = Position(initial, 0, (True, True), (True, True), (8, 8), 4)
    last_move = None
    first_open = True
    hist = []
    t_table = {}
    board.print_board()
    while True:
        state = copy.deepcopy(board)
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
