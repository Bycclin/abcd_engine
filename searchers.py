"""
This is the module for miscelanous
ideas for the brains of the chess engine.
Also the old search functions, for testing and
play. Of the old versions, it includes
v2024(1.0).
"""
'''
import copy
from engine import Engine, Board, openings

def play_chess():
    engine = Engine()
    board_obj = Board()
    player_color = 'White'

    mode = input("Play person or computer? ").strip().lower()
    if mode in ("person", "play against a person"):
        while True:
            board_obj.print_board()
            move_str = input(f"{player_color}'s move (like e2 e4): ")
            if move_str.lower() == 'quit':
                engine.save_q_table()
                return
            parsed = board_obj.parse_move(move_str)
            if not parsed:
                print("Invalid format.")
                continue
            (sx, sy), (ex, ey) = parsed
            # See if it's in gather_moves():
            all_moves = board_obj.gather_moves()
            if parsed in all_moves[player_color]:
                captured = board_obj.board[ex][ey]
                valid, mv = board_obj.move_piece((sx,sy),(ex,ey), board_obj.last_move)
                # handle promotion
                board_obj.handle_promotion(ex, ey, computer_promotion=False)
                if board_obj.is_checkmate()["Black" if player_color=='White' else "White"]:
                    print(f"Checkmate! {player_color} wins!")
                    engine.save_q_table()
                    return
                if board_obj.is_stalemate("Black" if player_color=='White' else "White"):
                    print("Stalemate!")
                    engine.save_q_table()
                    return
                player_color = "Black" if player_color=="White" else "White"
            else:
                print("Illegal move. Try again.")

    elif mode in ("computer", "play against the computer"):
        # White is human, Black is engine
        while True:
            # Human (White) move
            board_obj.print_board()
            move_str = input("White's move: ")
            if move_str.lower() == 'quit':
                engine.save_q_table()
                return
            parsed = board_obj.parse_move(move_str)
            if not parsed:
                print("Invalid input.")
                continue
            all_moves = board_obj.gather_moves()
            if parsed in all_moves["White"]:
                (sx, sy), (ex, ey) = parsed
                captured = board_obj.board[ex][ey]
                board_obj.move_piece((sx,sy),(ex,ey), board_obj.last_move)
                board_obj.handle_promotion(ex, ey, computer_promotion=False)
                if board_obj.is_checkmate()["black"]:
                    print("Checkmate! White wins!")
                    engine.save_q_table()
                    return
                if board_obj.is_stalemate("black"):
                    print("Stalemate!")
                    engine.save_q_table()
                    return
            else:
                print("Illegal move, try again.")
                continue

            # Engine (Black) move
            state_copy = copy.deepcopy(board_obj)
            best_move, depth_used, _ = engine.search(state_copy, "Black", engine.max_depth)
            if not best_move:
                # no moves => probably stalemate or checkmate
                if board_obj.is_checkmate()["black"]:
                    print("Checkmate! White wins!")
                else:
                    print("Stalemate!")
                engine.save_q_table()
                return

            (sx, sy), (ex, ey) = best_move
            captured = board_obj.board[ex][ey]
            board_obj.move_piece((sx,sy),(ex,ey), board_obj.last_move)
            board_obj.handle_promotion(ex, ey, computer_promotion=True)

            if board_obj.is_checkmate()["white"]:
                board_obj.print_board()
                print("Checkmate! Black wins!")
                engine.save_q_table()
                return
            if board_obj.is_stalemate("white"):
                board_obj.print_board()
                print("Stalemate!")
                engine.save_q_table()
                return

            # Q-learning update (if you want):
            old_eval = engine.evaluate(state_copy)
            new_eval = engine.evaluate(board_obj)
            reward = (new_eval - old_eval)  # simplistic approach
            engine.update_q_table(state_copy.board, best_move, reward, board_obj.board)

    else:
        print("Unknown mode. Try again.")
        play_chess()

def sim_chess():
    """
    Simple self-play for Q-learning demonstration
    """
    engine = Engine()
    board_obj = Board()
    while True:
        # White move
        state_copy = copy.deepcopy(board_obj)
        wmove, _, _ = engine.search(state_copy, "White", engine.max_depth)
        if not wmove:
            break
        (sx, sy), (ex, ey) = wmove
        captured = board_obj.board[ex][ey]
        board_obj.move_piece((sx,sy),(ex,ey), board_obj.last_move)
        board_obj.handle_promotion(ex,ey,True)

        if board_obj.is_game_over():
            break
        # Q-update
        old_eval = engine.evaluate(state_copy)
        new_eval = engine.evaluate(board_obj)
        reward_w = new_eval - old_eval
        engine.update_q_table(state_copy.board, wmove, reward_w, board_obj.board)

        # Black move
        state_copy = copy.deepcopy(board_obj)
        bmove, _, _ = engine.search(state_copy, "Black", engine.max_depth)
        if not bmove:
            break
        (sx, sy), (ex, ey) = bmove
        captured = board_obj.board[ex][ey]
        board_obj.move_piece((sx,sy),(ex,ey), board_obj.last_move)
        board_obj.handle_promotion(ex,ey,True)

        if board_obj.is_game_over():
            break
        old_eval = new_eval
        new_eval = engine.evaluate(board_obj)
        reward_b = -(new_eval - old_eval)
        engine.update_q_table(state_copy.board, bmove, reward_b, board_obj.board)

    engine.save_q_table()

def main():
    while True:
        choice = input("Type 'play', 'sim', or 'quit': ").strip().lower()
        if choice == 'play':
            play_chess()
        elif choice == 'sim':
            sim_chess()
        elif choice == 'quit':
            print("Bye!")
            break
        else:
            print("Unknown choice. Try again.")

if __name__ == "__main__":
    main()
import copy
import pickle

piece_values = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
unicode_pieces = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    ' ': ' '
}
openings = {
    "e2 e4": ["c7 c5", "c7 c6", "e7 e6", "e7 e5"],
    "d2 d4": ["g8 f6", "f7 f5", "d7 d5"],
    "g1 f3": ["d7 d5", "c7 c5", "g8 f6", "f7 f5"],
    "c2 c4": ["g8 f6", "c7 c5", "f7 f5", "e7 e5"],
}

# Directions for sliding pieces
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK_DIRS   = [(-1,  0), (1, 0), (0, -1), (0,  1)]
QUEEN_DIRS  = BISHOP_DIRS + ROOK_DIRS

KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                ( 1, -2), ( 1, 2), ( 2, -1), ( 2, 1)]
KING_MOVES   = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]

def in_bounds(x, y):
    return 0 <= x < 8 and 0 <= y < 8

initial_board = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"],
]

class Board:
    """
    A single-pass Board class. It gathers all moves for White and Black
    in one method, and also uses dictionary-based is_check, is_checkmate.
    """
    def __init__(self, board=None):
        # If you need to track castling rights:
        self.white_king_moved = False
        self.black_king_moved = False
        self.white_rook_moved_a = False  # track e.g. "a-rook" if needed
        self.white_rook_moved_h = False
        self.black_rook_moved_a = False
        self.black_rook_moved_h = False

        # You can store a last move for en passant
        self.last_move = None  # ((sx,sy),(ex,ey))

        self.board = copy.deepcopy(board) if board else copy.deepcopy(initial_board)

    def print_board(self, is_white=True):
        if is_white:
            print("  a b c d e f g h")
        else:
            print("  h g f e d c b a")
        for row in range(8):
            print(8 - row, end=" ")
            for col in range(8):
                print(unicode_pieces[self.board[row][col]], end=" ")
            print(8 - row)
        if is_white:
            print("  a b c d e f g h")
        else:
            print("  h g f e d c b a")

    def parse_move(self, move_str):
        """
        Convert something like 'e2 e4' -> ((6,4),(4,4)) in our 0-based row col.
        Because 'e2' is col='e'=4, row='2'=1 from the bottom => internal row=6.
        """
        move_str = move_str.replace(',', '')
        parts = move_str.split()
        if len(parts) != 2 or not all(len(part) == 2 for part in parts):
            return None
        col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        try:
            start_col, start_row = col_map[parts[0][0]], int(parts[0][1]) - 1
            end_col, end_row = col_map[parts[1][0]], int(parts[1][1]) - 1
            return ((7 - start_row, start_col), (7 - end_row, end_col))
        except (KeyError, ValueError):
            return None

    def find_king(self, color):
        """Find the king position for 'White' or 'Black'."""
        target = 'K' if color == 'White' else 'k'
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == target:
                    return (i, j)
        return None

    def same_color(self, p1, p2):
        """Return True if p1,p2 are non-space and have same uppercase/lowercase status."""
        if p1 == ' ' or p2 == ' ':
            return False
        return (p1.isupper() and p2.isupper()) or (p1.islower() and p2.islower())

    def make_move(self, start, end):
        """
        Execute the move on this board, no legality check beyond. 
        Return (True, (start,end)) for your code’s convenience.
        """
        sx, sy = start
        ex, ey = end
        piece = self.board[sx][sy]

        # Castling check
        if piece.lower() == 'k' and abs(sy - ey) == 2:
            # kingside or queenside
            if ey > sy:
                # kingside
                self.board[sx][ey] = piece
                self.board[sx][sy] = ' '
                # move rook from h to f
                self.board[sx][ey - 1] = self.board[sx][7]
                self.board[sx][7] = ' '
            else:
                # queenside
                self.board[sx][ey] = piece
                self.board[sx][sy] = ' '
                # move rook from a to d
                self.board[sx][ey + 1] = self.board[sx][0]
                self.board[sx][0] = ' '
            return True, (start, end)

        # En passant check
        if piece.lower() == 'p':
            if abs(sy - ey) == 1 and self.board[ex][ey] == ' ':
                # If it's a diagonal move but the target square is empty,
                # it might be en passant. We check the last move.
                if self.last_move:
                    (lsx, lsy), (lex, ley) = self.last_move
                    moved_piece = self.board[lex][ley]
                    if moved_piece.lower() == 'p' and abs(lsx - lex) == 2 and ley == ey:
                        # remove the enemy pawn
                        self.board[lex][ley] = ' '
        
        # Normal move
        self.board[ex][ey] = piece
        self.board[sx][sy] = ' '
        return True, (start, end)

    def handle_promotion(self, x, y, computer_promotion=False):
        """
        If a pawn reached last rank, handle promotion. 
        """
        piece = self.board[x][y]
        if piece == 'P' and x == 0:
            if computer_promotion:
                self.board[x][y] = 'Q'
            else:
                choice = ''
                while choice not in ['q','r','b','n']:
                    choice = input("Promote to (q/r/b/n): ").lower()
                self.board[x][y] = choice.upper()
        elif piece == 'p' and x == 7:
            if computer_promotion:
                self.board[x][y] = 'q'
            else:
                choice = ''
                while choice not in ['q','r','b','n']:
                    choice = input("Promote to (q/r/b/n): ").lower()
                self.board[x][y] = choice

    def undo_move(self, start, end, captured):
        """
        Undo a move: put the piece back at start, restore captured piece to end.
        If it was castling or en-passant, handle that specifically.
        """
        sx, sy = start
        ex, ey = end
        mover = self.board[ex][ey]
        self.board[sx][sy] = mover
        self.board[ex][ey] = captured

        # If castling
        if mover.lower() == 'k' and abs(sy - ey) == 2:
            if ey > sy:
                # kingside
                self.board[sx][7] = self.board[sx][ey - 1]
                self.board[sx][ey - 1] = ' '
            else:
                # queenside
                self.board[sx][0] = self.board[sx][ey + 1]
                self.board[sx][ey + 1] = ' '
        # If en passant
        if mover.lower() == 'p' and captured == ' ' and abs(sy - ey) == 1:
            # The captured piece must reappear behind the pawn.
            direction = -1 if mover.isupper() else 1
            self.board[ex - direction][ey] = 'p' if mover.islower() else 'P'

    # ------------------------------------------------------------------
    # Single-pass gather_moves
    # ------------------------------------------------------------------
    def gather_moves(self):
        """
        Return a dict: {
          "White": [((sx,sy),(ex,ey)), ...],
          "Black": [((sx,sy),(ex,ey)), ...]
        }
        Each move is checked so it doesn't leave own king in check.
        We handle castling & en passant in a single pass, using
        directions for sliders, etc.
        """
        moves = {"White": [], "Black": []}
        for x in range(8):
            for y in range(8):
                piece = self.board[x][y]
                if piece == ' ':
                    continue
                side = 'White' if piece.isupper() else 'Black'
                possible = self.generate_piece_moves(x, y, piece)
                # For each possible *candidate* move, test if it is legal
                for (ex, ey) in possible:
                    captured = self.board[ex][ey]
                    # Make the move on a copy of the board
                    saved_last_move = self.last_move
                    self.make_move((x,y),(ex,ey))
                    old_last_move = self.last_move
                    self.last_move = ((x,y),(ex,ey))

                    # Check if my own king is in check
                    in_check_dict = self.is_check()  # returns {"White":bool, "Black":bool}
                    # Undo
                    self.undo_move((x,y),(ex,ey), captured)
                    self.last_move = saved_last_move

                    if not in_check_dict[side]:
                        moves[side].append(((x,y),(ex,ey)))
        return moves

    def generate_piece_moves(self, x, y, piece):
        """
        Generate candidate moves for a piece at x,y, ignoring check/pins. 
        We unify bishop/rook/queen, etc. We also handle castling, en passant
        from a single pass, if you want to do so. 
        """
        out = []
        p = piece.lower()

        if p == 'n':  # Knight
            for dx, dy in KNIGHT_MOVES:
                nx, ny = x+dx, y+dy
                if in_bounds(nx, ny) and not self.same_color(piece, self.board[nx][ny]):
                    out.append((nx, ny))

        elif p == 'k':  # King + possible castling
            for dx, dy in KING_MOVES:
                nx, ny = x+dx, y+dy
                if in_bounds(nx, ny) and not self.same_color(piece, self.board[nx][ny]):
                    out.append((nx, ny))
            # Castling (very simplified):
            if piece.isupper() and not self.white_king_moved:
                # Check squares are empty, check rooks haven't moved, etc.
                # For example, short castle:
                if not self.white_rook_moved_h and y < 7:
                    if self.board[x][5] == ' ' and self.board[x][6] == ' ':
                        out.append((x, 6))  # e1 -> g1
                # For example, long castle:
                if not self.white_rook_moved_a and y > 0:
                    if (self.board[x][1] == ' ' and self.board[x][2] == ' ' and self.board[x][3] == ' '):
                        out.append((x, 2))  # e1 -> c1
            elif piece.islower() and not self.black_king_moved:
                # short castle black
                if not self.black_rook_moved_h and y < 7:
                    if self.board[x][5] == ' ' and self.board[x][6] == ' ':
                        out.append((x, 6))
                # long castle black
                if not self.black_rook_moved_a and y > 0:
                    if (self.board[x][1] == ' ' and self.board[x][2] == ' ' and self.board[x][3] == ' '):
                        out.append((x, 2))

        elif p in ('r','b','q'):  # Sliders
            directions = []
            if p == 'r': directions = ROOK_DIRS
            elif p == 'b': directions = BISHOP_DIRS
            else: directions = QUEEN_DIRS

            for (dx, dy) in directions:
                steps = 1
                while True:
                    nx = x + dx*steps
                    ny = y + dy*steps
                    if not in_bounds(nx, ny):
                        break
                    if self.same_color(piece, self.board[nx][ny]):
                        break
                    out.append((nx, ny))
                    # If capture an opponent piece, stop
                    if self.board[nx][ny] != ' ' and not self.same_color(piece, self.board[nx][ny]):
                        break
                    steps += 1

        elif p == 'p':
            # direction
            direction = -1 if piece.isupper() else 1
            start_row = 6 if piece.isupper() else 1

            # forward 1
            nx = x + direction
            if in_bounds(nx, y) and self.board[nx][y] == ' ':
                out.append((nx, y))
                # forward 2 from start row
                if x == start_row:
                    nx2 = x + 2*direction
                    if in_bounds(nx2,y) and self.board[nx2][y] == ' ':
                        out.append((nx2, y))

            # captures
            for sidecap in [-1, 1]:
                ny = y + sidecap
                cx = x + direction
                if in_bounds(cx, ny):
                    if self.board[cx][ny] != ' ' and not self.same_color(piece, self.board[cx][ny]):
                        out.append((cx, ny))
                    else:
                        # en passant
                        if self.last_move:
                            (lsx, lsy),(lex, ley) = self.last_move
                            moved_piece = self.board[lex][ley]
                            # if opponent moved a pawn 2 steps and is next to me
                            if moved_piece.lower() == 'p' and abs(lsx - lex) == 2 and ley == ny and lex == x:
                                # We can capture en passant
                                out.append((cx, ny))

        return out

    # ------------------------------------------------------------------
    # Checking logic
    # ------------------------------------------------------------------
    def is_check(self):
        """
        Return {"White": bool, "Black": bool}
        Tells if White/Black king is in check.
        We do a single pass gather_moves, then see if the king squares
        appear in the opponent’s move list.
        """
        moves_dict = {
            "White": [],
            "Black": []
        }
        # Instead of calling gather_moves once, we can do it:
        #  gather_moves() returns moves for both sides ignoring the king-in-check test from the side's perspective.
        # But we do want to see raw attacking squares. So let's gather "candidate" moves ignoring self-check:
        # Easiest is to do a version of gather_moves that doesn't skip leaving your king in check. 
        # We'll do a simpler approach: For each piece, gather 'attacks' ignoring same-color-check logic.
        # For the sake of clarity, let's re-use the same code but skip the self-check elimination:

        for x in range(8):
            for y in range(8):
                piece = self.board[x][y]
                if piece == ' ':
                    continue
                side = 'White' if piece.isupper() else 'Black'
                # Generate candidate squares ignoring self-check
                cands = self.generate_piece_moves(x, y, piece)
                for (ex, ey) in cands:
                    if not self.same_color(piece, self.board[ex][ey]):
                        moves_dict[side].append(((x,y),(ex,ey)))

        # Now see if the White king pos is in moves_dict["Black"], etc.
        wking = self.find_king("White")
        bking = self.find_king("Black")

        check_result = {"White": False, "Black": False}
        if wking:
            # is wking in black's targets?
            for mv in moves_dict["Black"]:
                if mv[1] == wking:
                    check_result["White"] = True
                    break
        if bking:
            for mv in moves_dict["White"]:
                if mv[1] == bking:
                    check_result["Black"] = True
                    break

        return check_result

    def is_checkmate(self):
        """
        Return {"White": bool, "Black": bool}, stating if each side is in checkmate.
        We consider: 
         - If side is NOT in check, then not checkmate.
         - If side is in check, see if there's ANY legal move in gather_moves() for that side.
        """
        ret = {"White": False, "Black": False}
        checks = self.is_check()
        # If side in check & has no moves => checkmate
        all_moves = self.gather_moves()  # all legal moves for both sides
        for side in ["White", "Black"]:
            if checks[side]:
                if len(all_moves[side]) == 0:
                    ret[side] = True
        return ret

    def is_stalemate(self, color):
        """
        If color is not in check, but has no legal moves, it's a stalemate.
        """
        # Instead of a color-based approach, we can do the dictionary approach:
        checks = self.is_check()
        all_moves = self.gather_moves()
        if not checks[color] and len(all_moves[color]) == 0:
            return True
        return False

    def is_game_over(self):
        cm = self.is_checkmate()
        if cm["White"] or cm["Black"]:
            return True
        # Could also check stalemate for both sides
        if self.is_stalemate("White") or self.is_stalemate("Black"):
            return True
        return False

    def move_piece(self, start, end, last_move):
        """
        For compatibility with your original usage. 
        Calls make_move, updates last_move, sets castling flags, etc.
        """
        sx, sy = start
        piece = self.board[sx][sy]
        # If it's a rook or king, update castling flags
        if piece == 'K':
            self.white_king_moved = True
        elif piece == 'k':
            self.black_king_moved = True
        elif piece == 'R':
            if sy == 0:
                self.white_rook_moved_a = True
            elif sy == 7:
                self.white_rook_moved_h = True
        elif piece == 'r':
            if sy == 0:
                self.black_rook_moved_a = True
            elif sy == 7:
                self.black_rook_moved_h = True

        # Actually do the move
        valid, mv = self.make_move(start, end)
        self.last_move = mv
        return (valid, mv)


# ----------------------------------------------
# The Engine class with Q-learning etc.
# ----------------------------------------------
class Engine:
    def __init__(self):
        self.max_depth = 2
        self.sim_time = 100
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.load_q_table()

    def load_q_table(self):
        try:
            with open("data.dat", "rb") as f:
                print("loading Q-table...")
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

    def save_q_table(self):
        with open("data.dat", "wb") as file:
            pickle.dump(self.q_table, file)

    def get_q_value(self, state, action):
        state_tuple = tuple(tuple(row) for row in state)
        return self.q_table.get((state_tuple, action), 0)

    def update_q_table(self, state, action, reward, next_state):
        state_tuple = tuple(tuple(row) for row in state)
        next_state_tuple = tuple(tuple(row) for row in next_state)
        current_q = self.get_q_value(state, action)
        # We'll get all next moves for both sides
        temp_board = Board(next_state)
        next_moves_white = temp_board.gather_moves()["White"]
        next_moves_black = temp_board.gather_moves()["Black"]
        all_next_moves = next_moves_white + next_moves_black
        next_q_values = [self.get_q_value(next_state, a) for a in all_next_moves]
        max_next_q = max(next_q_values, default=0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state_tuple, action)] = new_q

    def evaluate(self, board_obj):
        """
        Evaluate from White's perspective. 
        """
        board = board_obj.board
        score = 0
        # Just sum piece values minus
        for x in range(8):
            for y in range(8):
                piece = board[x][y]
                if piece == ' ':
                    continue
                val = piece_values.get(piece.upper(), 0)
                if piece.isupper():
                    score += val
                else:
                    score -= val

        # checkmate/stalemate weighting
        if board_obj.is_checkmate()["white"]:
            # White is checkmated => big negative
            score -= 999999
        if board_obj.is_checkmate()["black"]:
            # Black is checkmated => big positive
            score += 999999

        return score

    # a simple alpha-beta
    def negamax(self, board_obj, depth, alpha, beta, color):
        # color is +1 for White, -1 for Black
        if depth == 0 or board_obj.is_game_over():
            return self.evaluate(board_obj)*color, None

        side_str = "White" if color == 1 else "Black"
        moves_dict = board_obj.gather_moves()
        my_moves = moves_dict[side_str]
        if not my_moves:
            # no moves => maybe stalemate or checkmate
            return self.evaluate(board_obj)*color, None

        best_move = None
        for move in my_moves:
            (sx, sy), (ex, ey) = move
            captured = board_obj.board[ex][ey]
            saved_last_move = board_obj.last_move

            board_obj.move_piece((sx,sy),(ex,ey), saved_last_move)
            val, _ = self.negamax(board_obj, depth-1, -beta, -alpha, -color)
            board_obj.undo_move((sx,sy),(ex,ey), captured)
            board_obj.last_move = saved_last_move

            val = -val
            if val > alpha:
                alpha = val
                best_move = move
            if alpha >= beta:
                break
        return alpha, best_move

    def search(self, board_obj, computer_color="Black", depth=None):
        if depth is None:
            depth = self.max_depth
        color = -1 if computer_color == "Black" else 1
        alpha, beta = float('-inf'), float('inf')
        val, best_move = self.negamax(board_obj, depth, alpha, beta, color)
        return best_move, depth, None  # (score, etc.)

'''
import math
import copy
import random
import time
from piece import get_all_moves, move_piece, evaluate, is_game_over

class MCTSNode:
    def __init__(self, board, parent=None, move=None, is_white=True):
        self.board = copy.deepcopy(board)
        self.parent = parent
        self.children = []
        self.move = move
        self.visits = 0
        self.value = 0
        self.is_white = is_white  # Whether this node represents the white player's turn

    def is_fully_expanded(self):
        return len(self.children) > 0 and all(child.visits > 0 for child in self.children)

    def best_child(self, exploration_weight=math.sqrt(2)):
        if not self.children:
            return None
        return max(
            self.children,
            key=lambda child: (child.value / (child.visits + 1e-6)) +
                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )

    def expand(self):
        if self.children:
            return random.choice([child for child in self.children if child.visits == 0])
        moves = get_all_moves(self.board, "White" if self.is_white else "Black")
        for move in moves:
            child_board = copy.deepcopy(self.board)
            valid_move, _ = move_piece(child_board, move[0], move[1], None, sim_move=True)
            if valid_move:
                child_node = MCTSNode(child_board, parent=self, move=move, is_white=not self.is_white)
                self.children.append(child_node)
        return random.choice(self.children) if self.children else None

def mcts_search(root, iterations=1000, exploration_weight=math.sqrt(2)):
    for _ in range(iterations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(exploration_weight)
        if not is_game_over(node.board):
            child = node.expand()
            if child:
                node = child
        result = simulate(node.board, node.is_white)
        backpropagate(node, result)
    return root.best_child(exploration_weight=0).move  # Use exploitation only for the final move

def simulate(board, is_white):
    simulation_board = copy.deepcopy(board)
    simulation_is_white = is_white
    while not is_game_over(simulation_board):
        moves = get_all_moves(simulation_board, "White" if simulation_is_white else "Black")
        if not moves:
            break
        move = random.choice(moves)
        move_piece(simulation_board, move[0], move[1], None, sim_move=True)
        simulation_is_white = not simulation_is_white
    return evaluate(simulation_board)

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        if node.is_white:
            node.value += result
        else:
            node.value -= result
        node = node.parent

def negamax(board, alpha, beta, color, last_move):
    player_color = "White" if 1 else "Black"
    for depth in range(1, 10):
        best_move = None
        best_score = None
        for move in get_all_moves(board, last_move=last_move):
            startpos = move[0]
            endpos = move[1]
            temp_board = copy.deepcopy(board)
            temp_board[endpos[0]][endpos[1]] = board[startpos[0]][startpos[1]]
            temp_board[startpos[0]][startpos[1]] = " "
            score = color * evaluate(temp_board)
            if score >= beta:
                best_score = score
                yield depth, best_score, best_move
                break
            if score > alpha:
                best_move = move
                best_score = score
                alpha = score
        yield depth, best_score, best_move

def search(board, player_color, ttime, max_depth):
    color = -1 if player_color is "Black" else "White"
    for depth, score, move in negamax(board, float("-inf"), float("inf"), color, None):
        if depth >= max_depth or time.time() > ttime:
            break
    return move

def get_q_value(state, action):
    pass

def choose_action(state):
    exploration_rate = 0.1
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(get_all_moves(state, "Black"))
    q_values = {action: get_q_value(state, action) for action in get_all_moves(state)}
    max_q = max(q_values.values(), default=0)
    actions_with_max_q = [action for action, q in q_values.items() if q == max_q]
    return random.choice(actions_with_max_q)

def takespiece(board):
    bestmove = None
    moves = get_all_moves(board, "White", None)
    bestscore = float("-inf")
    for move in moves:
        temp_board = [row[:] for row in board]
        temp_board[move[1][0]][move[1][1]] = temp_board[move[0][0]][move[0][1]]
        temp_board[move[0][0]][move[0][1]] = " "
        score = evaluate(temp_board)
        if score > bestscore:
            bestscore = score
            bestmove = move
    return bestmove

#!/usr/bin/env python3

"""
A Minimalist Chess Engine (Mailbox Representation) with a Basic UCI Loop
Using Standard print() and input() Functions
=======================================================================

HOW TO USE:
- Run this script in a terminal or connect it as a UCI engine in a GUI.
- The engine waits for UCI commands via standard input (input()).
- It responds with standard output (print()).

UCI Commands Supported (minimal subset):
  uci
  isready
  ucinewgame
  position startpos [moves ...]
  go depth N
  stop
  quit

FEATURES:
- Mailbox board representation (120 squares).
- Pseudolegal move generation (no castling, no en passant, no promotion).
- Simple alpha-beta search with material-only evaluation.
- Minimal UCI loop using input() and print().

AUTHOR: OpenAI
DATE: 2024
"""

import math
import copy

# -----------------------------------------------------------------------------
# Mailbox Representation Constants
# -----------------------------------------------------------------------------

# Offsets for knights and kings
KNIGHT_OFFSETS = [-21, -19, -12, -8, 8, 12, 19, 21]
KING_OFFSETS   = [-11, -10, -9, -1, 1, 9, 10, 11]

# Directions for sliding pieces
ROOK_OFFSETS   = [-10, 10, -1, 1]
BISHOP_OFFSETS = [-11, -9, 9, 11]
QUEEN_OFFSETS  = ROOK_OFFSETS + BISHOP_OFFSETS

# Very basic piece values
PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
}

# -----------------------------------------------------------------------------
# Board Setup & Initialization
# -----------------------------------------------------------------------------

def create_mailbox_board():
    """
    Creates a 120-length list representing the board in mailbox format.
    By default, all squares are '.' meaning empty/off-board.
    """
    return ['.' for _ in range(120)]

def index_2d_to_mailbox(row, col):
    """
    Convert a (row, col) in [0..7] to a mailbox index.
    The top-left corner (row=0, col=0) would be 21 (in 2-digit format).
    """
    return (row + 2) * 10 + (col + 1)

def mailbox_to_2d_index(mailbox_idx):
    """
    Convert a mailbox index back to (row, col) in [0..7].
    """
    row = (mailbox_idx // 10) - 2
    col = (mailbox_idx % 10) - 1
    return row, col

def init_standard_setup(board):
    """
    Load the standard chess opening position onto the mailbox board.
    White on ranks 1-2, Black on ranks 7-8.
    """
    # Clear board
    for i in range(120):
        board[i] = '.'

    # White pieces
    board[index_2d_to_mailbox(0, 0)] = 'R'
    board[index_2d_to_mailbox(0, 1)] = 'N'
    board[index_2d_to_mailbox(0, 2)] = 'B'
    board[index_2d_to_mailbox(0, 3)] = 'Q'
    board[index_2d_to_mailbox(0, 4)] = 'K'
    board[index_2d_to_mailbox(0, 5)] = 'B'
    board[index_2d_to_mailbox(0, 6)] = 'N'
    board[index_2d_to_mailbox(0, 7)] = 'R'
    for col in range(8):
        board[index_2d_to_mailbox(1, col)] = 'P'

    # Black pieces
    board[index_2d_to_mailbox(7, 0)] = 'r'
    board[index_2d_to_mailbox(7, 1)] = 'n'
    board[index_2d_to_mailbox(7, 2)] = 'b'
    board[index_2d_to_mailbox(7, 3)] = 'q'
    board[index_2d_to_mailbox(7, 4)] = 'k'
    board[index_2d_to_mailbox(7, 5)] = 'b'
    board[index_2d_to_mailbox(7, 6)] = 'n'
    board[index_2d_to_mailbox(7, 7)] = 'r'
    for col in range(8):
        board[index_2d_to_mailbox(6, col)] = 'p'

# -----------------------------------------------------------------------------
# Board Printing Utility (for debugging)
# -----------------------------------------------------------------------------

def print_board(board):
    """
    Print the 8x8 portion of the mailbox board.
    """
    print("   +-----------------+")
    for row in range(8):
        print(f" {8 - row} |", end=" ")
        for col in range(8):
            idx = index_2d_to_mailbox(row, col)
            piece = board[idx]
            print(piece if piece != '.' else '.', end=" ")
        print("|")
    print("   +-----------------+")
    print("     a b c d e f g h ")

# -----------------------------------------------------------------------------
# Move Generation
# -----------------------------------------------------------------------------

def is_square_off_board(idx):
    """
    Check if a mailbox index is outside the real 8x8 portion.
    """
    # Valid squares range from 21..28, 31..38, 41..48, ..., 91..98.
    if (idx < 21 or idx > 98):
        return True
    row = idx // 10
    col = idx % 10
    if col < 1 or col > 8 or row < 2 or row > 9:
        return True
    return False

def generate_moves(board, side_to_move):
    """
    Generate all pseudo-legal moves (start, end) for side_to_move ('w' or 'b').
    """
    moves = []
    for sq in range(120):
        piece = board[sq]
        if piece == '.':
            continue
        if side_to_move == 'w' and piece.isupper():
            moves.extend(generate_piece_moves(board, sq, piece, 'w'))
        elif side_to_move == 'b' and piece.islower():
            moves.extend(generate_piece_moves(board, sq, piece, 'b'))
    return moves

def generate_piece_moves(board, sq, piece, side):
    """
    Given a piece and side, dispatch to the correct move generator.
    """
    moves = []
    if piece.upper() == 'P':
        moves.extend(generate_pawn_moves(board, sq, piece, side))
    elif piece.upper() == 'N':
        moves.extend(generate_knight_moves(board, sq, side))
    elif piece.upper() == 'B':
        moves.extend(generate_sliding_moves(board, sq, BISHOP_OFFSETS, side))
    elif piece.upper() == 'R':
        moves.extend(generate_sliding_moves(board, sq, ROOK_OFFSETS, side))
    elif piece.upper() == 'Q':
        moves.extend(generate_sliding_moves(board, sq, QUEEN_OFFSETS, side))
    elif piece.upper() == 'K':
        moves.extend(generate_king_moves(board, sq, side))
    return moves

def generate_pawn_moves(board, sq, piece, side):
    """
    Generate pawn moves (no promotions, no en passant).
    White pawns move with direction -10, black pawns with +10.
    """
    moves = []
    direction = -10 if side == 'w' else 10

    # One step forward
    fwd_sq = sq + direction
    if not is_square_off_board(fwd_sq) and board[fwd_sq] == '.':
        moves.append((sq, fwd_sq))
        # Two steps forward if on starting rank
        # White pawns start on row=1 in 0-based, which is mailbox row=3
        # Black pawns start on row=6, mailbox row=8
        start_row = (sq // 10) - 2
        if side == 'w' and start_row == 1:
            two_sq = sq + 2*direction
            if board[two_sq] == '.':
                moves.append((sq, two_sq))
        if side == 'b' and start_row == 6:
            two_sq = sq + 2*direction
            if board[two_sq] == '.':
                moves.append((sq, two_sq))

    # Captures
    for cap_offset in [-1, 1]:
        cap_sq = sq + direction + cap_offset
        if not is_square_off_board(cap_sq):
            # If there's an opponent piece
            if side == 'w' and board[cap_sq].islower():
                moves.append((sq, cap_sq))
            if side == 'b' and board[cap_sq].isupper():
                moves.append((sq, cap_sq))

    return moves

def generate_knight_moves(board, sq, side):
    moves = []
    for offset in KNIGHT_OFFSETS:
        to_sq = sq + offset
        if is_square_off_board(to_sq):
            continue
        if board[to_sq] == '.' or (side == 'w' and board[to_sq].islower()) or (side == 'b' and board[to_sq].isupper()):
            moves.append((sq, to_sq))
    return moves

def generate_king_moves(board, sq, side):
    moves = []
    for offset in KING_OFFSETS:
        to_sq = sq + offset
        if is_square_off_board(to_sq):
            continue
        if board[to_sq] == '.' or (side == 'w' and board[to_sq].islower()) or (side == 'b' and board[to_sq].isupper()):
            moves.append((sq, to_sq))
    return moves

def generate_sliding_moves(board, sq, offsets, side):
    """
    For rooks, bishops, queens. Slide in each direction until blocked or off-board.
    """
    moves = []
    for offset in offsets:
        current_sq = sq
        while True:
            current_sq += offset
            if is_square_off_board(current_sq):
                break
            if board[current_sq] == '.':
                moves.append((sq, current_sq))
            elif (side == 'w' and board[current_sq].islower()) or (side == 'b' and board[current_sq].isupper()):
                moves.append((sq, current_sq))
                break
            else:
                break
    return moves

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_position(board):
    """
    Extremely basic evaluation: sum material values.
    """
    score = 0
    for sq in range(120):
        piece = board[sq]
        if piece in PIECE_VALUES:
            score += PIECE_VALUES[piece]
    return score

# -----------------------------------------------------------------------------
# Make/Unmake Moves
# -----------------------------------------------------------------------------

def make_move(board, move):
    """
    Returns a new board (copy) with the move applied.
    Move is (start_idx, end_idx).
    """
    new_board = copy.deepcopy(board)
    start, end = move
    new_board[end] = new_board[start]
    new_board[start] = '.'
    return new_board

# -----------------------------------------------------------------------------
# Alpha-Beta Search
# -----------------------------------------------------------------------------

def alpha_beta_search(board, depth, alpha, beta, side_to_move):
    """
    Simple alpha-beta search. Returns (score, best_move).
    """
    if depth == 0:
        return evaluate_position(board), None

    moves = generate_moves(board, side_to_move)
    if not moves:
        # No moves -> checkmate or stalemate
        return evaluate_position(board), None

    best_move = None
    if side_to_move == 'w':
        max_eval = -math.inf
        for mv in moves:
            new_brd = make_move(board, mv)
            eval_, _ = alpha_beta_search(new_brd, depth - 1, alpha, beta, 'b')
            if eval_ > max_eval:
                max_eval = eval_
                best_move = mv
            alpha = max(alpha, eval_)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for mv in moves:
            new_brd = make_move(board, mv)
            eval_, _ = alpha_beta_search(new_brd, depth - 1, alpha, beta, 'w')
            if eval_ < min_eval:
                min_eval = eval_
                best_move = mv
            beta = min(beta, eval_)
            if beta <= alpha:
                break
        return min_eval, best_move

# -----------------------------------------------------------------------------
# UCI Helpers
# -----------------------------------------------------------------------------

def square_to_mailbox(sq_str):
    """
    Convert algebraic notation (e.g., 'e2') to mailbox index.
    'a1' -> row=7, col=0 in 0-based, which is mailbox (7+2)*10 + (0+1)= 90+1=91.
    """
    col = ord(sq_str[0]) - ord('a')
    rank = int(sq_str[1])
    row = 8 - rank  # rank=1 => row=7, rank=8 => row=0
    return index_2d_to_mailbox(row, col)

def mailbox_move_to_uci(move):
    """
    Convert a mailbox (start, end) move into algebraic notation like 'e2e4'.
    """
    start, end = move
    rowS, colS = mailbox_to_2d_index(start)
    rowE, colE = mailbox_to_2d_index(end)
    # row=7 -> rank='1', row=0 -> rank='8'
    rankS = str(8 - rowS)
    rankE = str(8 - rowE)
    fileS = chr(ord('a') + colS)
    fileE = chr(ord('a') + colE)
    return f"{fileS}{rankS}{fileE}{rankE}"

def apply_uci_move(board, move_str):
    """
    Parse and apply a UCI move string (e.g. 'e2e4') to the board.
    Returns the updated board.
    """
    start_sq = square_to_mailbox(move_str[:2])
    end_sq = square_to_mailbox(move_str[2:4])
    return make_move(board, (start_sq, end_sq))

# -----------------------------------------------------------------------------
# UCI Main Loop (Using input() & print())
# -----------------------------------------------------------------------------

def uci_main_loop():
    """
    Minimal UCI protocol handling with standard input()/print() calls.
    """
    board = create_mailbox_board()
    init_standard_setup(board)

    # Track how many half-moves have been made (to determine side-to-move).
    # If halfmove_count is even, it's White to move; if odd, Black to move.
    halfmove_count = 0

    engine_name = "MailboxEngine"
    engine_author = "OpenAI"
    search_depth = 3  # default search depth

    while True:
        try:
            line = input()
        except EOFError:
            break  # no more input

        if not line:
            continue

        line = line.strip()

        if line == "uci":
            print(f"id name {engine_name}")
            print(f"id author {engine_author}")
            print("uciok")
            # Flush ensures the GUI sees the output promptly
            continue

        elif line == "isready":
            print("readyok")
            continue

        elif line.startswith("setoption"):
            # Ignored in this minimal version
            continue

        elif line == "ucinewgame":
            # Reset the board to a new game
            board = create_mailbox_board()
            init_standard_setup(board)
            halfmove_count = 0
            continue

        elif line.startswith("position"):
            # Format:
            # position startpos [moves e2e4 e7e5 ...]
            # or
            # position fen <FEN string> [moves ...] (not implemented here)
            tokens = line.split()
            if "startpos" in tokens:
                # Start position
                board = create_mailbox_board()
                init_standard_setup(board)
                halfmove_count = 0
                if "moves" in tokens:
                    idx_moves = tokens.index("moves")
                    move_list = tokens[idx_moves + 1:]
                    for mv in move_list:
                        board = apply_uci_move(board, mv)
                        halfmove_count += 1
            continue

        elif line.startswith("go"):
            # e.g.: go depth 5
            # We'll look for "depth" argument in the tokens.
            tokens = line.split()
            depth = search_depth
            if "depth" in tokens:
                idx = tokens.index("depth") + 1
                if idx < len(tokens):
                    try:
                        depth = int(tokens[idx])
                    except ValueError:
                        depth = search_depth

            side_to_move = 'w' if (halfmove_count % 2) == 0 else 'b'
            score, best_move = alpha_beta_search(board, depth, -math.inf, math.inf, side_to_move)

            if best_move is not None:
                best_move_uci = mailbox_move_to_uci(best_move)
            else:
                best_move_uci = None

            if best_move_uci:
                print(f"bestmove {best_move_uci}")
            else:
                print("bestmove 0000")
            continue

        elif line == "stop":
            # For "stop", we do nothing extra here
            continue

        elif line == "quit":
            break

        # Ignore other unrecognized commands in this minimal version

def main():
    uci_main_loop()

if __name__ == "__main__":
    main()


# ======================
# then bit boards
# ======================
#!/usr/bin/env python3
"""
Minimal UCI Chess Engine in Python (Bitboard-Based), using print() and input().

Features:
  - UCI protocol: "uci", "isready", "ucinewgame", "position startpos moves ...", "go", "stop", "quit"
  - Bitboard representation for each piece type
  - Very basic alpha-beta search (material-only)
  - Minimal move generation (missing castling, en passant, etc.)
  - Start-position setup and incremental move parsing
  - Uses standard Python input/print for I/O.

"""

import math

# -----------------------------------------------------------------------
#  GLOBAL CONSTANTS AND BITBOARD HELPERS
# -----------------------------------------------------------------------

KNIGHT_OFFSETS = [17, 15, 10, 6, -17, -15, -10, -6]
KING_OFFSETS = [8, -8, 1, -1, 7, -7, 9, -9]

ROOK_DIRECTIONS = [8, -8, 1, -1]
BISHOP_DIRECTIONS = [9, -9, 7, -7]

PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
}

def get_bit(bitboard: int, square: int) -> int:
    return (bitboard >> square) & 1

def set_bit(bitboard: int, square: int) -> int:
    return bitboard | (1 << square)

def clear_bit(bitboard: int, square: int) -> int:
    return bitboard & ~(1 << square)

def popcount_64(x: int) -> int:
    return bin(x).count('1')

def bits_in_mask(mask: int):
    """Return a list of set-bit positions (0..63) in 'mask'."""
    res = []
    while mask:
        lsb = (mask & -mask)
        sq = lsb.bit_length() - 1
        res.append(sq)
        mask = mask & (mask - 1)
    return res

def is_on_board(from_sq: int, to_sq: int, offset: int) -> bool:
    """Naive boundary check to ensure we don't wrap around the board incorrectly."""
    if to_sq < 0 or to_sq > 63:
        return False
    from_file = from_sq % 8
    to_file = to_sq % 8
    # If offset is diagonal or horizontal, ensure we haven't jumped across the board
    if abs(offset) in [1, 7, 9]:
        if abs(from_file - to_file) > 2:  
            return False
    return True

# -----------------------------------------------------------------------
#  BOARD STATE
# -----------------------------------------------------------------------

class BoardState:
    """
    Each piece type has its own 64-bit bitboard.
    White: 'P','N','B','R','Q','K'
    Black: 'p','n','b','r','q','k'
    side_to_move = 'w' or 'b'
    """
    def __init__(self):
        self.bitboards = {
            'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
            'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0
        }
        self.side_to_move = 'w'

    def occupancy(self) -> int:
        occ = 0
        for bb in self.bitboards.values():
            occ |= bb
        return occ

    def white_occupancy(self) -> int:
        return (self.bitboards['P'] | self.bitboards['N'] | self.bitboards['B'] |
                self.bitboards['R'] | self.bitboards['Q'] | self.bitboards['K'])

    def black_occupancy(self) -> int:
        return (self.bitboards['p'] | self.bitboards['n'] | self.bitboards['b'] |
                self.bitboards['r'] | self.bitboards['q'] | self.bitboards['k'])

    def set_start_position(self):
        """Set the standard chess starting position (for new games)."""
        # Clear existing
        for p in self.bitboards:
            self.bitboards[p] = 0
        # White
        for sq in range(8, 16):
            self.bitboards['P'] = set_bit(self.bitboards['P'], sq)
        self.bitboards['R'] = set_bit(self.bitboards['R'], 0)
        self.bitboards['R'] = set_bit(self.bitboards['R'], 7)
        self.bitboards['N'] = set_bit(self.bitboards['N'], 1)
        self.bitboards['N'] = set_bit(self.bitboards['N'], 6)
        self.bitboards['B'] = set_bit(self.bitboards['B'], 2)
        self.bitboards['B'] = set_bit(self.bitboards['B'], 5)
        self.bitboards['Q'] = set_bit(self.bitboards['Q'], 3)
        self.bitboards['K'] = set_bit(self.bitboards['K'], 4)
        # Black
        for sq in range(48, 56):
            self.bitboards['p'] = set_bit(self.bitboards['p'], sq)
        self.bitboards['r'] = set_bit(self.bitboards['r'], 56)
        self.bitboards['r'] = set_bit(self.bitboards['r'], 63)
        self.bitboards['n'] = set_bit(self.bitboards['n'], 57)
        self.bitboards['n'] = set_bit(self.bitboards['n'], 62)
        self.bitboards['b'] = set_bit(self.bitboards['b'], 58)
        self.bitboards['b'] = set_bit(self.bitboards['b'], 61)
        self.bitboards['q'] = set_bit(self.bitboards['q'], 59)
        self.bitboards['k'] = set_bit(self.bitboards['k'], 60)

        self.side_to_move = 'w'

# -----------------------------------------------------------------------
#  MOVE REPRESENTATION
# -----------------------------------------------------------------------

class Move:
    def __init__(self, from_sq, to_sq, captured=None, promoted=None):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.captured = captured
        self.promoted = promoted

    def __repr__(self):
        return f"Move({self.from_sq}->{self.to_sq}, cap={self.captured}, prom={self.promoted})"

# -----------------------------------------------------------------------
#  MOVE GENERATION
# -----------------------------------------------------------------------

def generate_all_moves(board: BoardState):
    """Generate all pseudo-legal moves for board.side_to_move."""
    moves = []
    side = board.side_to_move
    if side == 'w':
        generate_pawn_moves(board, moves, is_white=True)
        generate_knight_moves(board, moves, 'N')
        generate_sliding_moves(board, moves, ['B','R','Q'])
        generate_king_moves(board, moves, 'K')
    else:
        generate_pawn_moves(board, moves, is_white=False)
        generate_knight_moves(board, moves, 'n')
        generate_sliding_moves(board, moves, ['b','r','q'])
        generate_king_moves(board, moves, 'k')
    return moves

def generate_pawn_moves(board: BoardState, moves, is_white=True):
    if is_white:
        WP = board.bitboards['P']
        opp_occ = board.black_occupancy()
        empty_squares = ~board.occupancy() & ((1 << 64) - 1)

        # Single push
        single_push = (WP << 8) & empty_squares
        for to_sq in bits_in_mask(single_push):
            from_sq = to_sq - 8
            moves.append(Move(from_sq, to_sq))
        # Double push (from rank 2)
        rank_2_mask = 0x000000000000FF00
        double_push = ((WP & rank_2_mask) << 16) & empty_squares << 8
        for to_sq in bits_in_mask(double_push):
            from_sq = to_sq - 16
            moves.append(Move(from_sq, to_sq))

        # Captures
        left_capture = (WP & ~0x8080808080808080) << 7
        left_capture &= opp_occ
        for to_sq in bits_in_mask(left_capture):
            from_sq = to_sq - 7
            cap = find_captured_piece(board, to_sq)
            moves.append(Move(from_sq, to_sq, cap))

        right_capture = (WP & ~0x0101010101010101) << 9
        right_capture &= opp_occ
        for to_sq in bits_in_mask(right_capture):
            from_sq = to_sq - 9
            cap = find_captured_piece(board, to_sq)
            moves.append(Move(from_sq, to_sq, cap))

    else:
        BP = board.bitboards['p']
        opp_occ = board.white_occupancy()
        empty_squares = ~board.occupancy() & ((1 << 64) - 1)

        # Single push
        single_push = (BP >> 8) & empty_squares
        for to_sq in bits_in_mask(single_push):
            from_sq = to_sq + 8
            moves.append(Move(from_sq, to_sq))

        # Double push (from rank 7)
        rank_7_mask = 0x00FF000000000000
        double_push = ((BP & rank_7_mask) >> 16) & empty_squares >> 8
        for to_sq in bits_in_mask(double_push):
            from_sq = to_sq + 16
            moves.append(Move(from_sq, to_sq))

        # Captures
        left_capture = (BP & ~0x0101010101010101) >> 9
        left_capture &= opp_occ
        for to_sq in bits_in_mask(left_capture):
            from_sq = to_sq + 9
            cap = find_captured_piece(board, to_sq)
            moves.append(Move(from_sq, to_sq, cap))

        right_capture = (BP & ~0x8080808080808080) >> 7
        right_capture &= opp_occ
        for to_sq in bits_in_mask(right_capture):
            from_sq = to_sq + 7
            cap = find_captured_piece(board, to_sq)
            moves.append(Move(from_sq, to_sq, cap))

def generate_knight_moves(board: BoardState, moves, knight_char):
    knights = board.bitboards[knight_char]
    if knight_char.isupper():
        own_occ = board.white_occupancy()
        opp_occ = board.black_occupancy()
    else:
        own_occ = board.black_occupancy()
        opp_occ = board.white_occupancy()

    for from_sq in bits_in_mask(knights):
        for offset in KNIGHT_OFFSETS:
            to_sq = from_sq + offset
            if is_on_board(from_sq, to_sq, offset):
                if not get_bit(own_occ, to_sq):
                    captured = None
                    if get_bit(opp_occ, to_sq):
                        captured = find_captured_piece(board, to_sq)
                    moves.append(Move(from_sq, to_sq, captured))

def generate_king_moves(board: BoardState, moves, king_char):
    king_bb = board.bitboards[king_char]
    if king_char.isupper():
        own_occ = board.white_occupancy()
        opp_occ = board.black_occupancy()
    else:
        own_occ = board.black_occupancy()
        opp_occ = board.white_occupancy()

    king_list = bits_in_mask(king_bb)
    if not king_list:
        return
    from_sq = king_list[0]

    for offset in KING_OFFSETS:
        to_sq = from_sq + offset
        if is_on_board(from_sq, to_sq, offset):
            if not get_bit(own_occ, to_sq):
                captured = None
                if get_bit(opp_occ, to_sq):
                    captured = find_captured_piece(board, to_sq)
                moves.append(Move(from_sq, to_sq, captured))

def generate_sliding_moves(board: BoardState, moves, piece_list):
    """Generate moves for rooks, bishops, queens (sliding)."""
    occupancy_all = board.occupancy()
    if board.side_to_move == 'w':
        own_occ = board.white_occupancy()
        opp_occ = board.black_occupancy()
    else:
        own_occ = board.black_occupancy()
        opp_occ = board.white_occupancy()

    for piece_char in piece_list:
        pieces_bb = board.bitboards[piece_char]
        directions = []
        if piece_char.lower() == 'r':
            directions = ROOK_DIRECTIONS
        elif piece_char.lower() == 'b':
            directions = BISHOP_DIRECTIONS
        elif piece_char.lower() == 'q':
            directions = ROOK_DIRECTIONS + BISHOP_DIRECTIONS

        for from_sq in bits_in_mask(pieces_bb):
            for d in directions:
                to_sq = from_sq
                while True:
                    to_sq += d
                    if not is_on_board(from_sq, to_sq, d):
                        break
                    if get_bit(own_occ, to_sq):
                        # blocked by own piece
                        break
                    captured = None
                    if get_bit(opp_occ, to_sq):
                        captured = find_captured_piece(board, to_sq)
                        moves.append(Move(from_sq, to_sq, captured))
                        break
                    else:
                        moves.append(Move(from_sq, to_sq))
                    # if we hit any piece, stop
                    if get_bit(occupancy_all, to_sq):
                        break

def find_captured_piece(board: BoardState, sq: int) -> str:
    for pc, bb in board.bitboards.items():
        if get_bit(bb, sq):
            return pc
    return None

# -----------------------------------------------------------------------
#  MAKE / UNDO MOVE
# -----------------------------------------------------------------------

def make_move(board: BoardState, move: Move):
    """Apply the move to board (pseudo-legal). Returns True if success."""
    piece_moved = None
    side = board.side_to_move
    if side == 'w':
        piece_list = ['P','N','B','R','Q','K']
    else:
        piece_list = ['p','n','b','r','q','k']

    # 1) Identify which piece is moving
    for pc in piece_list:
        if get_bit(board.bitboards[pc], move.from_sq):
            piece_moved = pc
            board.bitboards[pc] = clear_bit(board.bitboards[pc], move.from_sq)
            break
    if piece_moved is None:
        return False

    # 2) Remove captured piece
    if move.captured:
        board.bitboards[move.captured] = clear_bit(board.bitboards[move.captured], move.to_sq)

    # 3) Place the moved piece
    if move.promoted is not None:
        board.bitboards[move.promoted] = set_bit(board.bitboards[move.promoted], move.to_sq)
    else:
        board.bitboards[piece_moved] = set_bit(board.bitboards[piece_moved], move.to_sq)

    # 4) Switch side
    board.side_to_move = 'b' if side == 'w' else 'w'
    return True

def backup_state(board: BoardState):
    return {
        'bitboards': {pc: board.bitboards[pc] for pc in board.bitboards},
        'side_to_move': board.side_to_move
    }

def restore_state(board: BoardState, backup):
    for pc in backup['bitboards']:
        board.bitboards[pc] = backup['bitboards'][pc]
    board.side_to_move = backup['side_to_move']

# -----------------------------------------------------------------------
#  EVALUATION
# -----------------------------------------------------------------------

def evaluate(board: BoardState):
    """Material-only evaluation."""
    score = 0
    for pc, bb in board.bitboards.items():
        val = PIECE_VALUES.get(pc, 0)
        cnt = popcount_64(bb)
        score += val * cnt
    return score

# -----------------------------------------------------------------------
#  ALPHA-BETA
# -----------------------------------------------------------------------

def alpha_beta_search(board: BoardState, depth: int, alpha: int, beta: int):
    if depth == 0:
        return evaluate(board)

    moves = generate_all_moves(board)
    if not moves:
        # No moves -> could be checkmate or stalemate. Very naive approach:
        return -99999 if is_in_check(board) else 0

    side = board.side_to_move
    if side == 'w':
        value = -math.inf
        for mv in moves:
            bkp = backup_state(board)
            ok = make_move(board, mv)
            if not ok:
                restore_state(board, bkp)
                continue
            value = max(value, alpha_beta_search(board, depth - 1, alpha, beta))
            restore_state(board, bkp)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for mv in moves:
            bkp = backup_state(board)
            ok = make_move(board, mv)
            if not ok:
                restore_state(board, bkp)
                continue
            value = min(value, alpha_beta_search(board, depth - 1, alpha, beta))
            restore_state(board, bkp)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def is_in_check(board: BoardState) -> bool:
    """Extremely naive check detection:
    - Switch side_to_move (opponent's moves)
    - Generate all moves
    - If any move captures the king, it's check
    """
    bkp_side = board.side_to_move
    board.side_to_move = 'b' if bkp_side == 'w' else 'w'
    moves = generate_all_moves(board)
    board.side_to_move = bkp_side

    king_char = 'K' if bkp_side == 'w' else 'k'
    king_sq_list = bits_in_mask(board.bitboards[king_char])
    if not king_sq_list:
        return False
    king_sq = king_sq_list[0]

    for mv in moves:
        if mv.to_sq == king_sq:
            return True
    return False

# -----------------------------------------------------------------------
#  UCI PROTOCOL SUPPORT
# -----------------------------------------------------------------------

def square_to_algebraic(sq: int) -> str:
    """Convert 0..63 to algebraic notation (e.g. 0->a1, 7->h1, 63->h8)."""
    file = sq % 8
    rank = sq // 8
    return chr(ord('a') + file) + str(rank + 1)

def algebraic_to_square(algeb: str) -> int:
    """
    Convert algebraic like 'a1','h8' to 0..63. 
    'a1' -> 0, 'b1' -> 1, ..., 'a2' -> 8, etc.
    """
    file = ord(algeb[0]) - ord('a')
    rank = int(algeb[1]) - 1
    return rank * 8 + file

def move_to_uci(move: Move) -> str:
    """Convert our Move into UCI string, e.g. e2e4, e7e5, etc."""
    return square_to_algebraic(move.from_sq) + square_to_algebraic(move.to_sq)

def parse_position_cmd(board: BoardState, tokens: list):
    """
    position [startpos | fen <fenstring>] [moves ...]
    For simplicity, this code only properly handles "startpos" + moves.
    """
    idx = 0
    if tokens[idx] == "startpos":
        board.set_start_position()
        idx += 1
    elif tokens[idx] == "fen":
        # This engine does not fully support FEN in this minimal example
        # We'll just do an empty board for demonstration.
        for p in board.bitboards:
            board.bitboards[p] = 0
        board.side_to_move = 'w'
        # skip fen tokens until we see "moves" or end
        while idx < len(tokens) and tokens[idx] != "moves":
            idx += 1
    else:
        # Unrecognized, ignore
        return

    # If there's a 'moves' part, parse them
    if idx < len(tokens) and tokens[idx] == "moves":
        idx += 1
        while idx < len(tokens):
            move_str = tokens[idx]
            if len(move_str) < 4:
                idx += 1
                continue
            from_sq = algebraic_to_square(move_str[:2])
            to_sq = algebraic_to_square(move_str[2:4])
            mv = None
            # We do a quick search in pseudo-legal moves to find a matching move
            moves = generate_all_moves(board)
            for candidate in moves:
                if candidate.from_sq == from_sq and candidate.to_sq == to_sq:
                    mv = candidate
                    # For promotions: UCI might have e7e8q
                    if len(move_str) == 5:
                        mv.promoted = move_str[4].upper() if board.side_to_move == 'w' else move_str[4].lower()
                    break
            if mv:
                make_move(board, mv)
            idx += 1

def parse_go_cmd(board: BoardState, tokens: list):
    """
    go [searchmoves ... ] [wtime ...] [btime ...] [winc ...] [binc ...] ...
    We'll ignore time mgmt for now and just search a fixed depth.
    """
    depth = 3
    # You could parse times if you want, but let's skip for minimal example.
    best_move = search_best_move(board, depth)
    if best_move is not None:
        print(f"bestmove {move_to_uci(best_move)}")
    else:
        # No moves available
        print("bestmove 0000")

def search_best_move(board: BoardState, depth: int):
    """Return the best move via alpha-beta at given depth."""
    moves = generate_all_moves(board)
    if not moves:
        return None
    side = board.side_to_move
    best_move = None
    if side == 'w':
        best_eval = -math.inf
        for mv in moves:
            bkp = backup_state(board)
            ok = make_move(board, mv)
            if not ok:
                restore_state(board, bkp)
                continue
            val = alpha_beta_search(board, depth - 1, -math.inf, math.inf)
            restore_state(board, bkp)
            if val > best_eval:
                best_eval = val
                best_move = mv
    else:
        best_eval = math.inf
        for mv in moves:
            bkp = backup_state(board)
            ok = make_move(board, mv)
            if not ok:
                restore_state(board, bkp)
                continue
            val = alpha_beta_search(board, depth - 1, -math.inf, math.inf)
            restore_state(board, bkp)
            if val < best_eval:
                best_eval = val
                best_move = mv

    return best_move

# -----------------------------------------------------------------------
#  MAIN UCI LOOP (Using input() and print())
# -----------------------------------------------------------------------

def uci_loop():
    board = BoardState()
    board.set_start_position()

    # Announce engine's presence
    print("id name MinimalBitboardEngine")
    print("id author OpenAI")
    print("uciok")

    while True:
        line = ""
        try:
            line = input()
        except EOFError:
            break
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            print("id name MinimalBitboardEngine")
            print("id author OpenAI")
            print("uciok")

        elif cmd == "isready":
            print("readyok")

        elif cmd == "ucinewgame":
            board.set_start_position()

        elif cmd == "position":
            parse_position_cmd(board, tokens[1:])

        elif cmd == "go":
            parse_go_cmd(board, tokens[1:])

        elif cmd == "stop":
            # If we were doing a background search, we'd stop it.
            # Here, our search is synchronous, so we do nothing.
            pass

        elif cmd == "quit":
            break

        else:
            # Unknown command - we can safely ignore or print debug info
            pass

def main():
    uci_loop()

if __name__ == "__main__":
    main()

# v2024(1.0)
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
unicode_pieces = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    ' ': ' '
}
# idea:
# Is an added up some of ascii binary values of rnbkq
# as pieces and 0's as empty spaces.
# bin_startpos = 2496
initial_board = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"],
]
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
def print_board(board, is_white=True):
    if is_white:
        print("  a b c d e f g h")
    else:
        print("  h g f e d c b a")
    for row in range(8):
        print(8 - row, end=" ")
        for col in range(8):
            print(unicode_pieces[board[row][col]], end=" ")
        print(8 - row)
    if is_white:
        print("  a b c d e f g h")
    else:
        print("  h g f e d c b a")
    return board
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
    next_q_values = [get_q_value(next_state, a) for a in get_all_moves(next_state)]
    max_next_q = max(next_q_values, default=0)
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[(state_tuple, action)] = new_q
def get_t_key(board, t_table):
    if len(t_table) == 0:
        return 1000000000000000
    # copy just the keys
    keys = [_ for _ in t_table]
    keys.sort()
    for k, p in t_table.items():
        if p[0] == board:
            return k
    return keys[0] - 1
def negamax(board, depth, alpha, beta, color, last_move, t_table):
    if depth == 0 or is_game_over(board):
        #start = time.time()
        values = quiescence_search(board, alpha, beta, color, last_move, t_table)
        #print("Quiesce time:", time.time()-start)
        return values
    best_move = None
    total_nodes = 0
    alpha_original = alpha
    player_color = 'White' if color == 1 else 'Black'
    if depth >= 3 and not is_check(board)[player_color.lower()] and not is_endgame(board):
        #start = time.time()
        score, best_move, nodes = negamax(board, depth - 3, -beta, -beta + 1, -color, last_move, t_table)
        #print("No-branches search:", time.time()-start)
        total_nodes += nodes
        if score >= beta:
            return beta, best_move, total_nodes
    moves = get_all_moves(board, player_color, last_move)
    moves = order_moves(board, moves, player_color, last_move, depth)
    first_move = True
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
        is_in_check = is_check(temp_board)[player_color.lower()]
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
    key = get_t_key(board, t_table)
    if key in t_table:
        stand_pat = color * t_table[key][1]
    else:
        stand_pat = evaluate(board)
        t_table[key] = (board, stand_pat)
        stand_pat *= color
    if depth >= max_q_depth or is_game_over(board):
        return stand_pat, None, 1
    if stand_pat >= beta:
        return beta, None, 1
    if alpha < stand_pat:
        alpha = stand_pat
    total_nodes = 1
    player_color = 'White' if color == 1 else 'Black'
    if not is_check(board)[player_color.lower()] and not is_endgame(board):
        score = stand_pat
        if score >= beta:
            return beta, None, None
    allmoves = get_all_moves(board, player_color, last_move)
    moves = []
    for move in allmoves:
        end_pos = board[move[1][0]][move[1][1]]
        if end_pos != " ":
            moves.append(move)
    moves = order_moves(board, moves, player_color, last_move, depth)
    best_move = None
    for move in moves:
        temp_board = copy.deepcopy(board)
        temp_board[move[1][0]][move[1][1]] = temp_board[move[0][0]][move[0][1]]
        temp_board[move[0][0]][move[0][1]] = " "
        previous_last_move = last_move
        last_move = move
        score, _, nodes = quiescence_search(temp_board, -beta, -alpha, -color, last_move, t_table, depth + 1, max_q_depth)
        total_nodes += nodes
        last_move = previous_last_move
        score = -score
        if score >= beta:
            return beta, move, total_nodes
        if score > alpha:
            alpha = score
            best_move = move
    return alpha, best_move, total_nodes
def search(board, t_table, computer_color="Black", depth=max_depth, last_move=None):
    best_move = None
    color = -1 if computer_color == "Black" else 1
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
    opponent_color = 'White' if player_color == 'Black' else 'Black'
    move_scores = []
    for move in moves:
        score = 0
        temp_board = copy.deepcopy(board)
        piece = board[move[0][0]][move[0][1]]
        piece_val = piece_values[piece.upper()]
        end_pos = (move[1][0], move[1][1])
        start_pos = (move[0][0], move[0][1])
        temp_board[end_pos[0]][end_pos[1]] = piece
        temp_board[start_pos[0]][start_pos[1]] = " "
        if player_color == "Black":
            score += -evaluate(temp_board)
        else:
            score += evaluate(temp_board)
        player_squares = get_all_moves(board, player_color, last_move)
        opponent_squares = get_all_moves(board, opponent_color, last_move)
        if end_pos in opponent_squares and end_pos not in player_squares:
            score -= math.factorial(piece_val) + 10000
        q_value = get_q_value(board, move)
        score += q_value
        move_scores.append((score, move))
    move_scores.sort(reverse=True, key=lambda x: x[0])
    ordered_moves = [move for score, move in move_scores]
    return ordered_moves
def play_chess():
    board = copy.deepcopy(initial_board)
    player_color = 'White'
    last_move = None
    t_table = {}
    mode = input("Do you want to play against a person or the computer? ").strip().lower()
    if mode in ["person", "play against a person"]:
        while True:
            print_board(board)
            move = input(f"{player_color}'s turn. Enter your move (e.g., e2 e4): ").strip()
            if move.lower() == 'quit':
                save_q_table(q_table)
                return
            parsed_move = parse_move(move)
            if parsed_move is None:
                print("Invalid input. Please enter your move in the format 'e2 e4'.")
                continue
            start_pos, end_pos = parsed_move
            if is_valid_move(board, start_pos, end_pos, player_color, last_move, False):
                valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                if valid_move:
                    handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                    if is_checkmate(board, 'black' if player_color == 'White' else 'white'):
                        print(f"Checkmate! {player_color} wins!")
                        save_q_table(q_table)
                        return
                    if is_stalemate(board, 'black' if player_color == 'White' else 'white'):
                        print("Stalemate! It's a draw.")
                        save_q_table(q_table)
                        return
                    player_color = 'Black' if player_color == 'White' else 'White'
                else:
                    print("Invalid move. Try again.")
            else:
                print("Invalid move. Try again.")
    elif mode in ["computer", "play against the computer"]:
        print_board(board)
        first_open = True
        while True:
            move_valid = False
            while not move_valid:
                move = input("White's move (e.g., e2 e4): ").strip()
                if move.lower() == 'exit':
                    save_q_table(q_table)
                    return
                parsed_move = parse_move(move)
                if parsed_move is None:
                    print("Invalid input, please enter a move in the correct format (e.g., e2 e4).")
                    continue
                start_pos, end_pos = parsed_move
                move_result = is_valid_move(board, start_pos, end_pos, "White", last_move, False)
                if move_result:
                    valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                    if valid_move:
                        handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                        print_board(board)
                        if is_checkmate(board, 'black'):
                            print("Checkmate! White wins!")
                            save_q_table(q_table)
                            return
                        if is_stalemate(board, 'black'):
                            print("Stalemate! It's a draw.")
                            save_q_table(q_table)
                        move_valid = True
                    else:
                        print("Invalid move, try again.")
                else:
                    print("Invalid move, try again.")
            if first_open:
                white_move = ' '.join([f'{chr(start_pos[1] + 97)}{8 - start_pos[0]}' for start_pos in last_move])
                if white_move in openings:
                    computer_move = random.choice(openings[white_move])
                    start_pos, end_pos = parse_move(computer_move)
                    valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                    print_board(board)
                    first_open = False
                    continue
            state = copy.deepcopy(board)
            stime = time.time()
            action, depth, nodes = search(state, t_table)
            etime = time.time()
            ttime = etime - stime
            print(f"time: {ttime} nodes: {nodes} depth: {depth} score: {evaluate(board)}")
            start_pos, end_pos = action
            valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
            if valid_move:
                handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
                print_board(board)
                if is_checkmate(board, 'white'):
                    print("Checkmate! Black wins!")
                    save_q_table(q_table)
                    return
                if is_stalemate(board, 'white'):
                    print("Stalemate! It's a draw.")
                    save_q_table(q_table)
                    return
                reward = 0
                if is_check(board)["white"]:
                    reward = 1
                reward += evaluate(board) * -1
                next_state = copy.deepcopy(board)
                update_q_table(state, action, reward, next_state)
                first_open = False
    else:
        print("Invalid choice.")
        play_chess()
def sim_chess():
    board = copy.deepcopy(initial_board)
    last_move = None
    first_open = True
    hist = []
    t_table = {}
    print_board(board)
    while True:
        state = copy.deepcopy(initial_board)
        stime = time.time()
        white_move, depth, nodes = search(board, t_table, computer_color="White")
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
        valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
        if valid_move:
            handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
            print_board(board)
            if is_checkmate(board, 'black'):
                print("Checkmate! White wins!")
                save_q_table(q_table)
                return
            if is_stalemate(board, 'black'):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            white_reward = 0
            if is_check(board)["black"]:
                white_reward = 1
            white_reward += evaluate(board)
            next_state = copy.deepcopy(board)
            update_q_table(state, white_move, white_reward, next_state)
        if first_open:
            white_move = ' '.join([f'{chr(start_pos[1] + 97)}{8 - start_pos[0]}' for start_pos in last_move])
            if white_move in openings:
                computer_move = random.choice(openings[white_move])
                start_pos, end_pos = parse_move(computer_move)
                valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                print_board(board)
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
        valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
        if valid_move:
            handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
            print_board(board)
            if is_checkmate(board, 'white'):
                print("Checkmate! Black wins!")
                save_q_table(q_table)
                return
            if is_stalemate(board, 'white'):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            reward = 0
            if is_check(board)["white"]:
                reward = 1
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
# v2025(1.5 beta)
# created by ChatGPT

import random
import pickle
import time
import math
import copy
from piece import *

###############################################################################
# GLOBAL VARIABLES (unchanged)
###############################################################################

openings = {
    "e2 e4": ["c7 c5", "c7 c6", "e7 e6", "e7 e5"],
    "d2 d4": ["g8 f6", "f7 f5", "d7 d5"],
    "g1 f3": ["d7 d5", "c7 c5", "g8 f6", "f7 f5"],
    "c2 c4": ["g8 f6", "c7 c5", "f7 f5", "e7 e5"],
}

max_depth = 1
sim_time = 100  # ply's
pondering_move = None
pondering_thread = None
predict_opponent_move = None

unicode_pieces = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    ' ': ' '
}

initial_board = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"],
]

# Q-learning data structures
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

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def print_board(board, is_white=True):
    if is_white:
        print("  a b c d e f g h")
    else:
        print("  h g f e d c b a")
    for row in range(8):
        print(8 - row, end=" ")
        for col in range(8):
            print(unicode_pieces[board[row][col]], end=" ")
        print(8 - row)
    if is_white:
        print("  a b c d e f g h")
    else:
        print("  h g f e d c b a")
    return board

def parse_move(move):
    move = move.replace(',', '')
    parts = move.split()
    if len(parts) != 2 or not all(len(part) == 2 for part in parts):
        return None
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    try:
        start_col, start_row = col_map[parts[0][0]], int(parts[0][1]) - 1
        end_col, end_row = col_map[parts[1][0]], int(parts[1][1]) - 1
        # Flip row indexing to match your board representation if needed
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
    next_q_values = [get_q_value(next_state, a) for a in get_all_moves(next_state)]
    max_next_q = max(next_q_values, default=0)
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[(state_tuple, action)] = new_q

def get_t_key(board, t_table):
    if len(t_table) == 0:
        return 1000000000000000
    keys = [_ for _ in t_table]
    keys.sort()
    for k, p in t_table.items():
        if p[0] == board:
            return k
    return keys[0] - 1

###############################################################################
# FAST MAKE/UNMAKE MOVE FUNCTIONS
###############################################################################
def make_move(board, move):
    """
    Modifies the board in-place according to 'move', 
    then returns the piece that was captured (if any).
    """
    (start_x, start_y), (end_x, end_y) = move
    moving_piece = board[start_x][start_y]
    captured_piece = board[end_x][end_y]
    board[end_x][end_y] = moving_piece
    board[start_x][start_y] = " "
    return captured_piece

def unmake_move(board, move, captured_piece):
    """
    Reverts the board in-place by undoing 'move' and restoring the captured piece.
    """
    (start_x, start_y), (end_x, end_y) = move
    moving_piece = board[end_x][end_y]
    board[start_x][start_y] = moving_piece
    board[end_x][end_y] = captured_piece

###############################################################################
# NEGAMAX + QUIESCENCE SEARCH (with minimal copying)
###############################################################################
def negamax(board, depth, alpha, beta, color, last_move, t_table):
    if depth == 0 or is_game_over(board):
        return quiescence_search(board, alpha, beta, color, last_move, t_table)

    best_move = None
    total_nodes = 0
    alpha_original = alpha
    player_color = 'White' if color == 1 else 'Black'

    # Optional forward-pruning or null-move approach
    if depth >= 3 and not is_check(board)[player_color.lower()] and not is_endgame(board):
        score, best_move, nodes = negamax(board, depth - 3, -beta, -beta + 1, -color, last_move, t_table)
        total_nodes += nodes
        if score >= beta:
            return beta, best_move, total_nodes

    moves = get_all_moves(board, player_color, last_move)
    moves = order_moves(board, moves, player_color, last_move, depth)

    state_tuple = tuple(tuple(row) for row in board)

    old_last_move = last_move
    for move in moves:
        # Q-value check can short-circuit
        if (state_tuple, move) in q_table:
            if q_table[(state_tuple, move)] >= beta:
                return color * q_table[(state_tuple, move)], move, total_nodes

        # Make the move in-place
        captured_piece = make_move(board, move)
        last_move = move

        # Check to see if we can prune quickly
        is_in_check = is_check(board)[player_color.lower()]
        is_capture = (captured_piece != " ")
        piece_moved = board[move[1][0]][move[1][1]]
        is_promotion = (piece_moved.lower() == "p" and (move[1][0] == 0 or move[1][0] == 7))

        # Simple futility pruning
        if depth == 1 and not is_in_check and not is_capture and not is_promotion:
            static_eval = color * evaluate(board)
            if static_eval + 100 <= alpha:
                total_nodes += 1
                # Unmake and continue
                unmake_move(board, move, captured_piece)
                last_move = old_last_move
                continue

        score, _, nodes = negamax(board, depth - 1, -beta, -alpha, -color, last_move, t_table)
        total_nodes += nodes
        # Undo move
        unmake_move(board, move, captured_piece)
        last_move = old_last_move

        # Incorporate Q-value
        score = -score + get_q_value(board, move)

        if score >= beta:
            return beta, move, total_nodes

        if score > alpha:
            alpha = score
            best_move = move

    return alpha, best_move, total_nodes

def quiescence_search(board, alpha, beta, color, last_move, t_table, depth=0, max_q_depth=3):
    """
    A simpler form of search that only examines captures (or checks, etc.)
    to prevent horizon effect. Avoids large copying by using make/unmake move.
    """
    key = get_t_key(board, t_table)
    if key in t_table:
        stand_pat = color * t_table[key][1]
    else:
        stand_pat = evaluate(board)
        t_table[key] = (board, stand_pat)
        stand_pat *= color

    if depth >= max_q_depth or is_game_over(board):
        return stand_pat, None, 1

    if stand_pat >= beta:
        return beta, None, 1

    if alpha < stand_pat:
        alpha = stand_pat

    total_nodes = 1
    player_color = 'White' if color == 1 else 'Black'
    
    if not is_check(board)[player_color.lower()] and not is_endgame(board):
        if stand_pat >= beta:
            return beta, None, 1

    all_moves = get_all_moves(board, player_color, last_move)
    # Filter down to capturing moves
    capturing_moves = []
    for move in all_moves:
        (sx, sy), (ex, ey) = move
        if board[ex][ey] != " ":
            capturing_moves.append(move)

    capturing_moves = order_moves(board, capturing_moves, player_color, last_move, depth)
    old_last_move = last_move
    best_move = None

    for move in capturing_moves:
        captured_piece = make_move(board, move)
        last_move = move

        score, _, nodes = quiescence_search(board, -beta, -alpha, -color, last_move, t_table, depth + 1, max_q_depth)
        total_nodes += nodes

        unmake_move(board, move, captured_piece)
        last_move = old_last_move

        score = -score

        if score >= beta:
            return beta, move, total_nodes
        if score > alpha:
            alpha = score
            best_move = move

    return alpha, best_move, total_nodes

###############################################################################
# SEARCH WRAPPER
###############################################################################
def search(board, t_table, computer_color="Black", depth=max_depth, last_move=None):
    color = -1 if computer_color == "Black" else 1
    alpha = float('-inf')
    beta = float('inf')

    eval_score, best_move, total_nodes = negamax(board, depth, alpha, beta, color, last_move, t_table)

    # Castling stuff (if you track these with booleans)
    piece = board[best_move[0][0]][best_move[0][1]]
    if piece.isupper():
        if piece == "K":
            white_king_moved = True  # you'd define these globally or track them
        elif piece == "R":
            white_rook_moved = True
    else:
        if piece == "k":
            black_king_moved = True
        elif piece == "r":
            black_rook_moved = True

    return best_move, depth, total_nodes

###############################################################################
# MOVE ORDERING
###############################################################################
def order_moves(board, moves, player_color, last_move=None, depth=0):
    opponent_color = 'White' if player_color == 'Black' else 'Black'
    move_scores = []

    for move in moves:
        (sx, sy), (ex, ey) = move
        piece = board[sx][sy]
        piece_val = piece_values[piece.upper()]
        # Create a quick “score” for ordering
        captured_piece = board[ex][ey]
        # Basic static exchange or material changes
        # (No more deep copying; just a quick eval approach)

        # We'll do a small in-place move just for evaluation ordering:
        tmp_captured = board[ex][ey]
        board[ex][ey] = piece
        board[sx][sy] = " "

        # Evaluate (can do color-based or a simpler approach)
        if player_color == "Black":
            base_score = -evaluate(board)
        else:
            base_score = evaluate(board)

        # Revert quickly
        board[sx][sy] = piece
        board[ex][ey] = tmp_captured

        # Extra penalty or reward if capturing a valuable piece
        if captured_piece != " ":
            base_score += piece_values.get(captured_piece.upper(), 0) * 10

        # Add the Q-value
        q_value = get_q_value(board, move)
        move_score = base_score + q_value
        move_scores.append((move_score, move))

    # Sort descending
    move_scores.sort(reverse=True, key=lambda x: x[0])
    ordered_moves = [m for _, m in move_scores]
    return ordered_moves

###############################################################################
# MAIN GAME LOOPS
###############################################################################
def play_chess():
    board = copy.deepcopy(initial_board)
    player_color = 'White'
    last_move = None
    t_table = {}

    mode = input("Do you want to play against a person or the computer? ").strip().lower()

    # ----------- Person vs Person -----------
    if mode in ["person", "play against a person"]:
        while True:
            print_board(board)
            move = input(f"{player_color}'s turn. Enter your move (e.g., e2 e4): ").strip()
            if move.lower() == 'quit':
                save_q_table(q_table)
                return
            parsed_move = parse_move(move)
            if parsed_move is None:
                print("Invalid input. Please enter your move in the format 'e2 e4'.")
                continue
            start_pos, end_pos = parsed_move

            if is_valid_move(board, start_pos, end_pos, player_color, last_move, False):
                valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                if valid_move:
                    handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                    if is_checkmate(board, 'black' if player_color == 'White' else 'white'):
                        print(f"Checkmate! {player_color} wins!")
                        save_q_table(q_table)
                        return
                    if is_stalemate(board, 'black' if player_color == 'White' else 'white'):
                        print("Stalemate! It's a draw.")
                        save_q_table(q_table)
                        return
                    player_color = 'Black' if player_color == 'White' else 'White'
                else:
                    print("Invalid move. Try again.")
            else:
                print("Invalid move. Try again.")

    # ----------- Person vs Computer -----------
    elif mode in ["computer", "play against the computer"]:
        print_board(board)
        first_open = True
        while True:
            # Human move
            move_valid = False
            while not move_valid:
                move = input("White's move (e.g., e2 e4): ").strip()
                if move.lower() == 'exit':
                    save_q_table(q_table)
                    return
                parsed_move = parse_move(move)
                if parsed_move is None:
                    print("Invalid input, please enter a move in the correct format (e.g., e2 e4).")
                    continue
                start_pos, end_pos = parsed_move
                if is_valid_move(board, start_pos, end_pos, "White", last_move, False):
                    valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
                    if valid_move:
                        handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]])
                        print_board(board)
                        if is_checkmate(board, 'black'):
                            print("Checkmate! White wins!")
                            save_q_table(q_table)
                            return
                        if is_stalemate(board, 'black'):
                            print("Stalemate! It's a draw.")
                            save_q_table(q_table)
                            return
                        move_valid = True
                    else:
                        print("Invalid move, try again.")
                else:
                    print("Invalid move, try again.")

            # Check opening move shortcuts
            if first_open:
                white_move = ' '.join([f'{chr(start_pos[1] + 97)}{8 - start_pos[0]}' 
                                       for start_pos in last_move])
                if white_move in openings:
                    computer_move = random.choice(openings[white_move])
                    start_pos, end_pos = parse_move(computer_move)
                    move_piece(board, start_pos, end_pos, last_move)
                    print_board(board)
                    first_open = False
                    continue

            # Computer move
            state = copy.deepcopy(board)
            start = time.time()
            action, depth, nodes = search(state, t_table)
            print("time", time.time() - start, "nodes", nodes)
            start_pos, end_pos = action
            valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
            if valid_move:
                handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
                print_board(board)
                if is_checkmate(board, 'white'):
                    print("Checkmate! Black wins!")
                    save_q_table(q_table)
                    return
                if is_stalemate(board, 'white'):
                    print("Stalemate! It's a draw.")
                    save_q_table(q_table)
                    return
                reward = 0
                if is_check(board)["white"]:
                    reward = 1
                reward += evaluate(board) * -1
                next_state = copy.deepcopy(board)
                update_q_table(state, action, reward, next_state)
                first_open = False

    else:
        print("Invalid choice.")
        play_chess()

def sim_chess():
    board = copy.deepcopy(initial_board)
    last_move = None
    first_open = True
    hist = []
    t_table = {}
    print_board(board)

    while True:
        # White move
        state = copy.deepcopy(board)
        white_move, depth, nodes = search(board, t_table, computer_color="White")
        hist.append(white_move)
        if len(hist) >= sim_time:
            save_q_table(q_table)
            return

        start_pos, end_pos = white_move
        valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
        if valid_move:
            handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
            print_board(board)
            if is_checkmate(board, 'black'):
                print("Checkmate! White wins!")
                save_q_table(q_table)
                return
            if is_stalemate(board, 'black'):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            white_reward = 0
            if is_check(board)["black"]:
                white_reward = 1
            white_reward += evaluate(board)
            next_state = copy.deepcopy(board)
            update_q_table(state, white_move, white_reward, next_state)

        if first_open:
            wm_str = ' '.join([f'{chr(start_pos[1] + 97)}{8 - start_pos[0]}' for start_pos in last_move])
            if wm_str in openings:
                computer_move = random.choice(openings[wm_str])
                start_pos, end_pos = parse_move(computer_move)
                move_piece(board, start_pos, end_pos, last_move)
                print_board(board)
                first_open = False
                continue

        # Black move
        state = copy.deepcopy(board)
        black_move, depth, nodes = search(board, t_table)
        hist.append(black_move)
        if len(hist) >= sim_time:
            save_q_table(q_table)
            return

        start_pos, end_pos = black_move
        valid_move, last_move = move_piece(board, start_pos, end_pos, last_move)
        if valid_move:
            handle_promotion(board, end_pos[0], end_pos[1], board[end_pos[0]][end_pos[1]], True)
            print_board(board)
            if is_checkmate(board, 'white'):
                print("Checkmate! Black wins!")
                save_q_table(q_table)
                return
            if is_stalemate(board, 'white'):
                print("Stalemate! It's a draw.")
                save_q_table(q_table)
                return
            reward = 0
            if is_check(board)["white"]:
                reward = 1
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
        #    sim_chess1()  # presumably another simulation function you might have
        #    print("Good game.")
        #    quit()
        else:
            print("Awww.")
            quit()

if __name__ == "__main__":
    main()
