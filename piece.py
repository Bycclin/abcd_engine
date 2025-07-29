from random import getrandbits
from collections import namedtuple
from itertools import count
piece_values = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,}
                  #'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
#TODO: do this for the check:
#where when no legal move can prevent a situation where the king can be “captured”,
unicode_pieces = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    ' ': ' '
}
pst = {
    'P': ([   0,   0,   0,   0,   0,   0,   0,   0],
          [  78,  83,  86,  73, 102,  82,  85,  90],
          [   7,  29,  21,  44,  40,  31,  44,   7],
          [ -17,  16,  -2,  15,  14,   0,  15, -13],
          [ -26,   3,  10,   9,   6,   1,   0, -23],
          [ -22,   9,   5, -11, -10,  -2,   3, -19],
          [ -31,   8,  -7, -37, -36, -14,   3, -31],
          [   0,   0,   0,   0,   0,   0,   0,   0]),
    'N': ([ -66, -53, -75, -75, -10, -55, -58, -70],
          [  -3,  -6, 100, -36,   4,  62,  -4, -14],
          [  10,  67,   1,  74,  73,  27,  62,  -2],
          [  24,  24,  45,  37,  33,  41,  25,  17],
          [  -1,   5,  31,  21,  22,  35,   2,   0],
          [ -18,  10,  13,  22,  18,  15,  11, -14],
          [ -23, -15,   2,   0,   2,   0, -23, -20],
          [ -74, -23, -26, -24, -19, -35, -22, -69]),
    'B': ([ -59, -78, -82, -76, -23, -107, -37, -50],
          [ -11,  20,  35, -42, -39,  31,   2, -22],
          [  -9,  39, -32,  41,  52, -10,  28, -14],
          [  25,  17,  20,  34,  26,  25,  15,  10],
          [  13,  10,  17,  23,  17,  16,   0,   7],
          [  14,  25,  24,  15,   8,  25,  20,  15],
          [  19,  20,  11,   6,   7,   6,  20,  16],
          [  -7,   2, -15, -12, -14, -15, -10, -10]),
    'R': ([  35,  29,  33,   4,  37,  33,  56,  50],
          [  55,  29,  56,  67,  55,  62,  34,  60],
          [  19,  35,  28,  33,  45,  27,  25,  15],
          [   0,   5,  16,  13,  18,  -4,  -9,  -6],
          [ -28, -35, -16, -21, -13, -29, -46, -30],
          [ -42, -28, -42, -25, -25, -35, -26, -46],
          [ -53, -38, -31, -26, -29, -43, -44, -53],
          [ -30, -24, -18,   5,  -2, -18, -31, -32]),
    'Q': ([   6,   1,  -8, -104,  69,  24,  88,  26],
          [  14,  32,  60,  -10,  20,  76,  57,  24],
          [  -2,  43,  32,   60,  72,  63,  43,   2],
          [   1, -16,  22,   17,  25,  20, -13,  -6],
          [ -14, -15,  -2,   -5,  -1, -10, -20, -22],
          [ -30,  -6, -13,  -11, -16, -11, -16, -27],
          [ -36, -18,   0,  -19, -15, -15, -21, -38],
          [ -39, -30, -31,  -13, -31, -36, -34, -42]),
    'K': ([   4,  54,  47, -99, -99,  60,  83, -62],
          [ -32,  10,  55,  56,  56,  55,  10,   3],
          [ -62,  12, -57,  44, -67,  28,  37, -31],
          [ -55,  50,  11,  -4, -19,  13,   0, -49],
          [ -55, -43, -52, -28, -51, -47,  -8, -50],
          [ -47, -42, -43, -79, -64, -32, -29, -32],
          [  -4,   3, -14, -50, -57, -18,  13,   4],
          [  17,  30,  -3, -14,   6,  -1,  40,  18]),
}
manhattan_distance = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
N, E, S, W = 8, 1, -8, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}
ZOBRIST_TABLE = [[getrandbits(64) for _ in range(12)] for _ in range(64)]
initial = {
            'white': [
                #0x000000000000FF00,
                65280,
                0x0000000000000042,
                0x0000000000000024,
                0x0000000000000081,
                0x0000000000000008,
                #0x0000000000000010,
                16,
                
            ],
            'black': [
                #0x00FF000000000000,
                71776119061217280,
                0x4200000000000000,
                0x2400000000000000,
                0x8100000000000000,
                0x0800000000000000,
                0x1000000000000000, 
            ]
        }
class Position(namedtuple("Position", "board score wc bc ep kp")):
    def piece_square(self, coord, color):
        for piece, bitboard in zip("PNBRQK", self.board[color]):
            if bitboard & (1 << coord):
                return piece if color == 'white' else piece.lower()
        return " "
    def genMoves(self):
        friendly_board = 0
        for bb in self.board["white"]:
            friendly_board |= bb
        opponent_board = 0
        for bb in self.board["black"]:
            opponent_board |= bb
        overall = friendly_board | opponent_board
        for square in range(64):
            piece = self.piece_square(square, "white")
            if piece == " ":
                continue
            for d in directions[piece]:
                for o in count(square + d, d):#loop through the dir "rays"
                    if o < 0 or o > 63:# stay within board
                        break
                    if piece.upper() in "BQ":#stop BQ wrap
                        if abs((o % 8) - ((o - d) % 8)) > 1:
                            break
                    if piece.upper() in 'N':# prevent N wrap
                        old_rank, old_file = divmod(square, 8)
                        new_rank, new_file = divmod(o, 8)
                        dr = abs(new_rank - old_rank)
                        df = abs(new_file - old_file)
                        if not ((dr == 2 and df == 1) or (dr == 1 and df == 2)):
                            break
                    if self.piece_square(o, "white").isupper():# stay off friendly pieces
                        break
                    if piece.upper() == 'P':# pawn‐specific legality
                        old_file, new_file = square % 8, o % 8
                        if abs(new_file - old_file) > 2:
                            break
                        if friendly_board & (1 << o):
                            break
                        forward, diag_moves = 8, [7, 9]
                        if d == forward:
                            if overall & (1 << o):
                                break
                        elif d in diag_moves:
                            if not (opponent_board & (1 << o)):
                                break
                        if 0 <= o <= 7:
                            for prom in "NBRQ":
                                yield (square, o, prom)
                    yield (square, o, "")#move it
                    if piece in "PNK" or self.piece_square(o, "black") != " ":
                        break
                    if square == 4:# castling
                        if self.wc[0] and not (overall & ((1 << 5) | (1 << 6))):
                            yield (4, 6, "")
                        if self.wc[1] and not (overall & ((1 << 1) | (1 << 2) | (1 << 3))):
                            yield (4, 2, "")
    def move_piece(self, move):
        start, end = move[0], move[1]#TODO: promotion
        start_m = 1 << start
        end_m = 1 << end
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        put = lambda bb: (bb & ~start_m) | end_m
        piece = self.piece_square(start, "white")
        if start == 63: wc = (self.wc[0], False)
        if start == 56: wc = (False, self.wc[1])
        if end == 7: bc = (self.bc[0], False)
        if end == 0: bc = (False, self.bc[1])
        if piece.upper() == "K":
            wc = (False, False)
            if abs(end - start) == 2:
                king_index = piece_to_index['K']
                self.board["white"][king_index] = put(self.board["white"][king_index])
                if end > start:
                    rook_start = start + 3
                    rook_end = start + 1
                else:
                    rook_start = start - 4
                    rook_end = start - 1
                move_rook = lambda bb: (bb & ~(1 << rook_start)) | (1 << rook_end)
                board["white"][3] = move_rook(self.board["white"][3])
                kp = ((start + end) // 2, self.kp[1])
        index = piece_to_index[piece.upper()]
        board["white"][index] = put(self.board["white"][index])
        for idx, bb in enumerate(self.board["black"]):
            if bb & end_m:
                board["black"][idx] = bb & ~end_m
        return Position(board, self.score, wc, bc, ep, kp).rotate()
    def print_board(self, is_white=True):
        board_array = [[' ' for _ in range(8)] for _ in range(8)]
        for square_idx_a1_oriented in range(64):
            rank_idx_board_array = square_idx_a1_oriented // 8 
            file_idx_board_array = square_idx_a1_oriented % 8  
            row, col = rank_idx_board_array, file_idx_board_array
            determined_char_for_unicode = ' ' # Default to empty
            for p_base_idx, bb in enumerate(self.board["white"]):
                p_base = "PNBRQK"[p_base_idx]
                if bb & (1 << square_idx_a1_oriented):
                    determined_char_for_unicode = p_base.upper() if is_white else p_base.lower()
                    break
            if determined_char_for_unicode == ' ':
                for p_base_idx, bb in enumerate(self.board["black"]):
                    p_base = "PNBRQK"[p_base_idx]
                    if bb & (1 << square_idx_a1_oriented):
                        determined_char_for_unicode = p_base.lower() if is_white else p_base.upper()
                        break
            board_array[row][col] = unicode_pieces[determined_char_for_unicode]
        if is_white: # White's perspective (White at bottom)
            files_header = "  a b c d e f g h"
            print(files_header)
            for r_array_idx in range(7, -1, -1): # Iterate from board_array[7] down to board_array[0]
                rank_label = r_array_idx + 1 # Rank 8 for board_array[7], Rank 1 for board_array[0]
                print(f"{rank_label} ", end="")
                for c_array_idx in range(8): # Iterate files a to h
                    print(board_array[r_array_idx][c_array_idx], end=" ")
                print(f" {rank_label}") # Added space for alignment
            print(files_header)
        else: # Black's perspective (Black at bottom)
            files_header = "  h g f e d c b a"
            h = 0
            print(files_header)
            for r_array_idx in range(7, -1, -1): # Iterate from board_array[7] down to board_array[0]
                h += 1
                rank_label = h #reverse the rank numbers
                print(f"{rank_label} ", end="")
                for c_array_idx in range(7, -1, -1): # Iterate files h down to a
                    print(board_array[r_array_idx][c_array_idx], end=" ")
                print(f" {rank_label}") # Added space for alignment
            print(files_header)
    def undo_move(self, start, end, capture):
        start_sq = start #TODO get rid of this function and rely on TTable
        end_sq = end
        moving_piece = self.piece_square(end, "white")
        moving_index = piece_to_index[moving_piece]
        for piece in "PNBRQK":
            idx = piece_to_index[piece]
            if self.board["white"][idx] & (1 << end_sq):
                moving_piece = piece
                moving_index = idx
                break
        if moving_piece.upper() == "K" and abs(end_sq - start_sq) == 2:
            self.board["white"][moving_index] = (self.board["white"][moving_index] & ~(1 << end_sq)) | (1 << start_sq)
            if end_sq > start_sq:
                rook_original_sq = start_sq + 3
                rook_moved_sq = start_sq + 1
            else:
                rook_original_sq = start_sq - 4
                rook_moved_sq = start_sq - 1
            self.board["white"][3] = (self.board["white"][3] & ~(1 << rook_moved_sq)) | (1 << rook_original_sq)
            self.kp = start_sq
        else:
            self.board["white"][moving_index] = (self.board["white"][moving_index] & ~(1 << end_sq)) | (1 << start_sq)
        if capture != " ":
            black_piece_to_index = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
            cap_index = black_piece_to_index[capture]
            self.board["black"][cap_index] |= (1 << end_sq)
    def hash(self):
        hash_value = 0
        for color in ['white', 'black']:
            pieces = "PNBRQK" if color == 'white' else "pnbrqk"
            for idx, bitboard in enumerate(self.board[color]):
                for square in range(64):
                    if bitboard & (1 << square):
                        piece = pieces[idx]
                        hash_value ^= ZOBRIST_TABLE[square][piece_to_index[piece]]
        return hash_value
    def is_check(self):
        temp_board = self.rotate()
        for move in temp_board.genMoves():
            if move[1] == self.kp:
                return True
        return False
    def is_checkmate(self):
        if self.is_check() and len(list(self.genMoves())) == 0:
            return True
        return False
    def is_endgame(self):
        material = 0
        for color in ['white', 'black']:
            for idx, bitboard in enumerate(self.board[color]):
                material += bin(bitboard).count('1') * piece_values["PNBRQK"[idx]]
                if material >= 1000:
                    return False
        return True
    def material_count(self, color):
        counts = {'K': 0, 'N': 0, 'B': 0}
        for idx, bb in enumerate(self.board[color]):
            piece = "PNBRQK"[idx]
            count = bin(bb).count('1')
            if piece in counts:
                counts[piece] += count
            elif count > 0:
                return None
        return counts
    def is_draw(self):
        if len(list(self.genMoves())) == 0 and not self.is_check():#TODO: just rely on TTable
            return True
        white_counts = self.material_count('white')
        black_counts = self.material_count('black')
        if white_counts is not None and black_counts is not None:
            pieces = [white_counts, black_counts]
            total_knights = sum(p['N'] for p in pieces)
            total_bishops = sum(p['B'] for p in pieces)
            if total_knights == 0 and total_bishops == 0:
                return True
            if (total_bishops == 1 and total_knights == 0) or (total_bishops == 0 and total_knights == 1):
                return True
        return False
    def rotate(self):
        def flip_vertical(bb):
            return ((bb & 0x00000000000000FF) << 56) | \
                   ((bb & 0x000000000000FF00) << 40) | \
                   ((bb & 0x0000000000FF0000) << 24) | \
                   ((bb & 0x00000000FF000000) << 8)  | \
                   ((bb & 0x000000FF00000000) >> 8)  | \
                   ((bb & 0x0000FF0000000000) >> 24) | \
                   ((bb & 0x00FF000000000000) >> 40) | \
                   ((bb & 0xFF00000000000000) >> 56)
        new_board = {
            "white": [flip_vertical(bb) for bb in self.board["black"]],
            "black": [flip_vertical(bb) for bb in self.board["white"]]
        }
        return Position(
            new_board,
            -self.score,
            self.bc,
            self.wc,
            63 - self.ep if self.ep is not None else None,
            63 - self.kp if self.kp is not None else None,
        )
def pst_val(piece, square):
    piece = piece.upper()
    if piece not in pst:
        return 0
    row, col = square // 8, square % 8
    return pst[piece][row][col]
def evaluate(board):
    score = 0
    for square in range(64):
        for color in ['white', 'black']:
            for idx, bitboard in enumerate(board._initial[color]):
                if bitboard & (1 << square):
                    piece = "PNBRQK"[idx] if color == 'white' else "pnbrqk"[idx]
                    score += piece_values[piece]
                    score += pst_val(piece, square)
    return
