from random import getrandbits
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
#might go back to N, S = 13, -13
N, E, S, W = 8, 1, -8, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

class Position:
    def __init__(self):
        self.kp = 4
        self.wc = (True, True)
        self.bc = (True, True)
        self._initial = {
            'white': [
                0x000000000000FF00, 
                0x0000000000000042,
                0x0000000000000024,
                0x0000000000000081,
                0x0000000000000008,
                0x0000000000000010
            ],
            'black': [
                0x00FF000000000000, 
                0x4200000000000000,
                0x2400000000000000,
                0x8100000000000000,
                0x0800000000000000, 
                0x1000000000000000
            ]
        }
        self._history = []
        self._board = 0
        for col in self._initial:
            for bb in self._initial[col]:
                self._board |= bb
        table = []
        for _ in range(64):
            row = []
            for _ in range(12):
                row.append(getrandbits(64))
            table.append(row)
        self.ZOBRIST_TABLE = table
        self.ZOBRIST_PIECE_INDEX = piece_to_index
    def piece_square(self, coord, color):
        row, col = coord
        square = (7 - row) * 8 + col
        for piece, bitboard in zip("PNBRQK", self._initial[color]):
            if bitboard & (1 << square):
                return piece if color == 'white' else piece.lower()
        return " "
    def _square_to_coord(self, square):
        row = 7 - (square // 8)
        col = square % 8
        return (row, col)
    def genMoves(self, color):
        moves = []
        friendly_board = 0
        for bb in self._initial[color.lower()]:
            friendly_board |= bb
        overall = self._board
        opponent_board = overall & ~friendly_board
        for square in range(64):
            if not (friendly_board & (1 << square)):
                continue
            piece = self.piece_square(self._square_to_coord(square), color)
            if piece is None:
                continue
            if piece.upper() == 'P':
                for direction in directions[piece.upper()]:
                    adj_direction = direction
                    if piece.upper() == 'P' and color == 'black':
                        adj_direction = -direction
                    new_square = square + adj_direction
                    if new_square < 0 or new_square >= 64:
                        continue
                    old_file, new_file = square % 8, new_square % 8
                    if abs(new_file - old_file) > 2:
                        continue
                    if friendly_board & (1 << new_square):
                        continue
                    if piece.upper() == 'P':
                        if color == 'white':
                            forward, diag_moves = 8, [7, 9]
                        else:
                            forward, diag_moves = -8, [-7, -9]
                        if adj_direction == forward:
                            if overall & (1 << new_square):
                                continue
                        elif adj_direction in diag_moves:
                            if not (opponent_board & (1 << new_square)):
                                continue
                    moves.append((self._square_to_coord(square), self._square_to_coord(new_square)))
                if color == 'white' and square == 4:
                    if self.wc[0] and not (overall & ((1 << 5) | (1 << 6))):
                        moves.append((self._square_to_coord(4), self._square_to_coord(6)))
                    if self.wc[1] and not (overall & ((1 << 1) | (1 << 2) | (1 << 3))):
                        moves.append((self._square_to_coord(4), self._square_to_coord(2)))
                elif color == 'black' and square == 60:
                    if self.bc[0] and not (overall & ((1 << 61) | (1 << 62))):
                        moves.append((self._square_to_coord(60), self._square_to_coord(62)))
                    if self.bc[1] and not (overall & ((1 << 57) | (1 << 58) | (1 << 59))):
                        moves.append((self._square_to_coord(60), self._square_to_coord(58)))
        return moves
    def move_piece(self, start, end):
        s_x, s_y = start
        e_x, e_y = end
        start_sq = (7 - s_x) * 8 + s_y
        end_sq   = (7 - e_x) * 8 + e_y
        start_m = 1 << start_sq
        end_m   = 1 << end_sq
        put = lambda bb: (bb & ~start_m) | end_m
        f_piece = self.piece_square(start, "white")
        l_piece = self.piece_square(start, "black")
        if f_piece != ' ':
            piece = f_piece
        elif l_piece != ' ':
            piece = l_piece
        else:
            return False
        if start_sq == 63: self.wc = (self.wc[0], False)
        if start_sq == 56: self.wc = (False, self.wc[1])
        if end_sq == 7: self.bc = (self.bc[0], False)
        if end_sq == 0: self.bc = (False, self.bc[1])
        self._history.append((self._board, { 'white': self._initial['white'][:], 'black': self._initial['black'][:] }))
        if piece.upper() == "K":
            self.wc = (False, False)
            if abs(end_sq - start_sq) == 2:
                king_index = piece_to_index['K']
                self._initial["white"][king_index] = put(self._initial["white"][king_index])
                if end_sq > start_sq:
                    rook_start = start_sq + 3
                    rook_end = start_sq + 1
                else:
                    rook_start = start_sq - 4
                    rook_end = start_sq - 1
                move_rook = lambda bb: (bb & ~(1 << rook_start)) | (1 << rook_end)
                self._initial["white"][3] = move_rook(self._initial["white"][3])
                self.kp = ((start_sq + end_sq) // 2, self.kp[1])
        index = piece_to_index[piece.upper()]
        self._initial["white"][index] = put(self._initial["white"][index])
        for idx, bb in enumerate(self._initial["black"]):
            if bb & end_m:
                self._initial["black"][idx] = bb & ~end_m
        self._board = 0
        for col in ['white', 'black']:
            for bb in self._initial[col]:
                self._board |= bb
        self.rotate()
        return True
    def print_board(self, is_white=True):
        self.rotate()
        board_array = [[' ' for _ in range(8)] for _ in range(8)]
        for square in range(64):
            row = 7 - (square // 8)
            col = square % 8
            piece = None
            for p, bb in zip("PNBRQK", self._initial["white"]):
                if bb & (1 << square):
                    piece = p
                    break
            if piece is None:
                for p, bb in zip("pnbrqk", self._initial["black"]):
                    if bb & (1 << square):
                        piece = p
                        break
            if piece is None:
                piece = ' '
            board_array[row][col] = piece
        if is_white:
            print("  a b c d e f g h")
        else:
            print("  h g f e d c b a")
        for row in range(8):
            rank_label = 8 - row
            print(str(rank_label), end=" ")
            if is_white:
                for col in range(8):
                    print(unicode_pieces[board_array[row][col]], end=" ")
            else:
                for col in range(7, -1, -1):
                    print(unicode_pieces[board_array[row][col]], end=" ")
            print(str(rank_label))
        if is_white:
            print("  a b c d e f g h")
        else:
            print("  h g f e d c b a")
    def undo_move(self):
        if self._history: 
            prev_board, prev_initial = self._history.pop()
            self._board = prev_board
            self._initial = prev_initial
            return True
        return False
    def hash(self):
        hash_value = 0
        for color in ['white', 'black']:
            pieces = "PNBRQK" if color == 'white' else "pnbrqk"
            for idx, bitboard in enumerate(self._initial[color]):
                for square in range(64):
                    if bitboard & (1 << square):
                        piece = pieces[idx]
                        hash_value ^= self.ZOBRIST_TABLE[square][self.ZOBRIST_PIECE_INDEX[piece]]
        return hash_value
    def is_check(self):
        king_coord = self._square_to_coord(self.kp)
        for move in self.genMoves("black"):
            if move[1] == king_coord:
                return True
        return False
    def is_checkmate(self):
        if self.is_check() and len(self.genMoves("white")) == 0:
            return True
        return False
    def is_endgame(self):
        material = 0
        for color in ['white', 'black']:
            for idx, bitboard in enumerate(self._initial[color]):
                material += bin(bitboard).count('1') * piece_values["PNBRQK"[idx]]
                if material >= 1000:
                    return False
        return True
    def is_draw(self, to_move):
        if self.genMoves("white") == 0 or self.genMoves("black") == 0:
            return True
        for color in ['white', 'black']:
            if color == "black": self.rotate()
            for move in self.genMoves(color):
                total_moves = len(self.genMoves(color))
                check_move = 0
                if self.move_piece(move[0], move[1]):
                    if self.is_check():
                        check_move += 1
                    self.undo_move()
            if color == "black": self.rotate()
            if check_move == total_moves and to_move == color:
                return True
            else:
                return False
    def rotate(self):
        def rotate_bitboard(bb):
            rotated = 0
            for square in range(64):
                if bb & (1 << square):
                    rotated |= (1 << (63 - square))
            return rotated
        new_initial = {'white': [], 'black': []}
        for i in range(6):
            new_initial['white'].append(rotate_bitboard(self._initial['black'][i]))
            new_initial['black'].append(rotate_bitboard(self._initial['white'][i]))
        self._initial = new_initial
        self._board = 0
        for color in ['white', 'black']:
            for bb in self._initial[color]:
                self._board |= bb
        self.wc, self.bc = self.bc, self.wc
        new_white_king = 63 - self.kp
        self.kp = new_white_king
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
