piece_values = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
piece_to_index = {'P': 0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
                  'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
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
    'B': ([ -59, -78, -82, -76, -23,-107, -37, -50],
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
    'Q': ([   6,   1,  -8,-104,  69,  24,  88,  26],
          [  14,  32,  60, -10,  20,  76,  57,  24],
          [  -2,  43,  32,  60,  72,  63,  43,   2],
          [   1, -16,  22,  17,  25,  20, -13,  -6],
          [ -14, -15,  -2,  -5,  -1, -10, -20, -22],
          [ -30,  -6, -13, -11, -16, -11, -16, -27],
          [ -36, -18,   0, -19, -15, -15, -21, -38],
          [ -39, -30, -31, -13, -31, -36, -34, -42]),
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
        self.is_checkmate = [False, False]
        self.is_draw = False
        self.is_check = [False, False]
        self.wc = (False, False)
        self.bc = (False, False)
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
        seed = 0xABCDEF
        a = 6364136223846793005
        c = 1
        m = 2**64
        def lcg():
            nonlocal seed
            while True:
                seed = (a * seed + c) % m
                yield seed
        rng = lcg()
        table = []
        for _ in range(64):
            row = []
            for _ in range(12):
                row.append(next(rng))
            table.append(row)
        self.ZOBRIST_TABLE = table
        self.ZOBRIST_PIECE_INDEX = piece_to_index
    def piece_square(self, square, color):
        for piece, bitboard in zip("PNBRQK", self._initial[color]):
            if bitboard & (1 << square):
                return piece if color == 'white' else piece.lower()
        return None
    def _square_to_coord(self, square):
        row = 7 - (square // 8)
        col = square % 8
        return (row, col)
    def genMoves(self, color):
        moves = []
        friendly_board = 0
        for bb in self._initial[color]:
            friendly_board |= bb
        overall = self._board
        opponent_board = overall & ~friendly_board
        for square in range(64):
            if not (friendly_board & (1 << square)): continue
            piece = self.piece_square(square, color)
            if piece is None: continue
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
                if self.wc[0] and color == 'white' or self.bc[0] and color == "black":#
                    moves.append(())                                                  # Add check if the rook can "slide" to the king
                moves.append((self._square_to_coord(square), self._square_to_coord(new_square)))
        return moves
    def move_piece(self, start, end):
        s_x, s_y = start
        e_x, e_y = end
        start_sq = (7 - s_x) * 8 + s_y
        end_sq   = (7 - e_x) * 8 + e_y
        try:
            start_m = 1 << start_sq
            end_m   = 1 << end_sq
        except:
            return False
        self._history.append((self._board, { 'white': self._initial['white'][:], 'black': self._initial['black'][:] }))
        moved_color = None
        for color in ['white', 'black']:
            for idx, bb in enumerate(self._initial[color]):
                if bb & start_m:
                    self._initial[color][idx] = (bb & ~start_m) | end_m
                    moved_color = color
                    break
            if moved_color is not None:
                break
        opponent_color = 'black' if moved_color == 'white' else 'white'
        for idx, bb in enumerate(self._initial[opponent_color]):
            if bb & end_m:
                self._initial[opponent_color][idx] = bb & ~end_m
        self._board = 0
        for color in ['white', 'black']:
            for bb in self._initial[color]:
                self._board |= bb
        return True
    def print_board(self, is_white=True):
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
def evaluate_pawn_structure(board):
    score = 0
    for color in ['white', 'black']:
        pawn_positions = []
        for x in range(8):
            for y in range(8):
                piece = board[x][y]
                if (color == 'white' and piece == 'P') or (color == 'black' and piece == 'p'):
                    pawn_positions.append((x, y))
        for pawn in pawn_positions:
            x, y = pawn
            files_to_check = []
            if y > 0:
                files_to_check.append(y - 1)
            if y < 7:
                files_to_check.append(y + 1)
            isolated = True
            for fy in files_to_check:
                for fx in range(8):
                    if (fx, fy) in pawn_positions:
                        isolated = False
                        break
                if not isolated:
                    break
            if isolated:
                penalty = 15
                score -= penalty if color == 'white' else -penalty
        file_counts = {}
        for pawn in pawn_positions:
            x, y = pawn
            file_counts.setdefault(y, []).append(x)
        for file_pawns in file_counts.values():
            if len(file_pawns) > 1:
                penalty = 10 * (len(file_pawns) - 1)
                score -= penalty if color == 'white' else -penalty
        for pawn in pawn_positions:
            x, y = pawn
            passed = True
            if color == 'white':
                for fx in range(x):
                    if board[fx][y] == 'p':
                        passed = False
                        break
            else:
                for fx in range(x + 1, 8):
                    if board[fx][y] == 'P':
                        passed = False
                        break
            if passed:
                advancement = (7 - x) * 10 if color == 'white' else x * 10
                score += advancement if color == 'white' else -advancement
    return int(score / 5)
def get_pst_val(piece, place, color, pst=pst):
    piece = piece.upper()
    piece_table = pst[piece]
    if color == "Black":
        piece_table = [i for i in piece_table]
        piece_table.reverse()
    return piece_table[place[0]][place[1]]
def evaluate_king_safety(board):
    score = 0
    for color in ['white', 'black']:
        king_pos = find_king(board, color)
        if not king_pos:
            continue
        x, y = king_pos
        pawn_shield = 0
        if not is_endgame(board):
            if color == 'white':
                for dx in [-1, 0, 1]:
                    nx, ny = x - 1, y + dx
                    if 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == 'P':
                        pawn_shield += 1
            else:
                for dx in [-1, 0, 1]:
                    nx, ny = x + 1, y + dx
                    if 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == 'p':
                        pawn_shield += 1
            if pawn_shield < 2:
                penalty = (2 - pawn_shield) ** 20
                score -= penalty if color == 'white' else -penalty
    return score
def evaluate_bishop_pair(board):
    score = 0
    for color in ['white', 'black']:
        bishop_count = sum(
            1 for x in range(8) for y in range(8)
            if (board[x][y] == 'B' and color == 'white') or (board[x][y] == 'b' and color == 'black')
        )
        if bishop_count >= 2:
            bonus = 50
            score += bonus if color == 'white' else -bonus
    return score
def evaluate(board):
    white_moves = get_all_moves(board, 'White')
    black_moves = get_all_moves(board, 'Black')
    score = 0
    for x in range(8):
        for y in range(8):
            piece = board[x][y]
            if piece == " ":
                continue
            moves = generate_piece_moves(board, x, y)
            value = piece_values[piece.upper()]
            pst_value = get_pst_val(piece, (x, y), 'White' if piece.isupper() else 'Black')
            #if piece.lower() == "q" and not is_endgame(board):
            #    score -= 5
            if piece.lower() == "k":
                if piece.isupper():
                    score += pst_value
                else:
                    score -= pst_value
                continue
            if piece.isupper():
                score += value
                score += pst_value
                for move in moves:
                    if move in black_moves and move not in white_moves:
                        score -= value * 2
            else:
                score -= value
                score -= pst_value
                for move in moves:
                    if move in white_moves and move not in black_moves:
                        score += value * 2
    #pawn_structure_score = evaluate_pawn_structure(board)
    #score += pawn_structure_score
    #king_safety_score = evaluate_king_safety(board)
    #score += king_safety_score
    #bishop_pair_score = evaluate_bishop_pair(board)
    #score += bishop_pair_score
    mobility_score = (len(white_moves) - len(black_moves)) // 4
    score += mobility_score
    if is_checkmate(board, "white"):
        score = piece_values["K"] - 10 * piece_values["Q"]
    if is_checkmate(board, "black"):
        score = piece_values["K"] + 10 * piece_values["Q"]
    return score
