#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from functools import partial
import main
import piece

print = partial(print, flush=True)

def go_loop(hist, stop_event, max_movetime=0, max_depth=0, debug=False):
    if debug:
        print(f"Going movetime={max_movetime}, depth={max_depth}")

    if len(hist) >= 2:
        last_move = hist[-2]
        board = hist[-1]
    else:
        last_move = None
        board = hist[-1]

    computer_color = "Black" if (len(hist) // 2) % 2 == 0 else "White"

    best_move, depth_reached, nodes_searched = main.search(
        board, t_table={}, computer_color=computer_color,
        depth=max_depth, last_move=last_move
    )

    score = piece.evaluate(board)
    uci_move = convert_move_to_uci(best_move[0], best_move[1], board)

    print(f"info depth {depth_reached} score cp {score} nodes {nodes_searched}")
    print(f"bestmove {uci_move}")

def convert_move_to_uci(start_pos, end_pos, board, promotion=None):
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    piece_moved = board[start_row][start_col]
    promotion_piece = ''

    if piece_moved.lower() == 'p' and ((piece_moved.isupper() and end_row == 0) or (piece_moved.islower() and end_row == 7)):
        promotion_piece = 'q'  # Default to queen

    start_file = chr(ord('a') + start_col)
    start_rank = str(8 - start_row)
    end_file = chr(ord('a') + end_col)
    end_rank = str(8 - end_row)
    return f"{start_file}{start_rank}{end_file}{end_rank}{promotion_piece}"

def parse_fen(fen_str):
    board = []
    fen_parts = fen_str.strip().split()
    rows = fen_parts[0].split('/')
    for fen_row in rows:
        board_row = []
        for c in fen_row:
            if c.isdigit():
                board_row.extend([' '] * int(c))
            else:
                board_row.append(c)
        board.append(board_row)
    return board

def parse_uci_move(move_str):
    if len(move_str) < 4:
        return None
    col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    try:
        start_col = col_map[move_str[0]]
        start_row = 8 - int(move_str[1])
        end_col = col_map[move_str[2]]
        end_row = 8 - int(move_str[3])
        promotion = move_str[4] if len(move_str) > 4 else None
        return ((start_row, start_col), (end_row, end_col), promotion)
    except (KeyError, ValueError):
        return None

version = "abcd 2025(1.5)"

hist = [None, piece.Position(piece.initial, 0, (True, True), (True, True), (9, 8), 0)]
debug = False

with ThreadPoolExecutor(max_workers=1) as executor:
    go_future = executor.submit(lambda: None)
    do_stop_event = Event()

    while True:
        try:
            args = input().split()
            if not args:
                continue

            if args[0] == "quit":
                if not go_future.done():
                    do_stop_event.set()
                    go_future.result()
                break

            elif args[0] == "uci":
                print(f"id name {version}")
                print("uciok")

            elif args[0] == "isready":
                print("readyok")
            elif args[0] == "ucinewgame":
                hist = [None, copy.deepcopy(main.initial_board)]

            elif args[0] == "position":
                if args[1] == "startpos":
                    board = copy.deepcopy(main.initial_board)
                    moves = args[3:] if len(args) > 2 and args[2] == "moves" else []
                    hist = [None, board]
                    for move in moves:
                        parsed_move = parse_uci_move(move)
                        if parsed_move:
                            start_pos, end_pos, promotion = parsed_move
                            valid, last_move = piece.move_piece(board, start_pos, end_pos, hist[-2], sim_move=True)
                            if valid:
                                hist.append(board)
                elif args[1] == "fen":
                    fen_str = ' '.join(args[2:args.index("moves")]) if "moves" in args else ' '.join(args[2:])
                    board = parse_fen(fen_str)
                    moves = args[args.index("moves") + 1:] if "moves" in args else []
                    hist = [None, board]
                    for move in moves:
                        parsed_move = parse_uci_move(move)
                        if parsed_move:
                            start_pos, end_pos, promotion = parsed_move
                            valid, last_move = piece.move_piece(board, start_pos, end_pos, hist[-2], sim_move=True)
                            if valid:
                                hist.append(board)

            elif args[0] == "go":
                max_depth = 4
                if "depth" in args:
                    depth_index = args.index("depth") + 1
                    if depth_index < len(args):
                        max_depth = int(args[depth_index])

                do_stop_event.clear()
                go_future = executor.submit(
                    go_loop,
                    hist,
                    do_stop_event,
                    max_depth=max_depth,
                    debug=debug
                )

            elif args[0] == "stop":
                if not go_future.done():
                    do_stop_event.set()
                    go_future.result()

        except (KeyboardInterrupt, EOFError):
            if not go_future.done():
                do_stop_event.set()
                go_future.result()
            break

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
