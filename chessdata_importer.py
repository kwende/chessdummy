import fire
import zstandard
import sqlite3
import os
import re
import sqlite3
import chess
import datetime

# https://python-zstandard.readthedocs.io/en/latest/decompressor.html

def fen_to_vector(fen: str):
    vec = []
    for p in fen:
        if p == "/":
            pass
        elif p == ' ':
            break
        elif p.isalpha():
            vec.append(ord(p))
        else:
            for _ in range(0, int(p)):
                vec.append(0)
    return vec

def get_value(input : str) -> str:
    matched = re.match(r'\[(.+) \"(.+)\"', input)
    if matched:
        return matched.group(1), matched.group(2)
    else:
        return None, None

def import_data(lichess_data_file: str, output_path: str, sqlite_path: str) -> None:

    # decompress the data from the standard .zstd format used by lichess
    # only do so if required output file doesn't already exist.
    if not os.path.isfile(output_path):
        decompressor = zstandard.ZstdDecompressor()
        with open(lichess_data_file, 'rb') as input, open(output_path, 'wb') as output:
            decompressor.copy_stream(input, output)

    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    with open(output_path) as fin:
        site = None
        white = None
        black = None
        white_elo = None
        black_elo = None

        for line in fin:    
            k, v = get_value(line)
            if not k is None:
                if k == "Site":
                    site = v
                elif k == "White":
                    white = v
                elif k == "Black":
                    black = v
                elif k == "WhiteElo":
                    white_elo = int(v)
                elif k == "BlackElo":
                    black_elo = int(v)
            elif line.startswith('1.'):
                cur.execute("insert into game (url, white, black, whiteelo, blackelo, moves) values (?,?,?,?,?,?)",
                            (site, white, black, white_elo, black_elo, line))
                con.commit()
    con.close()

def generate_boards_from_pgns(sqlite_path: str, max_elo: int):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    cur.execute("select count(1) from game where whiteelo < ? and blackelo < ?", (max_elo, max_elo))
    number_of_games = cur.fetchone()[0]
    
    cur.execute("select id, moves from game where whiteelo < ? and blackelo < ?", (max_elo,max_elo))

    game_board_cur = con.cursor()
    games_processed = 0

    start_time = datetime.datetime.now()
    for (id, pgn) in cur:
        moves = re.findall(r'\d\.{1,3}\s([A-Za-z0-9\-\=\+]+)', pgn)
        move_num = 1

        # go through all the moves for this game to see if we encounter an illegal
        # move before attempting to save. Shouldn't happen, but it seems to still. 
        # this prevents us from saving up to an illegal move and having a spoiled game. 
        to_inserts = []
        next_moves = []
        next_fens = []
        from_square_to_square = []

        board = chess.Board()
        try:
            start_fen = board.fen()
            start_fen_vec = bytearray(fen_to_vector(start_fen))
            to_inserts.append((0, start_fen, start_fen_vec, id, None))

            for move in moves:
                move_data = board.push_san(move)
                
                fen = board.fen()
                fen_vec = bytearray(fen_to_vector(fen))
                to_inserts.append((move_num, fen, fen_vec, id, move))

                next_moves.append(move)
                next_fens.append(fen_vec)
                from_square_to_square.append((move_data.from_square, move_data.to_square))

                move_num += 1
        except chess.IllegalMoveError:
            to_inserts = []

        for i in range(0, len(to_inserts)):
            next_move_and_vec = (None,None, None, None)
            if len(next_moves) > i + 1:
                from_square, to_square = from_square_to_square[i]
                next_move_and_vec = (next_moves[i],next_fens[i], from_square, to_square)
            to_insert = next_move_and_vec + to_inserts[i]
            game_board_cur.execute('insert into gameposition (movefromhere, nextvector, fromsquare, tosquare, movenumber, fen, vector, gameid, movetohere) values (?,?,?,?,?,?,?,?,?)', to_insert)
        games_processed += 1

        if games_processed % 100 == 0:
            con.commit()
            delta = datetime.datetime.now() - start_time
            seconds_to_complete = delta.seconds * (number_of_games / (games_processed * 1.0))
            print(f"{games_processed} games processed in {delta.seconds} seconds. Should be done by {start_time + datetime.timedelta(seconds=seconds_to_complete)}")

    

if __name__ == '__main__':
    #generate_boards_from_pgns("E:/chess.db", 1000)
    fire.Fire()