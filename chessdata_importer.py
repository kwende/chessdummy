import fire
import zstandard
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
    games_saved = 0

    start_time = datetime.datetime.now()
    for (id, pgn) in cur:
        moves = re.findall(r'\d\.{1,3}\s([A-Za-z0-9\-\=\+]+)', pgn)
        board = chess.Board()
        move_num = 1
        for move in moves:
            board.push_san(move)
            fen = board.fen()
            fen_vec = bytearray(fen_to_vector(fen))
            game_board_cur.execute('insert into gameposition (movenumber, fen, vector, gameid, movetohere) values (?,?,?,?,?)', (move_num, fen, fen_vec, id, move))
            move_num += 1
        games_saved += 1

        if games_saved % 100 == 0:
            con.commit()
            delta = datetime.datetime.now() - start_time
            seconds_to_complete = delta.seconds * (number_of_games / (games_saved * 1.0))
            print(f"{games_saved} games saved in {delta.seconds} seconds. Should be done by {start_time + datetime.timedelta(seconds=seconds_to_complete)}")

if __name__ == '__main__':
    #generate_boads_from_pgns("E:/chess.db", 1000)
    fire.Fire()