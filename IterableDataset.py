from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import cycle, islice

import sqlite3

class ChessDataset(IterableDataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_connection = sqlite3.connect(db_path)
        self.cursor = self.db_connection.cursor()

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            self.cursor.execute("select Fen,Vector,FromSquare,ToSquare from GamePosition where rowid = abs(random()) % (select (select max(rowid) from GamePosition)+1);")

            fen, vector, from_square, to_square = self.cursor.fetchone()
            if fen is None or vector is None or from_square is None or to_square is None: 
                continue
            return fen, vector, from_square, to_square


if __name__ == "__main__":
    
    ds = ChessDataset('e:/chess.db')
    loader = DataLoader(ds, batch_size=1)

    for fen, x, y, z in loader:
        print(x)
