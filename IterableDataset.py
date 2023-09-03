from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import cycle, islice
import torch
import sqlite3
import numpy as np

class ChessDataset(IterableDataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_connection = sqlite3.connect(db_path)
        self.cursor = self.db_connection.cursor()

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            self.cursor.execute("select Vector,FromSquare,ToSquare from GamePosition where rowid = ((abs(random()) % ((select max(rowid) from GamePosition) - (select min(Id) from GamePosition)))  + (select min(Id) from GamePosition))")

            vector, from_square, to_square = self.cursor.fetchone()
            if vector is None or from_square is None or to_square is None: 
                continue
            answer_vec = np.zeros((128,))
            answer_vec[from_square] = 1.0
            answer_vec[to_square + 64] = 1.0
            return np.reshape(torch.tensor([float(x) for x in vector]), (1,8,8)), torch.tensor(answer_vec)


if __name__ == "__main__":
    
    ds = ChessDataset('e:/chess.db')
    loader = DataLoader(ds, batch_size=1)

    for x, y in loader:
        print(x)
