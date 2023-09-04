from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import cycle, islice
import torch
import sqlite3
import numpy as np

class ChessDataset(IterableDataset):
    def __init__(self, db_path):
        self.results = []
        self.db_path = db_path
        self.db_connection = sqlite3.connect(db_path)
        self.cursor = self.db_connection.cursor()
        self.cursor.execute("select min(id) from GamePosition")
        self.min_value = self.cursor.fetchone()[0]
        self.cursor.execute("select max(id) from GamePosition")
        self.max_value = self.cursor.fetchone()[0]

    def __iter__(self):
        return self
    
    def __next__(self):

        while True:
            if len(self.results) == 0:
                ids = [str(x) for x in np.random.randint(self.min_value, self.max_value+1, 1000)]
                resultString = ','.join(ids)
                self.cursor.execute(f"select Vector,FromSquare,ToSquare from GamePosition where id in ({resultString})")
                for result in self.cursor.fetchall():
                    self.results.append(result)

            vector, from_square, to_square = self.results.pop(0)
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

    print('done')
