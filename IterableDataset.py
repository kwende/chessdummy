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
                self.cursor.execute(f"select Vector,FromSquare,ToSquare,MoveNumber from GamePosition where id in ({resultString})")
                for result in self.cursor.fetchall():
                    self.results.append(result)

            vector, from_square, to_square, move_number = self.results.pop(0)
            if vector is None or from_square is None or to_square is None: 
                continue
            answer_vec = np.zeros((64,))
            answer_vec[from_square] = 1.0
            answer_vec[to_square] = 2.0
            # record this as a white move or a black move
            whites_move = move_number == 0 or move_number % 2 == 0
            if whites_move:
                board = np.zeros((9,9))
            else:
                board = np.ones((9,9))

            board[:8,:8] = np.reshape(torch.tensor([x for x in vector]), (8,8))
            board = torch.tensor(board.astype(np.float32))
            return np.reshape(board, (1,9,9)), torch.tensor(answer_vec)


if __name__ == "__main__":
    
    ds = ChessDataset('e:/chess.db')
    loader = DataLoader(ds, batch_size=1)

    for x, y in loader:
        print(x)

    print('done')
