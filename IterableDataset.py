from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import cycle, islice

import sqlite3

class IteratableDataset(IterableDataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_connection = sqlite3.connect(db_path)
        self.cursor = self.db_connection.cursor()

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            self.cursor.execute("select Fen from GamePosition where rowid = abs(random()) % (select (select max(rowid) from GamePosition)+1);")

            result = self.cursor.fetchone()
            if result is not None:
                random_value = result[0]
                break

        return random_value
    

if __name__ == "__main__":
    
    ds = IteratableDataset('e:/chess.db')
    loader = DataLoader(ds, batch_size=4)

    for batch in islice(loader, 8):
        print(batch)