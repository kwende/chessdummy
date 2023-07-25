CREATE TABLE IF NOT EXISTS "Game" (
	"URL"	TEXT NOT NULL,
	"White"	TEXT NOT NULL,
	"Black"	TEXT NOT NULL,
	"WhiteElo"	INTEGER NOT NULL,
	"BlackElo"	INTEGER NOT NULL,
	"Id"	INTEGER NOT NULL,
	"Moves"	TEXT NOT NULL,
	PRIMARY KEY("Id" AUTOINCREMENT)
);
CREATE INDEX "Game_EloRatings" ON "Game" (
	"WhiteElo"	ASC,
	"BlackElo"	ASC
);
CREATE TABLE IF NOT EXISTS "GamePosition" (
	"Id"	INTEGER NOT NULL,
	"Fen"	TEXT NOT NULL,
	"Vector"	BLOB NOT NULL,
	"GameId"	INTEGER NOT NULL,
	"MoveNumber"	INTEGER NOT NULL,
	"MoveToHere"	TEXT,
	"MoveFromHere"	TEXT,
	"NextVector"	BLOB,
	"FromSquare"	INTEGER,
	"ToSquare"	INTEGER,
	PRIMARY KEY("Id" AUTOINCREMENT)
);
CREATE INDEX "GamePosition_GameId" ON "GamePosition" (
	"GameId"
);
/* No STAT tables available */
