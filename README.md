PhilBot README

The PGN parser to generate the .csv file is not in this repo.
We can supply this upon request.

To play, run:
python3 main.py

features.py     --> Extracts numerical features from a board
                    position (piece counts, turn, etc.) used as model inputs.
main.py         --> Entry point for running the chess bot; trains the model,
                    reads the current board state, and outputs a move.
MoveChoice.py   --> Selects the best legal move by evaluating all candidate
                    moves using the trained model’s predictions based on Prof. Phil's dataset.
readcsv.py      --> Loads and preprocesses chess game data from CSV files,
                    converting positions and moves into a usable training format.
                    This is only necessary to generate the .csv file.
regression.py   --> Trains and evaluates the regression model that predicts
                    similarity to Prof. Phil's moves based on extracted features.
