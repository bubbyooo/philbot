import chess
from regression import train_model
from MoveChoice import choose_move_logreg

"""
This is the main file that runs the bot. It creates the board, 
and prompts the human to make a move. It then prints the board and the result.
"""

def prompt_human_move(board: chess.Board) -> chess.Move:
    while True:
        s = input("Your move (UCI like e2e4, or 'quit'): ").strip()
        if s.lower() in ("quit", "exit"):
            raise SystemExit
        try:
            move = chess.Move.from_uci(s)
        except ValueError:
            print("Invalid format. Use UCI like e2e4, g1f3, e7e8q.")
            continue
        if move not in board.legal_moves:
            print("Illegal move. Try again.")
            continue
        return move

if __name__ == "__main__":
    model = train_model("data/raw/Dr_Dragon_moves_dataset.csv")

    board = chess.Board()
    human_color = chess.WHITE

    print(board, "\n")

    while not board.is_game_over():
        if board.turn == human_color:
            move = prompt_human_move(board)
            board.push(move)
        else:
            bot_move = choose_move_logreg(board, model)
            print(f"Bot plays: {bot_move.uci()}")
            board.push(bot_move)

        print(board, "\n")

    print("Game over:", board.result())
