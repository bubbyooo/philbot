import csv
import hashlib
from pathlib import Path
from typing import Optional

import chess
import chess.pgn

PROF_USERNAME = "Dr_Dragon"  #username, change this and download path bellow to match 

IN_PGN = Path.home() / "Downloads" / "Dr_Dragon_all_games.pgn"
OUT_CSV = Path("Dr_Dragon_moves_dataset.csv")
BAD_PGN_LOG = Path("bad_pgn_chunks.log")


def norm_user(u: Optional[str]) -> str:
    return (u or "").strip().lower()


def game_id_from_headers(headers) -> str:
    """
    Stable-ish id from common headers. If URL exists, include it.
    """
    url = headers.get("Link") or headers.get("Site") or ""
    parts = [
        headers.get("Date", ""),
        headers.get("White", ""),
        headers.get("Black", ""),
        headers.get("Result", ""),
        url,
    ]
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:12]


def parse_prof_color(white: str, black: str, prof: str) -> Optional[str]:
    if norm_user(white) == prof:
        return "white"
    if norm_user(black) == prof:
        return "black"
    return None


def normalize_time_control(tc: str) -> str:
    return (tc or "").strip()


def extract_basic_result(headers, prof_color: str) -> str:
    """
    Returns W/L/D from professor perspective when possible.
    """
    res = (headers.get("Result") or "").strip()
    if res not in {"1-0", "0-1", "1/2-1/2"}:
        return ""
    if res == "1/2-1/2":
        return "D"
    if prof_color == "white":
        return "W" if res == "1-0" else "L"
    if prof_color == "black":
        return "W" if res == "0-1" else "L"
    return ""


def main():
    prof = PROF_USERNAME.strip().lower()

    if not IN_PGN.exists():
        raise FileNotFoundError(f"Could not find PGN file: {IN_PGN.resolve()}")

    rows_written = 0
    games_seen = 0
    games_used = 0
    games_failed = 0


    with IN_PGN.open("r", encoding="utf-8", errors="replace") as pgn_f, \
         OUT_CSV.open("w", newline="", encoding="utf-8") as out_f, \
         BAD_PGN_LOG.open("w", encoding="utf-8") as log_f:

        writer = csv.DictWriter(out_f, fieldnames=[
            "game_id",
            "date",
            "time_control",
            "rated",
            "rules",
            "white_username",
            "black_username",
            "professor_color",
            "result_prof_perspective",
            "ply",
            "fen",
            "move_uci",
            "move_san",
        ])
        writer.writeheader()

        while True:
            start_pos = pgn_f.tell()

            try:
                game = chess.pgn.read_game(pgn_f)
            except Exception as e:
                games_failed += 1
                pgn_f.seek(start_pos)
                chunk = pgn_f.read(4000)
                log_f.write(f"\n--- PARSE ERROR at byte {start_pos} ---\n")
                log_f.write(f"{type(e).__name__}: {e}\n")
                log_f.write(chunk)
                log_f.write("\n--- END CHUNK ---\n")

                
                pgn_f.seek(start_pos + max(1, len(chunk)))
                continue

            if game is None:
                break

            games_seen += 1
            headers = game.headers

            white = headers.get("White", "")
            black = headers.get("Black", "")
            prof_color = parse_prof_color(white, black, prof)

            if prof_color is None:
                continue

            games_used += 1
            gid = game_id_from_headers(headers)

            date = (headers.get("Date") or "").strip()
            time_control = normalize_time_control(headers.get("TimeControl") or "")
            rated = (headers.get("Rated") or "").strip()
            rules = (headers.get("Rules") or "").strip()
            res_prof = extract_basic_result(headers, prof_color)

            board = game.board()
            ply = 0

            for move in game.mainline_moves():
                ply += 1

                professors_turn = (
                    (prof_color == "white" and board.turn == chess.WHITE) or
                    (prof_color == "black" and board.turn == chess.BLACK)
                )

                if professors_turn:
                    fen_before = board.fen()
                    move_uci = move.uci()
                    move_san = board.san(move)

                    writer.writerow({
                        "game_id": gid,
                        "date": date,
                        "time_control": time_control,
                        "rated": rated,
                        "rules": rules,
                        "white_username": white,
                        "black_username": black,
                        "professor_color": prof_color,
                        "result_prof_perspective": res_prof,
                        "ply": ply,
                        "fen": fen_before,
                        "move_uci": move_uci,
                        "move_san": move_san,
                    })
                    rows_written += 1

                board.push(move)

    print(f"PGN read:        {IN_PGN.resolve()}")
    print(f"CSV written:     {OUT_CSV.resolve()}")
    print(f"Bad PGN log:     {BAD_PGN_LOG.resolve()}")
    print(f"Games seen:      {games_seen}")
    print(f"Games used:      {games_used} (professor present)")
    print(f"Games failed:    {games_failed} (parse errors; see log)")
    print(f"Rows (moves):    {rows_written} (professor moves only)")


if __name__ == "__main__":
    main()
