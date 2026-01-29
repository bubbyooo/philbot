import time
import requests
from pathlib import Path

USERNAME = "dr_dragon" #username

OUT = Path("Dr_Dragon_all_games.pgn")
USER_AGENT = "DragonBot"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def get_with_backoff(url, max_retries=10):
    delay = 2
    last_exc = None
    for _ in range(max_retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(delay)
            delay = min(delay * 2, 60)

    raise RuntimeError(f"Failed after retries: {url}\nLast error: {last_exc}")

def get_archive_month_urls(username: str):
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    r = get_with_backoff(archives_url)
    data = r.json()
    return data.get("archives", [])

archive_urls = get_archive_month_urls(USERNAME)
if not archive_urls:
    raise RuntimeError(f"No archives found for user: {USERNAME}")

print(f"Found {len(archive_urls)} archive months for {USERNAME}")

with OUT.open("w", encoding="utf-8") as f:
    game_count = 0

    for i, archive_url in enumerate(archive_urls, start=1):
        print(f"[{i}/{len(archive_urls)}] Downloading {archive_url}")
        r = get_with_backoff(archive_url)
        payload = r.json()

        month_games = 0
        for game in payload.get("games", []):
            pgn = game.get("pgn")
            if pgn:
                f.write(pgn.strip())
                f.write("\n\n") 
                game_count += 1
                month_games += 1

        print(f"  -> wrote {month_games} games from this month (total so far: {game_count})") #see how many games are lost in process

    
        time.sleep(1)

print("Wrote", OUT.resolve())
print("Total games:", game_count)
