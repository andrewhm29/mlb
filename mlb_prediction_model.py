#!/usr/bin/env python3
"""
MLB Prediction Model - Versión con Datos Locales
=================================================
Usa CSV locales para no descargar todo cada vez.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from mlb_api import MlbStatsClient, _to_float, _to_int

DATA_DIR = Path(__file__).parent / "data"
GAMES_CSV = DATA_DIR / "games_history.csv"
STATS_CSV = DATA_DIR / "teams_stats.csv"

TEAM_IDS = {
    "ari": 109, "atl": 144, "bal": 110, "bos": 111, "chc": 112,
    "cws": 145, "cin": 113, "cle": 114, "col": 115, "det": 116,
    "hou": 117, "kc": 118, "laa": 108, "lad": 119, "mia": 146,
    "mil": 158, "min": 142, "nym": 121, "nyy": 147, "oak": 133,
    "phi": 143, "pit": 134, "sd": 135, "sf": 137, "sea": 136,
    "stl": 138, "tb": 139, "tex": 140, "tor": 141, "was": 120
}

TEAM_NAMES = {v: k for k, v in TEAM_IDS.items()}

@dataclass
class TeamStats:
    team_id: int
    name: str = ""
    abbreviation: str = ""
    season: int = datetime.now().year
    
    runs_scored_avg: float = 0.0
    runs_allowed_avg: float = 0.0
    hits_avg: float = 0.0
    home_runs: float = 0.0
    strikeouts: float = 0.0
    walks: float = 0.0
    obp: float = 0.0
    slg: float = 0.0
    ops: float = 0.0
    
    era: float = 0.0
    whip: float = 0.0
    k_per_nine: float = 0.0
    bb_per_nine: float = 0.0
    
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.5
    streak: str = ""
    last_10: str = ""
    home_record: str = ""
    away_record: str = ""
    
    bullpen_era: float = 0.0
    recent_form: str = ""
    
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    
    run_differential: int = 0
    games_played: int = 0
    
    last_updated: str = ""


GAMES_CSV_FIELDS = [
    "game_pk", "date", "season", "game_type", "status",
    "away_team_id", "away_team_name", "away_team_abbr", "away_score",
    "home_team_id", "home_team_name", "home_team_abbr", "home_score",
    "away_pitcher_id", "away_pitcher_name",
    "home_pitcher_id", "home_pitcher_name",
    "venue", "game_time",
]
PITCHER_STATS_CSV = DATA_DIR / "pitcher_seasons.csv"
PITCHER_STATS_FIELDS = [
    "season", "player_id", "name", "era", "whip", "ip", "games_started",
]


def ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)


def init_games_csv():
    ensure_data_dir()
    if not GAMES_CSV.exists():
        with open(GAMES_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(GAMES_CSV_FIELDS)


def init_stats_csv():
    ensure_data_dir()
    if not STATS_CSV.exists():
        with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "team_id", "name", "abbreviation", "season",
                "runs_scored_avg", "runs_allowed_avg", "hits_avg", "home_runs",
                "strikeouts", "walks", "obp", "slg", "ops",
                "era", "whip", "k_per_nine", "bb_per_nine",
                "wins", "losses", "win_pct", "streak", "last_10",
                "home_record", "away_record", "bullpen_era",
                "recent_form", "home_wins", "home_losses",
                "away_wins", "away_losses", "run_differential",
                "games_played", "last_updated"
            ])


def load_games_from_csv() -> List[Dict]:
    init_games_csv()
    games = []
    with open(GAMES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            games.append(row)
    return games


def save_game_to_csv(game: Dict):
    init_games_csv()
    games = load_games_from_csv()
    
    exists = any(g.get("game_pk") == str(game.get("game_pk")) for g in games)
    
    if not exists:
        with open(GAMES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                game.get("game_pk", ""),
                game.get("date", ""),
                game.get("season", ""),
                game.get("game_type", ""),
                game.get("status", ""),
                game.get("away_team_id", ""),
                game.get("away_team_name", ""),
                game.get("away_team_abbr", ""),
                game.get("away_score", ""),
                game.get("home_team_id", ""),
                game.get("home_team_name", ""),
                game.get("home_team_abbr", ""),
                game.get("home_score", ""),
                game.get("away_pitcher_id", ""),
                game.get("away_pitcher_name", ""),
                game.get("home_pitcher_id", ""),
                game.get("home_pitcher_name", ""),
                game.get("venue", ""),
                game.get("game_time") or game.get("time", ""),
            ])


def save_games_history(games: List[Dict], merge: bool = True) -> int:
    """Escribe el historial de partidos (2010+) en data/games_history.csv."""
    ensure_data_dir()
    by_pk: Dict[str, Dict] = {}
    if merge and GAMES_CSV.exists():
        for row in load_games_from_csv():
            pk = str(row.get("game_pk") or "")
            if pk:
                by_pk[pk] = row
    for game in games:
        pk = str(game.get("game_pk") or "")
        if not pk:
            continue
        prev = by_pk.get(pk) or {}
        by_pk[pk] = {
            "game_pk": pk,
            "date": game.get("date", "") or prev.get("date", ""),
            "season": game.get("season", "") or prev.get("season", ""),
            "game_type": game.get("game_type", "") or prev.get("game_type", ""),
            "status": game.get("status", "") or prev.get("status", ""),
            "away_team_id": game.get("away_team_id", "") or prev.get("away_team_id", ""),
            "away_team_name": game.get("away_team_name", "") or prev.get("away_team_name", ""),
            "away_team_abbr": game.get("away_team_abbr", "") or prev.get("away_team_abbr", ""),
            "away_score": game.get("away_score", "") if game.get("away_score") not in (None, "") else prev.get("away_score", ""),
            "home_team_id": game.get("home_team_id", "") or prev.get("home_team_id", ""),
            "home_team_name": game.get("home_team_name", "") or prev.get("home_team_name", ""),
            "home_team_abbr": game.get("home_team_abbr", "") or prev.get("home_team_abbr", ""),
            "home_score": game.get("home_score", "") if game.get("home_score") not in (None, "") else prev.get("home_score", ""),
            "away_pitcher_id": game.get("away_pitcher_id") or prev.get("away_pitcher_id", ""),
            "away_pitcher_name": game.get("away_pitcher_name") or prev.get("away_pitcher_name", ""),
            "home_pitcher_id": game.get("home_pitcher_id") or prev.get("home_pitcher_id", ""),
            "home_pitcher_name": game.get("home_pitcher_name") or prev.get("home_pitcher_name", ""),
            "venue": game.get("venue", "") or prev.get("venue", ""),
            "game_time": game.get("game_time") or game.get("time") or prev.get("game_time", ""),
        }
    rows = sorted(by_pk.values(), key=lambda r: (str(r.get("date") or ""), str(r.get("game_pk") or "")))
    with open(GAMES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GAMES_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def download_history(from_year: int = 2010, to_year: Optional[int] = None) -> int:
    to_year = to_year or datetime.now().year
    if from_year > to_year:
        print("❌ from-year no puede ser mayor que to-year")
        return 1
    print()
    print("=" * 60)
    print(f"📥 HISTORIAL MLB Stats API  {from_year} → {to_year}")
    print("=" * 60)
    client = MlbStatsClient(timeout=90)
    all_final: List[Dict] = []
    for year in range(from_year, to_year + 1):
        print(f"  ⬇️  Temporada {year}...", flush=True)
        games = client.schedule_season(year)
        finals = [g for g in games if g.get("status") == "F"]
        print(f"     {len(finals)} partidos finalizados")
        all_final.extend(finals)
        time.sleep(0.35)
    total = save_games_history(all_final, merge=True)
    with_sp = sum(
        1
        for g in all_final
        if g.get("home_pitcher_id") and g.get("away_pitcher_id")
    )
    print(f"\n✅ Guardado: {total} partidos en {GAMES_CSV}")
    print(f"   Abridores en esta descarga: {with_sp}/{len(all_final)}")

    print("\n📥 ERA de abridores por temporada (para el modelo as-of)...")
    ensure_data_dir()
    pitcher_rows: List[Dict] = []
    for year in range(from_year - 1, to_year + 1):
        print(f"  ⬇️  Starters {year}...", flush=True)
        pitcher_rows.extend(client.season_starters(year))
        time.sleep(0.25)
    with open(PITCHER_STATS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PITCHER_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(pitcher_rows)
    print(f"✅ {len(pitcher_rows)} filas pitcher → {PITCHER_STATS_CSV}")
    return 0


def load_stats_from_csv() -> Dict[int, TeamStats]:
    init_stats_csv()
    stats = {}
    with open(STATS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = TeamStats(
                team_id=int(row["team_id"]),
                name=row["name"],
                abbreviation=row["abbreviation"],
                season=int(row["season"]) if row["season"] else datetime.now().year,
                runs_scored_avg=float(row["runs_scored_avg"]) if row["runs_scored_avg"] else 0,
                runs_allowed_avg=float(row["runs_allowed_avg"]) if row["runs_allowed_avg"] else 0,
                hits_avg=float(row["hits_avg"]) if row["hits_avg"] else 0,
                home_runs=float(row["home_runs"]) if row["home_runs"] else 0,
                strikeouts=float(row["strikeouts"]) if row["strikeouts"] else 0,
                walks=float(row["walks"]) if row["walks"] else 0,
                obp=float(row["obp"]) if row["obp"] else 0,
                slg=float(row["slg"]) if row["slg"] else 0,
                ops=float(row["ops"]) if row["ops"] else 0,
                era=float(row["era"]) if row["era"] else 0,
                whip=float(row["whip"]) if row["whip"] else 0,
                k_per_nine=float(row["k_per_nine"]) if row["k_per_nine"] else 0,
                bb_per_nine=float(row["bb_per_nine"]) if row["bb_per_nine"] else 0,
                wins=int(row["wins"]) if row["wins"] else 0,
                losses=int(row["losses"]) if row["losses"] else 0,
                win_pct=float(row["win_pct"]) if row["win_pct"] else 0.5,
                streak=row.get("streak", ""),
                last_10=row.get("last_10", ""),
                home_record=row.get("home_record", ""),
                away_record=row.get("away_record", ""),
                bullpen_era=float(row["bullpen_era"]) if row["bullpen_era"] else 0,
                recent_form=row.get("recent_form", ""),
                home_wins=int(row["home_wins"]) if row["home_wins"] else 0,
                home_losses=int(row["home_losses"]) if row["home_losses"] else 0,
                away_wins=int(row["away_wins"]) if row["away_wins"] else 0,
                away_losses=int(row["away_losses"]) if row["away_losses"] else 0,
                run_differential=int(float(row["run_differential"])) if row["run_differential"] else 0,
                games_played=int(row["games_played"]) if row["games_played"] else 0,
                last_updated=row.get("last_updated", "")
            )
            stats[ts.team_id] = ts
    return stats


def save_stats_to_csv(stats: Dict[int, TeamStats]):
    init_stats_csv()
    with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "team_id", "name", "abbreviation", "season",
            "runs_scored_avg", "runs_allowed_avg", "hits_avg", "home_runs",
            "strikeouts", "walks", "obp", "slg", "ops",
            "era", "whip", "k_per_nine", "bb_per_nine",
            "wins", "losses", "win_pct", "streak", "last_10",
            "home_record", "away_record", "bullpen_era",
            "recent_form", "home_wins", "home_losses",
            "away_wins", "away_losses", "run_differential",
            "games_played", "last_updated"
        ])
        
        for team_id, ts in stats.items():
            writer.writerow([
                ts.team_id, ts.name, ts.abbreviation, ts.season,
                ts.runs_scored_avg, ts.runs_allowed_avg, ts.hits_avg, ts.home_runs,
                ts.strikeouts, ts.walks, ts.obp, ts.slg, ts.ops,
                ts.era, ts.whip, ts.k_per_nine, ts.bb_per_nine,
                ts.wins, ts.losses, ts.win_pct, ts.streak, ts.last_10,
                ts.home_record, ts.away_record, ts.bullpen_era,
                ts.recent_form, ts.home_wins, ts.home_losses,
                ts.away_wins, ts.away_losses, ts.run_differential,
                ts.games_played, ts.last_updated
            ])


class MLBPredictor:
    def __init__(self, season: int = None):
        self.api = MlbStatsClient()
        self.session = self.api.session
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.season = season or datetime.now().year
        self.cache_stats = None
        self._hitting: Optional[Dict[int, Dict]] = None
        self._pitching: Optional[Dict[int, Dict]] = None
        self._hitting_l30: Optional[Dict[int, Dict]] = None
        self._standings: Optional[Dict[int, Dict]] = None
        self._pitcher_cache: Dict[int, Dict] = {}
        self._ml_bundle = None
        self._ml_tried = False
        self._ml_snap = None
        self._ml_as_of = None

    def _get(self, endpoint: str, params: dict = None) -> dict:
        return self.api.get(endpoint, params)

    def _ensure_season_tables(self) -> None:
        if self._hitting is not None and self._pitching is not None and self._standings is not None:
            return
        print("  ⬇️  MLB Stats API: temporada + últimos 30 días + standings...")
        self._hitting = self.api.season_group_stats(self.season, "hitting")
        self._pitching = self.api.season_group_stats(self.season, "pitching")
        self._standings = self.api.standings(self.season)
        end = datetime.now()
        start = end - timedelta(days=30)
        self._hitting_l30 = self.api.season_group_stats(
            self.season,
            "hitting",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
    
    def get_team_abbreviation(self, team_id: int) -> str:
        if team_id in TEAM_NAMES:
            return TEAM_NAMES[team_id].upper()
        for abbr, tid in TEAM_IDS.items():
            if tid == team_id:
                return abbr.upper()
        return "UNK"
    
    def get_games_for_date(self, date_str: str) -> List[Dict]:
        games = self.api.schedule(date=date_str)
        for game in games:
            hid = game.get("home_team_id")
            aid = game.get("away_team_id")
            if not game.get("home_team_abbr") and hid:
                game["home_team_abbr"] = self.get_team_abbreviation(int(hid))
            if not game.get("away_team_abbr") and aid:
                game["away_team_abbr"] = self.get_team_abbreviation(int(aid))
        return games
    
    def get_game_live_details(self, game_pk: int) -> Dict:
        data = self.api.boxscore(game_pk)
        
        details = {
            "away_hits": 0,
            "away_errors": 0,
            "home_hits": 0,
            "home_errors": 0,
            "winning_pitcher_name": "",
            "winning_pitcher_wins": 0,
            "winning_pitcher_era": 0.0,
            "losing_pitcher_name": "",
            "losing_pitcher_losses": 0,
            "losing_pitcher_era": 0.0,
            "save_pitcher_name": "",
        }
        
        if data.get("teams"):
            away = data["teams"].get("away", {})
            home = data["teams"].get("home", {})
            
            details["away_hits"] = away.get("teamStats", {}).get("batting", {}).get("hits", 0)
            details["away_errors"] = away.get("teamStats", {}).get("fielding", {}).get("errors", 0)
            details["home_hits"] = home.get("teamStats", {}).get("batting", {}).get("hits", 0)
            details["home_errors"] = home.get("teamStats", {}).get("fielding", {}).get("errors", 0)
        
        import re
        for team_key in ["away", "home"]:
            team = data.get("teams", {}).get(team_key, {})
            for pid, pdata in team.get("players", {}).items():
                person = pdata.get("person", {})
                full_name = person.get("fullName", "")
                pitch_stats = pdata.get("stats", {}).get("pitching", {})
                note = pitch_stats.get("note", "")
                
                if "(W," in note:
                    details["winning_pitcher_name"] = full_name
                    w_match = re.search(r'W,\s*(\d+)', note)
                    l_match = re.search(r'L,\s*(\d+)', note)
                    if w_match:
                        details["winning_pitcher_wins"] = int(w_match.group(1))
                    if l_match:
                        details["winning_pitcher_losses"] = int(l_match.group(1))
                
                if "(L," in note and not details["losing_pitcher_name"]:
                    details["losing_pitcher_name"] = full_name
                    w_match = re.search(r'W,\s*(\d+)', note)
                    l_match = re.search(r'L,\s*(\d+)', note)
                    if w_match:
                        details["winning_pitcher_wins"] = int(w_match.group(1))
                    if l_match:
                        details["losing_pitcher_losses"] = int(l_match.group(1))
        
        return details
    
    def get_team_stats(self, team_id: int) -> Optional[TeamStats]:
        if self.cache_stats and team_id in self.cache_stats:
            return self.cache_stats[team_id]

        self._ensure_season_tables()
        hitting = (self._hitting or {}).get(team_id) or {}
        pitching = (self._pitching or {}).get(team_id) or {}
        standing = (self._standings or {}).get(team_id) or {}

        gp = _to_int(hitting.get("gamesPlayed") or standing.get("games_played"))
        runs = _to_float(hitting.get("runs"))
        runs_allowed = _to_float(pitching.get("runs"))
        hits = _to_float(hitting.get("hits"))

        ts = TeamStats(
            team_id=team_id,
            name=standing.get("name") or hitting.get("_team_name") or "",
            abbreviation=self.get_team_abbreviation(team_id),
            season=self.season,
            runs_scored_avg=(runs / gp) if gp else 0.0,
            runs_allowed_avg=(runs_allowed / gp) if gp else 0.0,
            hits_avg=(hits / gp) if gp else 0.0,
            home_runs=_to_float(hitting.get("homeRuns")),
            strikeouts=_to_float(hitting.get("strikeOuts")),
            walks=_to_float(hitting.get("baseOnBalls")),
            obp=_to_float(hitting.get("obp")),
            slg=_to_float(hitting.get("slg")),
            ops=_to_float(hitting.get("ops")),
            era=_to_float(pitching.get("era")),
            whip=_to_float(pitching.get("whip")),
            k_per_nine=_to_float(pitching.get("strikeoutsPer9Inn")),
            bb_per_nine=_to_float(pitching.get("walksPer9Inn")),
            wins=_to_int(standing.get("wins")),
            losses=_to_int(standing.get("losses")),
            win_pct=_to_float(standing.get("win_pct"), 0.5),
            streak=str(standing.get("streak") or ""),
            last_10=str(standing.get("last_10") or ""),
            home_wins=_to_int(standing.get("home_wins")),
            home_losses=_to_int(standing.get("home_losses")),
            away_wins=_to_int(standing.get("away_wins")),
            away_losses=_to_int(standing.get("away_losses")),
            run_differential=_to_int(standing.get("run_differential")),
            games_played=gp,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        ts.home_record = f"{ts.home_wins}-{ts.home_losses}"
        ts.away_record = f"{ts.away_wins}-{ts.away_losses}"
        ts.recent_form = ts.last_10
        ts.bullpen_era = ts.era

        if self.cache_stats is None:
            self.cache_stats = {}
        self.cache_stats[team_id] = ts
        return ts
    
    def get_all_team_stats(self) -> Dict[int, TeamStats]:
        print("  📊 Descargando estadísticas de temporada (MLB Stats API)...")
        self._ensure_season_tables()
        stats = {}
        for _abbr, team_id in TEAM_IDS.items():
            ts = self.get_team_stats(team_id)
            if ts:
                stats[team_id] = ts
        self.cache_stats = stats
        save_stats_to_csv(stats)
        print(f"  ✅ Stats de {len(stats)} equipos guardadas")
        return stats
    
    def get_recent_games(self, team_id: int, days: int = 10) -> List[Dict]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        games = self.api.schedule(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            team_id=team_id,
        )
        return [g for g in games if g.get("status") == "F"]

    def get_pitcher_stats(self, person_id: Optional[int]) -> Dict:
        if not person_id:
            return {}
        pid = int(person_id)
        if pid in self._pitcher_cache:
            return self._pitcher_cache[pid]
        stats = self.api.pitcher_season_stats(pid, self.season)
        self._pitcher_cache[pid] = stats
        return stats
    
    def get_head_to_head(
        self,
        home_id: int,
        away_id: int,
        as_of: Optional[str] = None,
    ) -> Tuple[int, int, int, List[str]]:
        """H2H desde el CSV local (2010+). as_of = YYYY-MM-DD para no usar el futuro."""
        h2h_home_wins = 0
        h2h_away_wins = 0
        total = 0
        results: List[str] = []

        def _tid(value) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        local = load_games_from_csv()
        rows = local if local else self.get_recent_games(home_id, days=30)

        for game in rows:
            if str(game.get("status") or "") != "F":
                continue
            ds = str(game.get("date") or "")
            if as_of and ds and ds >= as_of:
                continue
            hid = _tid(game.get("home_team_id"))
            aid = _tid(game.get("away_team_id"))
            if {hid, aid} != {home_id, away_id}:
                continue
            try:
                home_score = int(float(game.get("home_score") or 0))
                away_score = int(float(game.get("away_score") or 0))
            except (TypeError, ValueError):
                continue
            total += 1
            home_side_won = home_score > away_score
            if hid == home_id:
                if home_side_won:
                    h2h_home_wins += 1
                    results.append("W")
                else:
                    h2h_away_wins += 1
                    results.append("L")
            else:
                if home_side_won:
                    h2h_away_wins += 1
                    results.append("L")
                else:
                    h2h_home_wins += 1
                    results.append("W")

        return h2h_home_wins, h2h_away_wins, total, results[:5]
    
    def predict_game(
        self,
        home_team_id: int,
        away_team_id: int,
        game: Optional[Dict] = None,
    ) -> Dict:
        home_stats = self.get_team_stats(home_team_id)
        away_stats = self.get_team_stats(away_team_id)
        
        if not home_stats or not away_stats:
            return {"predicted_winner": "N/A", "confidence": 0}

        game = game or {}
        home_p = self.get_pitcher_stats(game.get("home_pitcher_id"))
        away_p = self.get_pitcher_stats(game.get("away_pitcher_id"))
        home_p_name = game.get("home_pitcher_name") or ""
        away_p_name = game.get("away_pitcher_name") or ""
        
        home_score = 50
        away_score = 50
        
        if home_stats.runs_scored_avg > away_stats.runs_scored_avg:
            diff = home_stats.runs_scored_avg - away_stats.runs_scored_avg
            home_score += diff * 8
            away_score -= diff * 5
        
        if home_stats.runs_allowed_avg < away_stats.runs_allowed_avg:
            diff = away_stats.runs_allowed_avg - home_stats.runs_allowed_avg
            home_score += diff * 8
            away_score -= diff * 5
        
        if home_stats.home_wins > away_stats.away_wins:
            home_score += 3
        
        if home_stats.era < away_stats.era:
            home_score += 5
        if home_stats.era < 3.5:
            home_score += 3
        
        if home_stats.ops > away_stats.ops:
            home_score += 4
        
        if home_stats.run_differential > 0 and away_stats.run_differential < 0:
            home_score += 5

        home_l30 = _to_float(((self._hitting_l30 or {}).get(home_team_id) or {}).get("ops"))
        away_l30 = _to_float(((self._hitting_l30 or {}).get(away_team_id) or {}).get("ops"))
        if home_l30 and away_l30:
            home_score += (home_l30 - away_l30) * 25

        if home_p.get("ip", 0) >= 10 and away_p.get("ip", 0) >= 10:
            era_gap = away_p["era"] - home_p["era"]
            whip_gap = away_p["whip"] - home_p["whip"]
            home_score += era_gap * 7
            home_score += whip_gap * 10
        
        h2h_home, h2h_away, h2h_total, h2h_results = self.get_head_to_head(
            home_team_id,
            away_team_id,
            as_of=game.get("date"),
        )
        if h2h_total > 0:
            h2h_pct = h2h_home / h2h_total
            home_score += (h2h_pct - 0.5) * 20
        
        total = home_score + away_score
        home_prob = home_score / total if total > 0 else 0.5
        away_prob = away_score / total if total > 0 else 0.5

        try:
            from mlb_ml import load_bundle, snapshot_before, matchup_features, p_home_win

            if not self._ml_tried:
                self._ml_tried = True
                self._ml_bundle = load_bundle()
            if self._ml_bundle:
                as_of = str(game.get("date") or datetime.now().strftime("%Y-%m-%d"))
                if self._ml_as_of != as_of:
                    self._ml_snap = snapshot_before(as_of)
                    self._ml_as_of = as_of
                hera = home_p.get("era") if home_p.get("ip", 0) >= 10 else None
                aera = away_p.get("era") if away_p.get("ip", 0) >= 10 else None
                hid_p = game.get("home_pitcher_id")
                aid_p = game.get("away_pitcher_id")
                try:
                    hid_p = int(hid_p) if hid_p else None
                    aid_p = int(aid_p) if aid_p else None
                except (TypeError, ValueError):
                    hid_p, aid_p = None, None
                feats = matchup_features(
                    self._ml_snap,
                    int(home_team_id),
                    int(away_team_id),
                    as_of,
                    str(game.get("venue") or ""),
                    str(game.get("game_type") or "R"),
                    home_pitcher_id=hid_p,
                    away_pitcher_id=aid_p,
                    live_home_era=hera,
                    live_away_era=aera,
                )
                home_prob = p_home_win(self._ml_bundle, feats)
                away_prob = 1.0 - home_prob
                if self._ml_bundle.get("ridge_total") is not None:
                    from mlb_ml import predict_totals, over_under, round_line, team_context, win_factor_rows

                    totals = predict_totals(self._ml_bundle, feats)
                    line = game.get("ou_line")
                    if line is None:
                        line = getattr(self, "ou_line", None)
                    if line is None:
                        line = round_line(float(feats.get("park_rpg") or totals["total"]))
                    ou = over_under(totals["total"], float(line), totals["sigma"])
                    ha = self.get_team_abbreviation(home_team_id)
                    aa = self.get_team_abbreviation(away_team_id)
                    home_fac, away_fac = win_factor_rows(
                        self._ml_bundle, feats, ha, aa, home_prob
                    )
                    game["_ml_home_fac"] = home_fac
                    game["_ml_away_fac"] = away_fac
                    # stash on self via return dict below using locals
                    self._last_totals = {
                        "home_runs": totals["home_runs"],
                        "away_runs": totals["away_runs"],
                        "total": totals["total"],
                        "ou": ou,
                        "home_ctx": team_context(self._ml_snap, int(home_team_id)),
                        "away_ctx": team_context(self._ml_snap, int(away_team_id)),
                        "home_fac": home_fac,
                        "away_fac": away_fac,
                        "feats": feats,
                    }
        except Exception as exc:
            print(f"  ⚠️  Modelo ML no usado: {exc}")
        
        prob_diff = abs(home_prob - away_prob)
        # Escala de confianza tipo app: siempre entre 50% y 95%.
        # Si el juego es parejo (50/50), la confianza muestra 50%.
        # Mientras mayor sea la diferencia entre probabilidades, sube hasta 95%.
        confidence = min(95, 50 + (prob_diff * 112.5))
        
        predicted_winner = home_stats.name if home_prob > away_prob else away_stats.name
        predicted_abbr = self.get_team_abbreviation(home_team_id) if home_prob > away_prob else self.get_team_abbreviation(away_team_id)
        
        factors_for_winner = []
        factors_for_loser = []
        
        if home_prob > away_prob:
            winner_stats = home_stats
            loser_stats = away_stats
            winner_abbr = self.get_team_abbreviation(home_team_id)
            loser_abbr = self.get_team_abbreviation(away_team_id)
        else:
            winner_stats = away_stats
            loser_stats = home_stats
            winner_abbr = self.get_team_abbreviation(away_team_id)
            loser_abbr = self.get_team_abbreviation(home_team_id)
        
        if winner_stats.runs_scored_avg > loser_stats.runs_scored_avg:
            diff = winner_stats.runs_scored_avg - loser_stats.runs_scored_avg
            factors_for_winner.append(
                f"{winner_abbr} bateo en buen momento: anota {winner_stats.runs_scored_avg:.1f} carreras/juego "
                f"(ventaja de +{diff:.1f} en producción ofensiva) ⭐⭐"
            )
        
        if winner_stats.runs_allowed_avg < loser_stats.runs_allowed_avg:
            diff = loser_stats.runs_allowed_avg - winner_stats.runs_allowed_avg
            factors_for_winner.append(
                f"{winner_abbr} pitcheo más sólido: permite {winner_stats.runs_allowed_avg:.1f} carreras/juego "
                f"(mejora de {diff:.1f} vs rival) ⭐⭐⭐"
            )
        
        if winner_stats.home_wins > loser_stats.away_wins + 2:
            factors_for_winner.append(
                f"{winner_abbr} se hace fuerte en su parque: récord local {winner_stats.home_wins}-{winner_stats.home_losses} ⭐"
            )
        
        if winner_stats.run_differential > loser_stats.run_differential + 5:
            factors_for_winner.append(
                f"{winner_abbr} domina el diferencial de carreras: {winner_stats.run_differential:+.0f} vs "
                f"{loser_stats.run_differential:+.0f} ⭐⭐"
            )

        winner_p = home_p if home_prob > away_prob else away_p
        loser_p = away_p if home_prob > away_prob else home_p
        winner_p_name = home_p_name if home_prob > away_prob else away_p_name
        loser_p_name = away_p_name if home_prob > away_prob else home_p_name
        if winner_p.get("ip", 0) >= 10 and loser_p.get("ip", 0) >= 10 and winner_p["era"] + 0.25 < loser_p["era"]:
            factors_for_winner.append(
                f"Abridor {winner_p_name or winner_abbr}: ERA {winner_p['era']:.2f}, WHIP {winner_p['whip']:.2f} "
                f"vs {loser_p_name or loser_abbr} ERA {loser_p['era']:.2f} ⭐⭐⭐"
            )
        elif loser_p.get("ip", 0) >= 10 and winner_p.get("ip", 0) >= 10 and loser_p["era"] + 0.25 < winner_p["era"]:
            factors_for_loser.append(
                f"Mejor abridor del lado de {loser_abbr}: {loser_p_name} ERA {loser_p['era']:.2f} ⭐⭐"
            )
        
        if loser_stats.runs_scored_avg > 4.5:
            factors_for_loser.append(
                f"{loser_abbr} también produce carreras ({loser_stats.runs_scored_avg:.1f}/juego), puede responder con el bate ⭐"
            )
        
        if loser_stats.home_wins < loser_stats.home_losses:
            factors_for_loser.append(
                f"{loser_abbr} llega con irregularidad reciente: récord de {loser_stats.wins}-{loser_stats.losses} ⭐"
            )
        
        home_abbr = self.get_team_abbreviation(home_team_id)
        away_abbr = self.get_team_abbreviation(away_team_id)
        
        out = {
            "predicted_winner": predicted_winner,
            "predicted_abbr": predicted_abbr,
            "confidence": confidence,
            "home_prob": home_prob * 100,
            "away_prob": away_prob * 100,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "factors_for_winner": factors_for_winner[:4],
            "factors_for_loser": factors_for_loser[:4],
            "winner_abbr": winner_abbr,
            "loser_abbr": loser_abbr,
            "home_pitcher_name": home_p_name,
            "away_pitcher_name": away_p_name,
            "home_pitcher_era": home_p.get("era"),
            "away_pitcher_era": away_p.get("era"),
            "home_pitcher_whip": home_p.get("whip"),
            "away_pitcher_whip": away_p.get("whip"),
        }
        extra = getattr(self, "_last_totals", None)
        if extra:
            out.update(
                {
                    "pred_home_runs": extra["home_runs"],
                    "pred_away_runs": extra["away_runs"],
                    "pred_total": extra["total"],
                    "ou_line": extra["ou"]["line"],
                    "ou_pick": extra["ou"]["pick"],
                    "ou_p_over": extra["ou"]["p_over"] * 100,
                    "ou_p_under": extra["ou"]["p_under"] * 100,
                    "ou_diff": extra["ou"]["diff"],
                    "home_ctx": extra["home_ctx"],
                    "away_ctx": extra["away_ctx"],
                    "ml_home_fac": extra["home_fac"],
                    "ml_away_fac": extra["away_fac"],
                    "ml_feats": extra["feats"],
                }
            )
            self._last_totals = None
        return out
    
    def format_prediction(self, game: Dict, prediction: Dict, details: Dict = None) -> str:
        is_final = game.get("status") == "F"
        away_abbr = prediction.get("away_abbr", "") or self.get_team_abbreviation(game.get("away_team_id", 0))
        home_abbr = prediction.get("home_abbr", "") or self.get_team_abbreviation(game.get("home_team_id", 0))
        pred_abbr = prediction.get("predicted_abbr", "")
        winner = prediction.get("winner_abbr", pred_abbr)
        loser = prediction.get("loser_abbr", "")
        away_prob = float(prediction.get("away_prob", 0) or 0)
        home_prob = float(prediction.get("home_prob", 0) or 0)
        line = "─" * 80
        out: List[str] = [line, f"⚾️  PARTIDO: {away_abbr} @ {home_abbr}"]
        if is_final:
            out.append(
                f"🏁 FINAL: {away_abbr} {game.get('away_score', '')} - {game.get('home_score', '')} {home_abbr}"
            )
        away_pn = prediction.get("away_pitcher_name") or game.get("away_pitcher_name") or ""
        home_pn = prediction.get("home_pitcher_name") or game.get("home_pitcher_name") or ""
        if away_pn or home_pn:
            ae = prediction.get("away_pitcher_era")
            he = prediction.get("home_pitcher_era")
            ae_s = f" ERA {ae:.2f}" if ae else ""
            he_s = f" ERA {he:.2f}" if he else ""
            out.append(f"🧢 Abridores: {away_pn or 'TBD'}{ae_s}  vs  {home_pn or 'TBD'}{he_s}")
        out.append(line)
        out.append("")
        out.append(f"🎯 EL MODELO DICE: {pred_abbr} GANA")
        out.append(f"   Confianza: {prediction.get('confidence', 0):.0f}%")
        out.append(f"   {away_abbr}: {away_prob:.0f}% chance  |  {home_abbr}: {home_prob:.0f}% chance")
        out.append("")

        def _block(title: str, rows: List[str], ctx: List[str]) -> None:
            out.append(title)
            out.append(line)
            i = 1
            for row in rows:
                out.append(f"  {i}. {row}")
                i += 1
            for row in ctx:
                out.append(f"  {i}. • {row}")
                i += 1
            if i == 1:
                out.append("  Sin factores claros")
            out.append("")

        home_fac = list(prediction.get("ml_home_fac") or [])
        away_fac = list(prediction.get("ml_away_fac") or [])
        if not home_fac and not away_fac:
            home_fac = list(prediction.get("factors_for_winner") or [])
            away_fac = list(prediction.get("factors_for_loser") or [])
        hctx = prediction.get("home_ctx") or {}
        actx = prediction.get("away_ctx") or {}
        ctx_home = []
        ctx_away = []
        if hctx.get("home_n", 0) >= 3:
            ctx_home.append(
                f"{home_abbr} en casa (últimos {int(hctx['home_n'])} como local): "
                f"ganó {int(hctx['home_wins'])} de {int(hctx['home_n'])}, "
                f"anota {hctx['home_rs']:.1f} y permite {hctx['home_ra']:.1f} R/juego"
            )
        if actx.get("away_n", 0) >= 3:
            ctx_away.append(
                f"{away_abbr} de visitante (últimos {int(actx['away_n'])}): "
                f"ganó {int(actx['away_wins'])} de {int(actx['away_n'])}, "
                f"anota {actx['away_rs']:.1f} y permite {actx['away_ra']:.1f} R/juego"
            )
        if hctx and actx:
            cross_h = f"{home_abbr} anota {hctx.get('rs', 0):.1f} R/juego y {away_abbr} permite {actx.get('ra', 0):.1f} (dif {hctx.get('rs', 0) - actx.get('ra', 0):+.1f})"
            cross_a = f"{away_abbr} anota {actx.get('rs', 0):.1f} R/juego y {home_abbr} permite {hctx.get('ra', 0):.1f} (dif {actx.get('rs', 0) - hctx.get('ra', 0):+.1f})"
            if winner == home_abbr:
                ctx_home.append(cross_h)
                ctx_away.append(cross_a)
            else:
                ctx_away.append(cross_a)
                ctx_home.append(cross_h)

        win_rows = home_fac if winner == home_abbr else away_fac
        lose_rows = away_fac if winner == home_abbr else home_fac
        win_ctx = ctx_home if winner == home_abbr else ctx_away
        lose_ctx = ctx_away if winner == home_abbr else ctx_home
        _block(f"✅ ¿POR QUÉ FAVORECE A {winner}?", win_rows[:5], win_ctx[:2])
        _block(f"❌ ¿QUÉ FAVORECE A {loser}?", lose_rows[:5], lose_ctx[:2])
        out.append("  [pp = puntos porcentuales de probabilidad que aporta el factor · • = dato de contexto, no del modelo]")
        hw = int(hctx.get("season_w") or 0)
        hl = int(hctx.get("season_l") or 0)
        if hw + hl >= 10:
            if hctx.get("season_wp", 0) >= 0.45 and actx.get("season_wp", 0) >= 0.45:
                out.append("  ℹ️  Ambos equipos con récord de contender (≥45% de victorias)")
        out.append("")

        pred_total = prediction.get("pred_total")
        if pred_total is not None:
            hr = float(prediction.get("pred_home_runs") or 0)
            ar = float(prediction.get("pred_away_runs") or 0)
            ou_line = float(prediction.get("ou_line") or 8.5)
            pick = prediction.get("ou_pick") or "UNDER"
            p_over = float(prediction.get("ou_p_over") or 50)
            p_under = float(prediction.get("ou_p_under") or 50)
            diff = float(prediction.get("ou_diff") or 0)
            out.append("🎯 PREDICCIÓN DE CARRERAS:")
            out.append(f"   {home_abbr}: {hr:.1f} carreras")
            out.append(f"   {away_abbr}: {ar:.1f} carreras")
            out.append(f"   Total: {float(pred_total):.1f} carreras")
            out.append("")
            side_p = p_over if pick == "OVER" else p_under
            out.append(f"🧾 O/U {ou_line:g}: {pick}  ({pick.lower()} prob: {side_p:.0f}%)")
            hrs = float(hctx.get("rs") or 4.5)
            hra = float(hctx.get("ra") or 4.5)
            ars = float(actx.get("rs") or 4.5)
            ara = float(actx.get("ra") or 4.5)
            side_home = 0.5 * hrs + 0.5 * ara
            side_away = 0.5 * ars + 0.5 * hra
            out.append(
                f"   → Ofensiva local {home_abbr} ({hrs:.1f}) vs Defensa visitante {away_abbr} ({ara:.1f}) "
                f"[últimos 5 juegos] → ~{side_home:.1f} R esperadas de ese lado"
            )
            out.append(
                f"   → Ofensiva visitante {away_abbr} ({ars:.1f}) vs Defensa local {home_abbr} ({hra:.1f}) "
                f"[últimos 5 juegos] → ~{side_away:.1f} R esperadas de ese lado"
            )
            vs = "por encima" if diff > 0 else "por debajo"
            out.append(f"   → Total predicho {float(pred_total):.1f} {vs} de la línea ({diff:+.1f})")
            out.append("")
            away_p01 = away_prob / 100.0
            out.append(
                f"DATA|{away_abbr}|{home_abbr}||{away_p01:.4f}|{float(pred_total):.1f}|{ou_line:g}|{pick}"
            )

        if details and is_final:
            out.append("")
            if details.get("winning_pitcher_name"):
                out.append(f"🟢 GANA: {details['winning_pitcher_name']}")
            if details.get("losing_pitcher_name"):
                out.append(f"🔴 PIERDE: {details['losing_pitcher_name']}")
        out.append("")
        return "\n".join(out)


def sync_daily():
    print()
    print("=" * 60)
    print("📥 SINCRONIZANDO DATOS DIARIOS")
    print("=" * 60)
    
    ensure_data_dir()
    init_games_csv()
    init_stats_csv()
    
    model = MLBPredictor()
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n📅 Fecha de ayer: {yesterday}")
    print(f"📅 Fecha de hoy: {today}")
    
    print("\n⬇️  Descargando partidos de AYER...")
    games = model.get_games_for_date(yesterday)
    
    final_games = [g for g in games if g.get("status") == "F"]
    print(f"  ✅ {len(final_games)} partidos finalizados encontrados")
    
    for game in final_games:
        print(f"  💾 Guardando: {game.get('away_team_abbr')} @ {game.get('home_team_abbr')}")
        save_game_to_csv(game)
        
        details = model.get_game_live_details(game.get("game_pk"))
        game.update(details)
    
    print("\n📊 Descargando estadísticas de equipos...")
    model.get_all_team_stats()
    
    print("\n" + "=" * 60)
    print("✅ SINCRONIZACIÓN COMPLETA")
    print("=" * 60)


def run_predictions_for_date(
    model: MLBPredictor,
    date_to_use: str,
    save_csv: bool = True,
    ou_line: float | None = None,
):
    if ou_line is not None:
        model.ou_line = ou_line
    from pathlib import Path
    
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    if date_to_use not in [today, yesterday]:
        print(f"\n⚠️  Fecha {date_to_use} no está en caché. Descargando...")
        games = model.get_games_for_date(date_to_use)
    else:
        games = model.get_games_for_date(date_to_use)
    
    if not games:
        print("❌ No hay partidos programados para esta fecha.")
        return
    
    print(f"\n✅ Partidos detectados: {len(games)}")
    print("\n📥 Cargando modelo...")
    print(f"\n⏳ Calculando {len(games)} predicción(es)...")
    print()
    print("=" * 80)
    print("⚾ PREDICCIONES - CALENDARIO")
    print("=" * 80)
    print()
    
    predictions = []
    for game in games:
        pred = model.predict_game(
            home_team_id=game["home_team_id"],
            away_team_id=game["away_team_id"],
            game=game,
        )
        pred["game_time"] = game.get("time", "")
        
        details = None
        if game.get("status") == "F":
            details = model.get_game_live_details(game["game_pk"])
        
        predictions.append((game, pred, details))
    
    for game, pred, details in predictions:
        print(model.format_prediction(game, pred, details))
        print()
    
    if save_csv:
        csv_path = Path(__file__).with_name(f"predictions_{date_to_use}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "date", "game_pk", "status",
                "away_team", "home_team",
                "predicted_winner", "predicted_abbr", "confidence",
                "away_prob", "home_prob",
                "away_score", "home_score",
                "away_pitcher", "home_pitcher",
                "away_pitcher_era", "home_pitcher_era",
                "pred_away_runs", "pred_home_runs", "pred_total",
                "ou_line", "ou_pick", "ou_p_over",
                "game_time", "venue",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for game, pred, _ in predictions:
                writer.writerow({
                    "date": game.get("date", ""),
                    "game_pk": game.get("game_pk", ""),
                    "status": game.get("status", ""),
                    "away_team": game.get("away_team_name", ""),
                    "home_team": game.get("home_team_name", ""),
                    "predicted_winner": pred.get("predicted_winner", ""),
                    "predicted_abbr": pred.get("predicted_abbr", ""),
                    "confidence": round(pred.get("confidence", 0), 2),
                    "away_prob": round(pred.get("away_prob", 0), 2),
                    "home_prob": round(pred.get("home_prob", 0), 2),
                    "away_score": game.get("away_score", ""),
                    "home_score": game.get("home_score", ""),
                    "away_pitcher": pred.get("away_pitcher_name") or game.get("away_pitcher_name", ""),
                    "home_pitcher": pred.get("home_pitcher_name") or game.get("home_pitcher_name", ""),
                    "away_pitcher_era": pred.get("away_pitcher_era") or "",
                    "home_pitcher_era": pred.get("home_pitcher_era") or "",
                    "pred_away_runs": round(pred.get("pred_away_runs") or 0, 2) if pred.get("pred_total") is not None else "",
                    "pred_home_runs": round(pred.get("pred_home_runs") or 0, 2) if pred.get("pred_total") is not None else "",
                    "pred_total": round(pred.get("pred_total") or 0, 2) if pred.get("pred_total") is not None else "",
                    "ou_line": pred.get("ou_line", ""),
                    "ou_pick": pred.get("ou_pick", ""),
                    "ou_p_over": round(pred.get("ou_p_over") or 0, 2) if pred.get("ou_p_over") is not None else "",
                    "game_time": pred.get("game_time", ""),
                    "venue": game.get("venue", ""),
                })
        
        print(f"\n📄 CSV exportado: {csv_path}")


def get_date_input(prompt: str = "Fecha (YYYY-MM-DD) [ENTER = hoy]: ") -> str:
    date_str = input(prompt).strip()
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        print("❌ Formato inválido. Usa YYYY-MM-DD")
        return get_date_input(prompt)


def show_menu():
    print()
    print("=" * 60)
    print("⚾  MLB PREDICT - MENÚ PRINCIPAL")
    print("=" * 60)
    print()
    print("  1 — 📅 Ver partidos de HOY")
    print("  2 — 📆 Ver partidos de AYER")
    print("  3 — 📅 Elegir fecha específica")
    print()
    print("  4 — 🔄 Sincronizar datos (ayer + stats)")
    print()
    print("  5 — 📄 Exportar CSV de hoy")
    print("  6 — 📄 Exportar CSV de ayer")
    print()
    print("  A — ⬇️  Descargar historial (2010 → hoy)")
    print("  T — 🏋️  Entrenar modelo (as-of, holdout 2026)")
    print()
    print("  0 — 🚪 Salir")
    print()
    print("=" * 60)


def cmd_menu(_args: argparse.Namespace | None = None) -> int:
    while True:
        show_menu()
        choice = input("Elige una opción: ").strip().upper()

        if choice == "1":
            date_to_use = datetime.now().strftime("%Y-%m-%d")
            model = MLBPredictor()
            run_predictions_for_date(model, date_to_use)
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "2":
            date_to_use = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            model = MLBPredictor()
            run_predictions_for_date(model, date_to_use)
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "3":
            date_to_use = get_date_input()
            model = MLBPredictor()
            run_predictions_for_date(model, date_to_use)
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "4":
            sync_daily()
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "5":
            date_to_use = datetime.now().strftime("%Y-%m-%d")
            model = MLBPredictor()
            print(f"\n📄 Exportando CSV para {date_to_use}...")
            run_predictions_for_date(model, date_to_use, save_csv=True)
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "6":
            date_to_use = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            model = MLBPredictor()
            print(f"\n📄 Exportando CSV para {date_to_use}...")
            run_predictions_for_date(model, date_to_use, save_csv=True)
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "A":
            download_history(2010)
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "T":
            cmd_train(argparse.Namespace(holdout=str(datetime.now().year)))
            input("\n⏎ Presiona ENTER para continuar...")

        elif choice == "0":
            print("\n🚪 ¡Hasta luego!")
            break

        else:
            print("\n❌ Opción inválida. Intenta de nuevo.")
            input("\n⏎ Presiona ENTER para continuar...")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from_year = getattr(args, "from_year", None)
    if from_year:
        return download_history(int(from_year), getattr(args, "to_year", None))
    sync_daily()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from mlb_ml import train_and_save, print_metrics

    hold = str(getattr(args, "holdout", None) or datetime.now().year)
    print(f"\n🏋️  Entrenando (holdout temporada {hold})...")
    bundle = train_and_save(holdout_season=hold)
    print_metrics(bundle)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    date_to_use = getattr(args, "date", None) or datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(date_to_use, "%Y-%m-%d")
    except ValueError:
        print("❌ Formato inválido. Usa --date YYYY-MM-DD")
        return 1
    model = MLBPredictor()
    ou = getattr(args, "ou_line", None)
    run_predictions_for_date(model, date_to_use, ou_line=ou)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLB Predict — calendario y menú")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser(
        "download",
        help="Sincronizar ayer, o bajar historial 2010+ con --from-year",
    )
    d.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Bajar temporadas desde este año (mínimo recomendado: 2010)",
    )
    d.add_argument(
        "--to-year",
        type=int,
        default=None,
        help="Hasta este año (por defecto el año actual)",
    )
    d.set_defaults(func=cmd_download)

    pr = sub.add_parser("predict", help="Predecir partidos del calendario (fecha dada)")
    pr.add_argument(
        "--date",
        type=str,
        default=None,
        help="YYYY-MM-DD (por defecto hoy en local)",
    )
    pr.add_argument(
        "--ou-line",
        type=float,
        default=None,
        help="Línea de over/under (carreras). Por defecto: total típico del parque (.5)",
    )
    pr.set_defaults(func=cmd_predict)

    tr = sub.add_parser("train", help="Entrenar logística as-of y evaluar holdout")
    tr.add_argument(
        "--holdout",
        type=str,
        default=None,
        help="Temporada de test (por defecto el año actual)",
    )
    tr.set_defaults(func=cmd_train)

    sub.add_parser("menu", help="Menú interactivo con todas las opciones").set_defaults(
        func=cmd_menu
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return cmd_menu()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
