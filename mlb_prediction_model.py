#!/usr/bin/env python3
"""
MLB Prediction Model - Versión con Datos Locales
=================================================
Usa CSV locales para no descargar todo cada vez.
"""

import csv
import os
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path

API_BASE = "https://statsapi.mlb.com/api/v1"

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


def ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)


def init_games_csv():
    ensure_data_dir()
    if not GAMES_CSV.exists():
        with open(GAMES_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_pk", "date", "status",
                "away_team_id", "away_team_name", "away_team_abbr", "away_score",
                "home_team_id", "home_team_name", "home_team_abbr", "home_score",
                "venue", "game_time"
            ])


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
                game.get("status", ""),
                game.get("away_team_id", ""),
                game.get("away_team_name", ""),
                game.get("away_team_abbr", ""),
                game.get("away_score", ""),
                game.get("home_team_id", ""),
                game.get("home_team_name", ""),
                game.get("home_team_abbr", ""),
                game.get("home_score", ""),
                game.get("venue", ""),
                game.get("game_time", "")
            ])


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
                run_differential=int(row["run_differential"]) if row["run_differential"] else 0,
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
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MLB-Predictor/1.0"})
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.season = season or datetime.now().year
        self.cache_stats = None
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{API_BASE}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  ⚠️  API Error: {e}")
            return {}
    
    def get_team_abbreviation(self, team_id: int) -> str:
        if team_id in TEAM_NAMES:
            return TEAM_NAMES[team_id].upper()
        for abbr, tid in TEAM_IDS.items():
            if tid == team_id:
                return abbr.upper()
        return "UNK"
    
    def get_games_for_date(self, date_str: str) -> List[Dict]:
        data = self._get("schedule", {"date": date_str, "sportId": 1})
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get("statusCode", "")
                if status in ["S", "P", "PRE", "F", "I"]:
                    games.append({
                        "game_pk": game.get("gamePk"),
                        "date": date_entry.get("date"),
                        "status": status,
                        "status_detailed": game.get("status", {}).get("detailedState", ""),
                        "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                        "away_team_name": game.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                        "away_team_abbr": game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", ""),
                        "away_score": game.get("teams", {}).get("away", {}).get("score"),
                        "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                        "home_team_name": game.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                        "home_team_abbr": game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", ""),
                        "home_score": game.get("teams", {}).get("home", {}).get("score"),
                        "venue": game.get("venue", {}).get("name"),
                        "venue_city": game.get("venue", {}).get("location", {}).get("city", ""),
                        "venue_state": game.get("venue", {}).get("location", {}).get("state", ""),
                        "time": game.get("gameDate", "")[:16] if game.get("gameDate") else "",
                        "current_inning": game.get("linescore", {}).get("currentInning", ""),
                        "inning_state": game.get("linescore", {}).get("inningState", ""),
                    })
        return games
    
    def get_game_live_details(self, game_pk: int) -> Dict:
        data = self._get(f"game/{game_pk}/boxscore")
        
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
        
        standings = self._get("standings", {"season": self.season, "sportId": 1})
        
        for record in standings.get("records", []):
            for team_record in record.get("teamRecords", []):
                if team_record.get("team", {}).get("id") == team_id:
                    stats = team_record
                    break
        
        if not any(team_record.get("team", {}).get("id") == team_id 
                   for record in standings.get("records", []) 
                   for team_record in record.get("teamRecords", [])):
            team_data = self._get(f"teams/{team_id}")
            team_record = team_data.get("teams", [{}])[0]
            stats = {"team": team_record}
        
        for record in standings.get("records", []):
            for team_record in record.get("teamRecords", []):
                if team_record.get("team", {}).get("id") == team_id:
                    stats = team_record
        
        team_info = stats.get("team", {})
        
        ts = TeamStats(
            team_id=team_id,
            name=team_info.get("name", ""),
            abbreviation=team_info.get("abbreviation", ""),
            season=self.season,
        )
        
        splits = stats.get("splits", [])
        if splits:
            bs = splits[0].get("stat", {})
            ts.runs_scored_avg = float(bs.get("runsScoredPerGame", 0))
            ts.runs_allowed_avg = float(bs.get("runsAllowedPerGame", 0))
            ts.hits_avg = float(bs.get("hits", 0)) / max(1, int(bs.get("gamesPlayed", 1)))
            ts.home_runs = float(bs.get("homeRuns", 0))
            ts.strikeouts = float(bs.get("strikeOuts", 0))
            ts.walks = float(bs.get("baseOnBalls", 0))
            ts.obp = float(bs.get("obp", 0))
            ts.slg = float(bs.get("slg", 0))
            ts.ops = float(bs.get("ops", 0))
            ts.era = float(bs.get("era", 0))
            ts.whip = float(bs.get("whip", 0))
            ts.k_per_nine = float(bs.get("strikeoutsPer9Inn", 0))
            ts.bb_per_nine = float(bs.get("walksPer9Inn", 0))
        
        ts.wins = stats.get("wins", 0)
        ts.losses = stats.get("losses", 0)
        ts.win_pct = ts.wins / max(1, ts.wins + ts.losses)
        ts.streak = stats.get("streak", {}).get("streakCode", "")
        ts.games_played = ts.wins + ts.losses
        
        ts.run_differential = ts.runs_scored_avg * ts.games_played - ts.runs_allowed_avg * ts.games_played
        
        last_10 = stats.get("last10", {})
        ts.last_10 = f"{last_10.get('wins', 0)}-{last_10.get('losses', 0)}"
        
        records = stats.get("records", {})
        ts.home_record = records.get("homeRecord", "")
        ts.away_record = records.get("roadRecord", "")
        
        if ts.home_record:
            parts = ts.home_record.split("-")
            if len(parts) >= 2:
                ts.home_wins = int(parts[0].split("-")[0] if "-" not in parts[0] else 0)
                try:
                    ts.home_wins = int(parts[0])
                    ts.home_losses = int(parts[1].split("-")[0])
                except:
                    pass
        
        if ts.away_record:
            parts = ts.away_record.split("-")
            if len(parts) >= 2:
                try:
                    ts.away_wins = int(parts[0])
                    ts.away_losses = int(parts[1].split("-")[0])
                except:
                    pass
        
        ts.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if self.cache_stats is None:
            self.cache_stats = {}
        self.cache_stats[team_id] = ts
        
        return ts
    
    def get_all_team_stats(self) -> Dict[int, TeamStats]:
        print("  📊 Cargando estadísticas de equipos...")
        
        cached_stats = load_stats_from_csv()
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if cached_stats:
            for team_id, stats in cached_stats.items():
                if stats.last_updated == today_str:
                    print(f"  ✅ {stats.name}: stats ya actualizadas")
                    if self.cache_stats is None:
                        self.cache_stats = {}
                    self.cache_stats[team_id] = stats
        
        stats = {}
        for abbr, team_id in TEAM_IDS.items():
            if self.cache_stats and team_id in self.cache_stats:
                stats[team_id] = self.cache_stats[team_id]
                continue
            
            print(f"  ⬇️  Descargando stats de {abbr.upper()}...")
            ts = self.get_team_stats(team_id)
            if ts:
                stats[team_id] = ts
        
        self.cache_stats = stats
        save_stats_to_csv(stats)
        
        return stats
    
    def get_recent_games(self, team_id: int, days: int = 10) -> List[Dict]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        games = []
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            day_games = self.get_games_for_date(date_str)
            for g in day_games:
                if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id:
                    if g.get("status") == "F":
                        games.append(g)
            current += timedelta(days=1)
        
        return games
    
    def get_head_to_head(self, home_id: int, away_id: int) -> Tuple[int, int, int, List[str]]:
        recent = self.get_recent_games(home_id, days=30)
        
        h2h_home_wins = 0
        h2h_away_wins = 0
        total = 0
        results = []
        
        for game in recent:
            if game.get("status") != "F":
                continue
            
            if game.get("home_team_id") == home_id and game.get("away_team_id") == away_id:
                total += 1
                home_score = game.get("home_score", 0)
                away_score = game.get("away_score", 0)
                
                if home_score > away_score:
                    h2h_home_wins += 1
                    results.append("W")
                else:
                    h2h_away_wins += 1
                    results.append("L")
        
        return h2h_home_wins, h2h_away_wins, total, results[:5]
    
    def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
        home_stats = self.get_team_stats(home_team_id)
        away_stats = self.get_team_stats(away_team_id)
        
        if not home_stats or not away_stats:
            return {"predicted_winner": "N/A", "confidence": 0}
        
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
        
        h2h_home, h2h_away, h2h_total, h2h_results = self.get_head_to_head(home_team_id, away_team_id)
        if h2h_total > 0:
            h2h_pct = h2h_home / h2h_total
            home_score += (h2h_pct - 0.5) * 20
        
        total = home_score + away_score
        home_prob = home_score / total if total > 0 else 0.5
        away_prob = away_score / total if total > 0 else 0.5
        
        confidence = abs(home_prob - 0.5) * 100 * 2
        
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
            factors_for_winner.append(f"{winner_abbr} buena ofensiva: {winner_stats.runs_scored_avg:.1f} carreras/juego ⭐⭐")
        
        if winner_stats.runs_allowed_avg < loser_stats.runs_allowed_avg:
            diff = loser_stats.runs_allowed_avg - winner_stats.runs_allowed_avg
            factors_for_winner.append(f"{winner_abbr} defensa sólida: permite {winner_stats.runs_allowed_avg:.1f} carreras/juego ⭐⭐⭐")
        
        if winner_stats.home_wins > loser_stats.away_wins + 2:
            factors_for_winner.append(f"{winner_abbr} fuerte en casa: {winner_stats.home_wins}-{winner_stats.home_losses} ⭐")
        
        if winner_stats.era < loser_stats.era:
            factors_for_winner.append(f"{winner_abbr} mejor pitcheo: ERA {winner_stats.era:.2f} vs {loser_stats.era:.2f} ⭐⭐⭐")
        
        if winner_stats.ops > loser_stats.ops + 0.05:
            factors_for_winner.append(f"{winner_abbr} mejor ofensiva: OPS {winner_stats.ops:.3f} vs {loser_stats.ops:.3f} ⭐")
        
        if loser_stats.runs_scored_avg > 4.5:
            factors_for_loser.append(f"{loser_abbr} tiene buena ofensiva pero no le alcanza ⭐")
        
        if loser_stats.home_wins < loser_stats.home_losses:
            factors_for_loser.append(f"{loser_abbr} débil en casa: {loser_stats.home_wins}-{loser_stats.home_losses} ⭐")
        
        home_abbr = self.get_team_abbreviation(home_team_id)
        away_abbr = self.get_team_abbreviation(away_team_id)
        
        return {
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
            "loser_abbr": loser_abbr
        }
    
    def format_prediction(self, game: Dict, prediction: Dict, details: Dict = None) -> str:
        is_final = game.get("status") == "F"
        
        away_abbr = prediction.get("away_abbr", "") or self.get_team_abbreviation(game.get("away_team_id", 0))
        home_abbr = prediction.get("home_abbr", "") or self.get_team_abbreviation(game.get("home_team_id", 0))
        
        game_time = game.get("time", "")
        if game_time:
            try:
                dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                time_str = dt.strftime("%-I:%M %p")
            except:
                time_str = "Por definir"
        else:
            time_str = "Por definir"
        
        output = []
        
        output.append("─" * 80)
        output.append(f"⚾️ PARTIDO: {away_abbr} @ {home_abbr}")
        output.append(f"📅 {time_str}")
        if is_final:
            away_score = game.get("away_score", "")
            home_score = game.get("home_score", "")
            output.append(f"🏁 FINAL: {away_abbr} {away_score} - {home_score} {home_abbr}")
        output.append("─" * 80)
        output.append("")
        
        pred_abbr = prediction.get("predicted_abbr", "")
        confidence = prediction.get("confidence", 0)
        
        output.append(f"🎯 EL MODELO DICE: {pred_abbr} GANA")
        output.append(f"   Confianza: {confidence:.0f}%")
        
        away_prob = prediction.get("away_prob", 0)
        home_prob = prediction.get("home_prob", 0)
        output.append(f"   {away_abbr}: {away_prob:.0f}% chance | {home_abbr}: {home_prob:.0f}% chance")
        
        output.append("")
        output.append("✅ ¿POR QUÉ FAVORECE A " + prediction.get("winner_abbr", pred_abbr) + "?")
        output.append("─" * 80)
        
        factors_winner = prediction.get("factors_for_winner", [])
        if factors_winner:
            for factor in factors_winner:
                output.append(f"  {factor}")
        else:
            output.append("  Sin factores claros")
        
        output.append("")
        output.append("❌ ¿QUÉ FAVORECE A " + prediction.get("loser_abbr", "") + "?")
        output.append("─" * 80)
        
        factors_loser = prediction.get("factors_for_loser", [])
        if factors_loser:
            for factor in factors_loser:
                output.append(f"  {factor}")
        else:
            output.append("  Sin factores en contra")
        
        if details and is_final:
            output.append("")
            output.append("─" * 80)
            output.append("📊 RESULTADO FINAL")
            output.append("─" * 80)
            
            if details.get("winning_pitcher_name"):
                wins = details.get("winning_pitcher_wins", 0)
                output.append(f"🟢 GANA: {details['winning_pitcher_name']} ({wins}-0)")
            
            if details.get("losing_pitcher_name"):
                losses = details.get("losing_pitcher_losses", 0)
                output.append(f"🔴 PIERDE: {details['losing_pitcher_name']} (0-{losses})")
        
        return "\n".join(output)


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


def run_predictions_for_date(model: MLBPredictor, date_to_use: str, save_csv: bool = True):
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
    print(f"⏳ Cargando datos de partidos...")
    
    predictions = []
    for game in games:
        pred = model.predict_game(
            home_team_id=game["home_team_id"],
            away_team_id=game["away_team_id"]
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
    print("  0 — 🚪 Salir")
    print()
    print("=" * 60)


def main():
    while True:
        show_menu()
        choice = input("Elige una opción: ").strip()
        
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
            
        elif choice == "0":
            print("\n🚪 ¡Hasta luego!")
            break
            
        else:
            print("\n❌ Opción inválida. Intenta de nuevo.")
            input("\n⏎ Presiona ENTER para continuar...")


if __name__ == "__main__":
    main()
