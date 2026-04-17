#!/usr/bin/env python3
"""
MLB Prediction Model - Versión Completa
========================================
Predice TODOS los partidos del día con análisis estilo sportsbook.
"""

import csv
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

API_BASE = "https://statsapi.mlb.com/api/v1"

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
    name: str
    abbreviation: str
    team_id: int
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
    last_10: Tuple[int, int] = (0, 0)
    home_record: str = ""
    away_record: str = ""
    
    bullpen_era: float = 0.0
    recent_form: List[str] = field(default_factory=list)
    
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    
    run_differential: int = 0
    games_played: int = 0

@dataclass 
class HeadToHead:
    home_wins: int = 0
    home_losses: int = 0
    total_games: int = 0
    recent_results: List[str] = field(default_factory=list)

class MLBPredictor:
    def __init__(self, season: int = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MLB-Predictor/1.0"})
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.season = season or datetime.now().year
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{API_BASE}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            return {}
    
    def get_team_id(self, team_name: str) -> Optional[int]:
        team_name_lower = team_name.lower()
        for abbr, tid in TEAM_IDS.items():
            if abbr == team_name_lower:
                return tid
        teams = self._get("teams", {"sportId": 1})
        for team in teams.get("teams", []):
            if team_name_lower in team.get("name", "").lower() or team_name_lower in team.get("abbreviation", "").lower():
                return team["id"]
        return None
    
    def get_team_abbreviation(self, team_id: int) -> str:
        if team_id in TEAM_NAMES:
            return TEAM_NAMES[team_id].upper()
        for abbr, tid in TEAM_IDS.items():
            if tid == team_id:
                return abbr.upper()
        return "UNK"
    
    def get_team_info(self, team_id: int) -> Dict:
        teams = self._get("teams", {"sportId": 1})
        for team in teams.get("teams", []):
            if team.get("id") == team_id:
                return team
        return {}
    
    def get_schedule(self, start_date: str, end_date: str, team_id: int = None) -> List[Dict]:
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "sportId": 1
        }
        if team_id:
            params["teamId"] = team_id
        
        data = self._get("schedule", params)
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get("statusCode") == "F":
                    games.append({
                        "game_pk": game.get("gamePk"),
                        "date": date_entry.get("date"),
                        "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                        "home_team_name": game.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                        "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                        "away_team_name": game.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                        "home_score": game.get("teams", {}).get("home", {}).get("score"),
                        "away_score": game.get("teams", {}).get("away", {}).get("score"),
                        "venue": game.get("venue", {}).get("name")
                    })
        return games
    
    def get_todays_games(self) -> List[Dict]:
        return self.get_games_for_date(self.today)
    
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
        
        info = data.get("info", [])
        
        for team_key in ["away", "home"]:
            team = data.get("teams", {}).get(team_key, {})
            for pid, pdata in team.get("players", {}).items():
                person = pdata.get("person", {})
                full_name = person.get("fullName", "")
                pitch_stats = pdata.get("stats", {}).get("pitching", {})
                note = pitch_stats.get("note", "")
                
                if "(W," in note:
                    details["winning_pitcher_name"] = full_name
                    import re
                    w_match = re.search(r'W,\s*(\d+)', note)
                    l_match = re.search(r'L,\s*(\d+)', note)
                    if w_match:
                        details["winning_pitcher_wins"] = int(w_match.group(1))
                    if l_match:
                        details["winning_pitcher_losses"] = int(l_match.group(1))
                
                if "(L," in note and not details["losing_pitcher_name"]:
                    details["losing_pitcher_name"] = full_name
                    import re
                    w_match = re.search(r'W,\s*(\d+)', note)
                    l_match = re.search(r'L,\s*(\d+)', note)
                    if w_match:
                        details["losing_pitcher_wins"] = int(w_match.group(1))
                    if l_match:
                        details["losing_pitcher_losses"] = int(l_match.group(1))
        
        for item in info:
            label = item.get("label", "")
            value = item.get("value", "")
            if label in ["Save", "SAVE"]:
                details["save_pitcher_name"] = value.split(".")[0].strip() if "." in value else value
        
        return details
    
    def get_recent_games(self, team_id: int, num_games: int = 20) -> List[Dict]:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        all_games = self.get_schedule(start_date, self.today, team_id)
        
        team_games = []
        for g in all_games:
            if g["home_team_id"] == team_id or g["away_team_id"] == team_id:
                g["is_home"] = g["home_team_id"] == team_id
                team_games.append(g)
        
        return team_games[-num_games:]
    
    def get_standings(self, season: int = None) -> List[Dict]:
        if season is None:
            season = self.season
        
        params = {"leagueId": "103,104", "season": season}
        data = self._get("standings", params)
        standings_list = []
        
        for league in data.get("records", []):
            for team_rec in league.get("teamRecords", []):
                standings_list.append({
                    "team_id": team_rec.get("team", {}).get("id"),
                    "team_name": team_rec.get("team", {}).get("name"),
                    "wins": team_rec.get("leagueRecord", {}).get("wins", 0),
                    "losses": team_rec.get("leagueRecord", {}).get("losses", 0),
                    "pct": team_rec.get("leagueRecord", {}).get("percentage", 0.5),
                    "streak": team_rec.get("streak", {}).get("streak", ""),
                    "run_diff": team_rec.get("runDifferential", 0),
                    "home_wins": 0,
                    "home_losses": 0,
                    "away_wins": 0,
                    "away_losses": 0
                })
                
                records = team_rec.get("records", {})
                for rec_type in ["homeRecords", "awayRecords"]:
                    rec_list = records.get(rec_type, [])
                    if rec_list:
                        rec = rec_list[0]
                        if rec_type == "homeRecords":
                            standings_list[-1]["home_wins"] = rec.get("wins", 0)
                            standings_list[-1]["home_losses"] = rec.get("losses", 0)
                        else:
                            standings_list[-1]["away_wins"] = rec.get("wins", 0)
                            standings_list[-1]["away_losses"] = rec.get("losses", 0)
        
        return standings_list
    
    def get_team_stats(self, team_id: int, season: int = None) -> Dict:
        if season is None:
            season = self.season
        
        params = {"stats": "season", "group": "hitting,pitching", "season": season}
        data = self._get(f"teams/{team_id}/stats", params)
        result = {"hitting": {}, "pitching": {}}
        
        for stat_group in data.get("stats", []):
            group = stat_group.get("group", {}).get("displayName", "")
            if group in ["hitting", "pitching"]:
                for split in stat_group.get("splits", []):
                    result[group] = split.get("stat", {})
                    break
        
        return result
    
    def get_head_to_head(self, home_team_id: int, away_team_id: int) -> HeadToHead:
        h2h = HeadToHead()
        
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        games = self.get_schedule(start_date, self.today)
        
        team_games = [g for g in games if 
                     (g["home_team_id"] == home_team_id and g["away_team_id"] == away_team_id) or
                     (g["home_team_id"] == away_team_id and g["away_team_id"] == home_team_id)]
        
        h2h.total_games = len(team_games)
        for g in team_games:
            if g["home_team_id"] == home_team_id:
                if g["home_score"] > g["away_score"]:
                    h2h.home_wins += 1
                else:
                    h2h.home_losses += 1
                h2h.recent_results.append("W" if g["home_score"] > g["away_score"] else "L")
            else:
                if g["away_score"] > g["home_score"]:
                    h2h.home_wins += 1
                else:
                    h2h.home_losses += 1
                h2h.recent_results.append("W" if g["away_score"] > g["home_score"] else "L")
        
        h2h.recent_results = h2h.recent_results[-10:]
        return h2h
    
    def calculate_team_metrics(self, team_id: int, games: List[Dict]) -> TeamStats:
        team_info = self.get_team_info(team_id)
        
        abbr = team_info.get("abbreviation", "")
        if not abbr:
            for a, tid in TEAM_IDS.items():
                if tid == team_id:
                    abbr = a.upper()
                    break
        
        stats = TeamStats(
            name=team_info.get("name", "Unknown"),
            abbreviation=abbr,
            team_id=team_id,
            season=self.season
        )
        
        standings = self.get_standings()
        for s in standings:
            if s["team_id"] == team_id:
                stats.wins = s["wins"]
                stats.losses = s["losses"]
                stats.win_pct = s["pct"]
                stats.streak = s["streak"]
                stats.home_wins = s["home_wins"]
                stats.home_losses = s["home_losses"]
                stats.away_wins = s["away_wins"]
                stats.away_losses = s["away_losses"]
                stats.run_differential = s["run_diff"]
                stats.games_played = stats.wins + stats.losses
                break
        
        team_games = [g for g in games if g["home_team_id"] == team_id or g["away_team_id"] == team_id]
        
        if len(team_games) >= 3:
            last_3 = team_games[-3:]
            stats.runs_scored_avg = sum(
                g["home_score"] if g["is_home"] else g["away_score"] 
                for g in last_3
            ) / 3
            stats.runs_allowed_avg = sum(
                g["away_score"] if g["is_home"] else g["home_score"] 
                for g in last_3
            ) / 3
        
        if len(team_games) >= 5:
            last_5 = team_games[-5:]
            wins_5 = sum(1 for g in last_5 if (
                g["home_score"] > g["away_score"] and g["is_home"]
            ) or (g["away_score"] > g["home_score"] and not g["is_home"]))
            stats.last_10 = (wins_5, 5 - wins_5)
            
            stats.recent_form = []
            for g in last_5:
                is_win = (g["home_score"] > g["away_score"] and g["is_home"]) or \
                         (g["away_score"] > g["home_score"] and not g["is_home"])
                stats.recent_form.append("W" if is_win else "L")
        
        team_season_stats = self.get_team_stats(team_id)
        
        hitting = team_season_stats.get("hitting", {})
        if hitting:
            avg_str = hitting.get("avg", "0")
            if isinstance(avg_str, str):
                try:
                    stats.hits_avg = float(avg_str)
                except:
                    stats.hits_avg = 0.0
            else:
                stats.hits_avg = avg_str
            
            stats.obp = float(hitting.get("obp", 0) or 0)
            stats.slg = float(hitting.get("slg", 0) or 0)
            stats.ops = float(hitting.get("ops", 0) or 0)
            stats.home_runs = float(hitting.get("homeRuns", 0) or 0)
            stats.strikeouts = float(hitting.get("strikeOuts", 0) or 0)
            stats.walks = float(hitting.get("walks", 0) or 0)
        
        pitching = team_season_stats.get("pitching", {})
        if pitching:
            stats.era = float(pitching.get("era", 0) or 0)
            stats.whip = float(pitching.get("whip", 0) or 0)
            stats.k_per_nine = float(pitching.get("strikeoutsPer9Inn", 0) or 0)
            stats.bb_per_nine = float(pitching.get("walksPer9Inn", 0) or 0)
        
        return stats
    
    def calculate_win_probability(
        self, 
        home_stats: TeamStats, 
        away_stats: TeamStats,
        h2h: HeadToHead = None
    ) -> Tuple[float, float, Dict[str, List[str]]]:
        factors = {"home": [], "away": []}
        
        home_score = 50.0
        away_score = 50.0
        
        if home_stats.runs_scored_avg > away_stats.runs_scored_avg:
            diff = home_stats.runs_scored_avg - away_stats.runs_scored_avg
            home_score += diff * 5
            factors["home"].append(f"Home offense: {home_stats.runs_scored_avg:.1f} carreras/juego (últimos 3)")
        else:
            diff = away_stats.runs_scored_avg - home_stats.runs_scored_avg
            away_score += diff * 5
            factors["away"].append(f"Away offense: {away_stats.runs_scored_avg:.1f} carreras/juego (últimos 3)")
        
        if home_stats.era < away_stats.era:
            diff = away_stats.era - home_stats.era
            if diff > 0.2:
                home_score += diff * 4
                factors["home"].append(f"Home ERA: {home_stats.era:.2f} (buen pitcheo)")
        else:
            diff = home_stats.era - away_stats.era
            if diff > 0.2:
                away_score += diff * 4
                factors["away"].append(f"Away ERA: {away_stats.era:.2f} (buen pitcheo)")
        
        if home_stats.runs_allowed_avg < away_stats.runs_allowed_avg:
            diff = away_stats.runs_allowed_avg - home_stats.runs_allowed_avg
            if diff > 0.3:
                home_score += diff * 5
                factors["home"].append(f"Home defensa sólida: permite {home_stats.runs_allowed_avg:.1f} carreras (últimos 5)")
        else:
            diff = home_stats.runs_allowed_avg - away_stats.runs_allowed_avg
            if diff > 0.3:
                away_score += diff * 5
                factors["away"].append(f"Away defensa sólida: permite {away_stats.runs_allowed_avg:.1f} carreras (últimos 5)")
        
        if home_stats.run_differential > away_stats.run_differential:
            diff = home_stats.run_differential - away_stats.run_differential
            if diff > 20:
                home_score += 3
                factors["home"].append(f"Home run differential: +{home_stats.run_differential}")
        
        if home_stats.win_pct > away_stats.win_pct + 0.05:
            home_score += 5
            factors["home"].append(f"Home mejor record: {home_stats.wins}-{home_stats.losses}")
        
        if away_stats.win_pct > home_stats.win_pct + 0.05:
            away_score += 5
            factors["away"].append(f"Away mejor record: {away_stats.wins}-{away_stats.losses}")
        
        if home_stats.home_wins > home_stats.home_losses:
            home_score += 2
        
        if away_stats.away_wins > away_stats.away_losses:
            away_score += 2
        
        recent_wins = sum(1 for r in home_stats.recent_form if r == "W")
        if recent_wins >= 3:
            home_score += 5
            factors["home"].append(f"Home momentum: {recent_wins} victorias en últimos {len(home_stats.recent_form)}")
        
        recent_wins_away = sum(1 for r in away_stats.recent_form if r == "W")
        if recent_wins_away >= 3:
            away_score += 5
            factors["away"].append(f"Away momentum: {recent_wins_away} victorias en últimos {len(away_stats.recent_form)}")
        
        if h2h and h2h.total_games >= 3:
            h2h_pct = h2h.home_wins / h2h.total_games
            if h2h_pct > 0.6:
                home_score += 5
                factors["home"].append(f"Home domina H2H: {h2h.home_wins}-{h2h.home_losses}")
            elif h2h_pct < 0.4:
                away_score += 5
                factors["away"].append(f"Away domina H2H: {h2h.home_losses}-{h2h.home_wins}")
        
        total = home_score + away_score
        home_prob = (home_score / total) * 100
        away_prob = (away_score / total) * 100
        
        home_prob = max(20, min(80, home_prob))
        away_prob = 100 - home_prob
        
        return home_prob, away_prob, factors
    
    def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
        home_games = self.get_recent_games(home_team_id, 15)
        away_games = self.get_recent_games(away_team_id, 15)
        
        home_stats = self.calculate_team_metrics(home_team_id, home_games)
        away_stats = self.calculate_team_metrics(away_team_id, away_games)
        
        h2h = self.get_head_to_head(home_team_id, away_team_id)
        
        home_prob, away_prob, factors = self.calculate_win_probability(
            home_stats, away_stats, h2h
        )
        
        predicted_winner = home_stats.name if home_prob > away_prob else away_stats.name
        predicted_abbr = home_stats.abbreviation if home_prob > away_prob else away_stats.abbreviation
        confidence = max(home_prob, away_prob)
        
        return {
            "home_team": home_stats,
            "away_team": away_stats,
            "home_prob": home_prob,
            "away_prob": away_prob,
            "predicted_winner": predicted_winner,
            "predicted_abbr": predicted_abbr,
            "confidence": confidence,
            "factors": factors,
            "h2h": h2h,
            "season": self.season
        }
    
    def format_full_calendar(self, game: Dict, prediction: Dict, details: Dict = None) -> str:
        home = prediction["home_team"]
        away = prediction["away_team"]
        
        status = game.get("status", "")
        is_final = status == "F"
        is_live = status == "I"
        
        output = []
        
        away_score = game.get("away_score", 0) or 0
        home_score = game.get("home_score", 0) or 0
        
        away_hits = details.get("away_hits", 0) if details else 0
        away_errors = details.get("away_errors", 0) if details else 0
        home_hits = details.get("home_hits", 0) if details else 0
        home_errors = details.get("home_errors", 0) if details else 0
        
        output.append("Jueves, 16 de Abril, 2026")
        output.append("Momios de")
        output.append("Caliente")
        output.append("")
        
        if is_live:
            inning = game.get("current_inning", "")
            if inning:
                output.append(f"End of {inning}º")
            output.append("R")
            output.append("H")
            output.append("E")
        else:
            output.append("F")
            output.append("R")
            output.append("H")
            output.append("E")
        
        output.append("")
        
        output.append(f"{away.name}")
        output.append(f"({away.wins}-{away.losses}-{away.games_played - away.wins - away.losses}Visitante)")
        output.append(f"{away_score}")
        output.append(f"{away_hits}")
        output.append(f"{away_errors}")
        output.append("")
        
        output.append(f"{home.name}")
        output.append(f"({home.wins}-{home.losses}-{home.games_played - home.wins - home.losses}Local)")
        output.append(f"{home_score}")
        output.append(f"{home_hits}")
        output.append(f"{home_errors}")
        
        if is_live:
            output.append("B")
            output.append("S")
            output.append("O")
            output.append("")
            output.append("Última jugada:En juego")
        elif is_final:
            output.append("")
            pred_abbr = prediction['predicted_abbr']
            
            win_name = details.get("winning_pitcher_name", "") if details else ""
            win_wins = details.get("winning_pitcher_wins", 0) if details else 0
            win_losses = details.get("winning_pitcher_losses", 0) if details else 0
            lose_name = details.get("losing_pitcher_name", "") if details else ""
            lose_wins = details.get("losing_pitcher_wins", 0) if details else 0
            lose_losses = details.get("losing_pitcher_losses", 0) if details else 0
            save_name = details.get("save_pitcher_name", "") if details else ""
            
            output.append("GANA")
            output.append(f"{win_name}")
            output.append(f"{win_wins}-{win_losses} ERA")
            output.append("")
            output.append("PIERDE")
            output.append(f"{lose_name}")
            output.append(f"{lose_wins}-{lose_losses} ERA")
            
            if save_name:
                output.append("")
                output.append("SALVA")
                output.append(f"{save_name}")
            
            output.append("")
            output.append("🎯 PREDICCION")
            output.append(f"Gana: {pred_abbr}")
            output.append(f"Confianza: {prediction['confidence']:.0f}%")
        
        if not is_final and not is_live:
            game_time = game.get("time", "")
            if game_time:
                try:
                    dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                    time_str = dt.strftime("%-I:%M %p")
                except:
                    time_str = ""
                output.append("")
                output.append("Juego")
                output.append(f"{time_str}")
        
        output.append("")
        output.append("=" * 40)
        
        return "\n".join(output)


def run_predictions_for_date(model: MLBPredictor, date_to_use: str, save_csv: bool = True):
    from pathlib import Path
    
    print()
    print("📥 Cargando modelo...")
    print()
    
    games = model.get_games_for_date(date_to_use)
    
    if not games:
        print("❌ No hay partidos programados para esta fecha.")
        return
    
    print(f"✅ Partidos detectados: {len(games)}")
    print()
    print(f"⏳ Cargando datos de partidos...")
    print()
    
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
        print(model.format_full_calendar(game, pred, details))
        print()
    
    if save_csv:
        csv_path = Path(__file__).with_name(f"predictions_{date_to_use}.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "date",
                "game_pk",
                "status",
                "away_team",
                "home_team",
                "predicted_winner",
                "predicted_abbr",
                "confidence",
                "away_prob",
                "home_prob",
                "away_score",
                "home_score",
                "game_time",
                "venue",
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
        
        print(f"📄 CSV exportado: {csv_path}")


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
    print("  4 — 📄 Exportar CSV de hoy")
    print("  5 — 📄 Exportar CSV de ayer")
    print("  6 — 📄 Exportar CSV de fecha específica")
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
            date_to_use = datetime.now().strftime("%Y-%m-%d")
            model = MLBPredictor()
            print(f"\n📄 Exportando CSV para {date_to_use}...")
            run_predictions_for_date(model, date_to_use, save_csv=True)
            input("\n⏎ Presiona ENTER para continuar...")
            
        elif choice == "5":
            date_to_use = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            model = MLBPredictor()
            print(f"\n📄 Exportando CSV para {date_to_use}...")
            run_predictions_for_date(model, date_to_use, save_csv=True)
            input("\n⏎ Presiona ENTER para continuar...")
            
        elif choice == "6":
            date_to_use = get_date_input()
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
