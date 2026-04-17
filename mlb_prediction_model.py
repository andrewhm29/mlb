#!/usr/bin/env python3
"""
MLB Prediction Model - Con Historial Multi-Temporada
====================================================
Modelo predictivo para partidos de MLB con análisis detallado.
Incluye datos históricos de múltiples temporadas.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import json

API_BASE = "https://statsapi.mlb.com/api/v1"

TEAM_IDS = {
    "ari": 109, "atl": 144, "bal": 110, "bos": 111, "chc": 112,
    "cws": 145, "cwx": 145, "cin": 113, "cle": 114, "col": 115, "det": 116,
    "hou": 117, "kc": 118, "kan": 118, "laa": 108, "lad": 119, "mia": 146,
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
    fip: float = 0.0
    
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

class MLBPredictionModel:
    def __init__(self, season: int = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MLB-Prediction-Model/1.0"})
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.season = season or datetime.now().year
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{API_BASE}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error ({endpoint}): {e}")
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
        
        current_year = datetime.now().year
        
        if season < current_year:
            date_str = f"{season}-10-15"
        elif season == current_year:
            date_str = self.today
        else:
            date_str = f"{season}-04-01"
        
        params = {
            "leagueId": "103,104",
            "season": season
        }
        
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
        
        params = {
            "stats": "season",
            "group": "hitting,pitching",
            "season": season
        }
        
        data = self._get(f"teams/{team_id}/stats", params)
        result = {"hitting": {}, "pitching": {}}
        
        for stat_group in data.get("stats", []):
            group = stat_group.get("group", {}).get("displayName", "")
            if group in ["hitting", "pitching"]:
                for split in stat_group.get("splits", []):
                    result[group] = split.get("stat", {})
                    break
        
        return result
    
    def get_probable_pitcher(self, team_id: int, game_date: str = None) -> Optional[Dict]:
        if game_date is None:
            game_date = self.today
        
        games = self.get_schedule(game_date, game_date, team_id)
        
        for game in games:
            if game["home_team_id"] == team_id or game["away_team_id"] == team_id:
                boxscore = self._get(f"game/{game['game_pk']}/boxscore")
                for side in ["home", "away"]:
                    team = boxscore.get("teams", {}).get(side, {})
                    if team.get("team", {}).get("id") == team_id:
                        pitcher = team.get("probablePitcher", {})
                        if pitcher:
                            return {
                                "id": pitcher.get("id"),
                                "name": pitcher.get("fullName"),
                                "hand": pitcher.get("pitchHand", {}).get("code", "R")
                            }
        return None
    
    def get_pitcher_stats(self, pitcher_id: int, season: int = None) -> Dict:
        if season is None:
            season = self.season
        
        params = {
            "stats": "season,seasonAdvanced",
            "group": "pitching",
            "season": season
        }
        
        data = self._get(f"people/{pitcher_id}/stats", params)
        
        for stat_group in data.get("stats", []):
            if stat_group.get("type", {}).get("displayName") == "seasonAdvanced":
                splits = stat_group.get("splits", [])
                if splits:
                    return splits[0].get("stat", {})
        
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            if splits:
                return splits[0].get("stat", {})
        
        return {}
    
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
    
    def get_historical_seasons(self, team_id: int, years: int = 5) -> List[Dict]:
        current_year = datetime.now().year
        seasons_data = []
        
        for year in range(max(2020, current_year - years), current_year + 1):
            standings = self.get_standings(year)
            for s in standings:
                if s["team_id"] == team_id:
                    seasons_data.append({
                        "season": year,
                        "wins": s["wins"],
                        "losses": s["losses"],
                        "pct": s["pct"],
                        "run_diff": s["run_diff"]
                    })
                    break
        
        return seasons_data
    
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
                factors["home"].append(f"Home ERA: {home_stats.era:.2f} (buen pitcheo) ⭐⭐⭐")
        else:
            diff = home_stats.era - away_stats.era
            if diff > 0.2:
                away_score += diff * 4
                factors["away"].append(f"Away ERA: {away_stats.era:.2f} (buen pitcheo) ⭐⭐⭐")
        
        if home_stats.runs_allowed_avg < away_stats.runs_allowed_avg:
            diff = away_stats.runs_allowed_avg - home_stats.runs_allowed_avg
            if diff > 0.3:
                home_score += diff * 5
                factors["home"].append(f"Home defensa sólida: {home_stats.runs_allowed_avg:.1f} carreras (últimos 5) ⭐⭐⭐⭐")
        else:
            diff = home_stats.runs_allowed_avg - away_stats.runs_allowed_avg
            if diff > 0.3:
                away_score += diff * 5
                factors["away"].append(f"Away defensa sólida: {away_stats.runs_allowed_avg:.1f} carreras (últimos 5) ⭐⭐⭐⭐")
        
        if home_stats.run_differential > away_stats.run_differential:
            diff = home_stats.run_differential - away_stats.run_differential
            if diff > 20:
                home_score += 3
                factors["home"].append(f"Home run differential: +{home_stats.run_differential}")
        
        if home_stats.win_pct > away_stats.win_pct + 0.05:
            home_score += 5
            factors["home"].append(f"Home mejor record: {home_stats.wins}-{home_stats.losses} ({home_stats.win_pct:.3f})")
        
        if away_stats.win_pct > home_stats.win_pct + 0.05:
            away_score += 5
            factors["away"].append(f"Away mejor record: {away_stats.wins}-{away_stats.losses} ({away_stats.win_pct:.3f})")
        
        if home_stats.home_wins > home_stats.home_losses:
            home_score += 2
            factors["home"].append(f"Home sólido en casa: {home_stats.home_wins}-{home_stats.home_losses}")
        
        if away_stats.away_wins > away_stats.away_losses:
            away_score += 2
            factors["away"].append(f"Away sólido de visita: {away_stats.away_wins}-{away_stats.away_losses}")
        
        recent_wins = sum(1 for r in home_stats.recent_form if r == "W")
        if recent_wins >= 3:
            home_score += 5
            factors["home"].append(f"Home momentum: {recent_wins} victorias últimos {len(home_stats.recent_form)} ⭐")
        
        recent_wins_away = sum(1 for r in away_stats.recent_form if r == "W")
        if recent_wins_away >= 3:
            away_score += 5
            factors["away"].append(f"Away momentum: {recent_wins_away} victorias últimos {len(away_stats.recent_form)} ⭐")
        
        if h2h and h2h.total_games >= 3:
            h2h_pct = h2h.home_wins / h2h.total_games
            if h2h_pct > 0.6:
                home_score += 5
                factors["home"].append(f"Home domina H2H: {h2h.home_wins}-{h2h.home_losses} ({h2h_pct:.0%})")
            elif h2h_pct < 0.4:
                away_score += 5
                factors["away"].append(f"Away domina H2H: {h2h.home_losses}-{h2h.home_wins} ({1-h2h_pct:.0%})")
        
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
        confidence = max(home_prob, away_prob)
        
        historical = {
            "home": self.get_historical_seasons(home_team_id, 5),
            "away": self.get_historical_seasons(away_team_id, 5)
        }
        
        return {
            "home_team": home_stats,
            "away_team": away_stats,
            "home_prob": home_prob,
            "away_prob": away_prob,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "factors": factors,
            "h2h": h2h,
            "historical": historical,
            "season": self.season
        }

    def format_prediction(self, prediction: Dict) -> str:
        home = prediction["home_team"]
        away = prediction["away_team"]
        h2h = prediction.get("h2h", HeadToHead())
        
        line_length = 80
        border = "═" * line_length
        short_border = "─" * line_length
        
        output = []
        output.append(f"{border}")
        output.append(f"⚾ PARTIDO: {away.abbreviation} @ {home.abbreviation} | Temporada {prediction['season']}")
        output.append(border)
        
        output.append("")
        output.append(f"🎯 EL MODELO DICE: {prediction['predicted_winner'].upper()} GANA")
        output.append(f"   Confianza: {prediction['confidence']:.0f}%")
        output.append(f"   {away.abbreviation}: {prediction['away_prob']:.0f}% | {home.abbreviation}: {prediction['home_prob']:.0f}%")
        
        output.append("")
        
        if h2h.total_games > 0:
            output.append(f"📊 HISTORIAL H2H ({h2h.total_games} partidos)")
            output.append(short_border)
            output.append(f"   {home.abbreviation}: {h2h.home_wins} | {away.abbreviation}: {h2h.home_losses}")
            if h2h.recent_results:
                output.append(f"   Últimos: {'-'.join(h2h.recent_results[-5:])}")
            output.append("")
        
        output.append(f"✅ ¿POR QUÉ FAVORECE A {'HOME' if prediction['home_prob'] > prediction['away_prob'] else 'AWAY'}?")
        output.append(border)
        
        home_factors = prediction["factors"]["home"]
        away_factors = prediction["factors"]["away"]
        
        favored_team = "HOME" if prediction['home_prob'] > prediction['away_prob'] else "AWAY"
        
        if favored_team == "HOME":
            for factor in home_factors[:6]:
                output.append(f"  ⭐ {factor}")
            if not home_factors:
                output.append(f"  ⭐ {home.abbreviation} tiene ventaja general")
            output.append("")
            output.append(f"❌ ¿QUÉ FAVORECE A {away.abbreviation}?")
            output.append(border)
            for factor in away_factors[:4]:
                output.append(f"  ⚠️ {factor}")
            if not away_factors:
                output.append(f"  ⚠️ {away.abbreviation} competitivo")
        else:
            for factor in away_factors[:6]:
                output.append(f"  ⭐ {factor}")
            if not away_factors:
                output.append(f"  ⭐ {away.abbreviation} tiene ventaja general")
            output.append("")
            output.append(f"❌ ¿QUÉ FAVORECE A {home.abbreviation}?")
            output.append(border)
            for factor in home_factors[:4]:
                output.append(f"  ⚠️ {factor}")
            if not home_factors:
                output.append(f"  ⚠️ {home.abbreviation} competitivo")
        
        output.append("")
        output.append("📈 ESTADÍSTICAS COMPARATIVAS")
        output.append(border)
        output.append(f"  {'STAT':<18} {home.abbreviation:>8} | {away.abbreviation:>8}")
        output.append(short_border)
        output.append(f"  {'Record':<18} {home.wins}-{home.losses:>7} | {away.wins}-{away.losses:>7}")
        output.append(f"  {'Win %':<18} {home.win_pct:>8.3f} | {away.win_pct:>8.3f}")
        output.append(f"  {'ERA':<18} {home.era:>8.2f} | {away.era:>8.2f}")
        output.append(f"  {'WHIP':<18} {home.whip:>8.2f} | {away.whip:>8.2f}")
        output.append(f"  {'K/9':<18} {home.k_per_nine:>8.1f} | {away.k_per_nine:>8.1f}")
        output.append(f"  {'OPS':<18} {home.ops:>8.3f} | {away.ops:>8.3f}")
        output.append(f"  {'Run Diff':<18} {home.run_differential:>+8} | {away.run_differential:>+8}")
        output.append(f"  {'Últimos 5':<18} {''.join(home.recent_form) if home.recent_form else 'N/A':>8} | {''.join(away.recent_form) if away.recent_form else 'N/A':>8}")
        output.append(f"  {'Racha':<18} {home.streak or 'N/A':>8} | {away.streak or 'N/A':>8}")
        output.append(f"  {'Casa':<18} {home.home_wins}-{home.home_losses:>7} | --")
        output.append(f"  {'Visita':<18} -- | {away.away_wins}-{away.away_losses:>7}")
        
        output.append("")
        output.append("📅 RENDIMIENTO POR TEMPORADA")
        output.append(border)
        
        hist = prediction.get("historical", {})
        home_hist = hist.get("home", [])
        away_hist = hist.get("away", [])
        
        if home_hist:
            output.append(f"  {home.abbreviation}:")
            for h in home_hist[-4:]:
                output.append(f"    {h['season']}: {h['wins']}-{h['losses']} ({h['pct']:.3f}) | Diff: {h['run_diff']:+d}")
        
        output.append("")
        if away_hist:
            output.append(f"  {away.abbreviation}:")
            for h in away_hist[-4:]:
                output.append(f"    {h['season']}: {h['wins']}-{h['losses']} ({h['pct']:.3f}) | Diff: {h['run_diff']:+d}")
        
        output.append("")
        output.append(f"{border}")
        
        return "\n".join(output)


def main():
    print("⚾ MLB PREDICTION MODEL - Con Historial Multi-Temporada")
    print("=" * 60)
    print()
    
    print("Selecciona temporada:")
    print("  1. 2026 (actual)")
    print("  2. 2025")
    print("  3. 2024")
    print("  4. 2023")
    print("  5. Todas las temporadas")
    print()
    print("Temporada (1-5): ", end="")
    
    season_choice = input().strip()
    
    seasons_to_use = []
    if season_choice == "1" or season_choice == "":
        seasons_to_use = [2026]
    elif season_choice == "2":
        seasons_to_use = [2025]
    elif season_choice == "3":
        seasons_to_use = [2024]
    elif season_choice == "4":
        seasons_to_use = [2023]
    elif season_choice == "5":
        seasons_to_use = [2026, 2025, 2024, 2023, 2022]
    else:
        seasons_to_use = [2026]
    
    primary_season = seasons_to_use[0] if seasons_to_use else 2026
    print(f"\nUsando temporada(s): {seasons_to_use}")
    print()
    
    print("Equipos disponibles:")
    teams_list = sorted(TEAM_IDS.keys())
    for i in range(0, len(teams_list), 6):
        row = teams_list[i:i+6]
        print("  " + " | ".join([f"{t.upper():>4}" for t in row]))
    print()
    
    print("Ingresa equipo LOCAL (ej: bos, nyy, lad): ", end="")
    home_input = input().strip().lower()
    print("Ingresa equipo VISITA (ej: nyy, bos, lad): ", end="")
    away_input = input().strip().lower()
    
    model = MLBPredictionModel(season=primary_season)
    
    home_id = model.get_team_id(home_input)
    away_id = model.get_team_id(away_input)
    
    if not home_id or not away_id:
        print("Error: Equipo no encontrado.")
        return
    
    home_abbr = model.get_team_abbreviation(home_id)
    away_abbr = model.get_team_abbreviation(away_id)
    
    print(f"\nObteniendo datos para: {away_abbr} @ {home_abbr}")
    print()
    
    for season in seasons_to_use:
        print(f"\n{'='*80}")
        print(f"📊 ANÁLISIS TEMPORADA {season}")
        print(f"{'='*80}")
        
        season_model = MLBPredictionModel(season=season)
        
        prediction = season_model.predict_game(
            home_team_id=home_id,
            away_team_id=away_id
        )
        
        if prediction["home_team"].wins == 0 and prediction["away_team"].wins == 0:
            print(f"  No hay datos disponibles para la temporada {season}")
            continue
        
        print(season_model.format_prediction(prediction))
        
        if len(seasons_to_use) > 1:
            input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()
