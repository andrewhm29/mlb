"""Cliente de la MLB Stats API oficial (sin clave)."""

from __future__ import annotations

from typing import Any

import requests

API_BASE = "https://statsapi.mlb.com/api/v1"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_schedule_games(payload: dict) -> list[dict]:
    games: list[dict] = []
    for date_entry in payload.get("dates") or []:
        for game in date_entry.get("games") or []:
            status = (game.get("status") or {}).get("statusCode", "")
            if status not in ["S", "P", "PRE", "PW", "F", "I", "IR"]:
                continue
            away = (game.get("teams") or {}).get("away") or {}
            home = (game.get("teams") or {}).get("home") or {}
            away_team = away.get("team") or {}
            home_team = home.get("team") or {}
            venue = game.get("venue") or {}
            location = venue.get("location") or {}
            game_date = game.get("gameDate") or ""
            away_p = away.get("probablePitcher") or {}
            home_p = home.get("probablePitcher") or {}
            season = game.get("season") or (date_entry.get("date") or "")[:4]
            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "date": date_entry.get("date"),
                    "season": season,
                    "game_type": game.get("gameType") or "",
                    "status": status,
                    "status_detailed": (game.get("status") or {}).get("detailedState", ""),
                    "away_team_id": away_team.get("id"),
                    "away_team_name": away_team.get("name"),
                    "away_team_abbr": away_team.get("abbreviation") or away_team.get("fileCode", ""),
                    "away_score": away.get("score"),
                    "home_team_id": home_team.get("id"),
                    "home_team_name": home_team.get("name"),
                    "home_team_abbr": home_team.get("abbreviation") or home_team.get("fileCode", ""),
                    "home_score": home.get("score"),
                    "away_pitcher_id": away_p.get("id"),
                    "away_pitcher_name": away_p.get("fullName") or "",
                    "home_pitcher_id": home_p.get("id"),
                    "home_pitcher_name": home_p.get("fullName") or "",
                    "venue": venue.get("name"),
                    "venue_city": location.get("city", ""),
                    "venue_state": location.get("state", ""),
                    "time": game_date[:16] if game_date else "",
                    "current_inning": (game.get("linescore") or {}).get("currentInning", ""),
                    "inning_state": (game.get("linescore") or {}).get("inningState", ""),
                }
            )
    return games


def splits_by_team_id(payload: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    stats = payload.get("stats") or []
    if not stats:
        return out
    for split in stats[0].get("splits") or []:
        team = split.get("team") or {}
        tid = team.get("id")
        if tid is None:
            continue
        row = dict(split.get("stat") or {})
        row["_team_name"] = team.get("name", "")
        out[int(tid)] = row
    return out


def standings_by_team_id(payload: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for rec in payload.get("records") or []:
        for tr in rec.get("teamRecords") or []:
            team = tr.get("team") or {}
            tid = team.get("id")
            if tid is None:
                continue
            splits = {
                (s.get("type") or "").lower(): s
                for s in (tr.get("records") or {}).get("splitRecords") or []
            }
            home = splits.get("home") or {}
            away = splits.get("away") or {}
            last10 = splits.get("lastten") or splits.get("last10") or {}
            out[int(tid)] = {
                "name": team.get("name", ""),
                "wins": _to_int(tr.get("wins")),
                "losses": _to_int(tr.get("losses")),
                "win_pct": _to_float(tr.get("winningPercentage"), 0.5),
                "games_played": _to_int(tr.get("gamesPlayed")),
                "run_differential": _to_int(tr.get("runDifferential")),
                "streak": ((tr.get("streak") or {}).get("streakCode") or ""),
                "home_wins": _to_int(home.get("wins")),
                "home_losses": _to_int(home.get("losses")),
                "away_wins": _to_int(away.get("wins")),
                "away_losses": _to_int(away.get("losses")),
                "last_10": f"{_to_int(last10.get('wins'))}-{_to_int(last10.get('losses'))}"
                if last10
                else "",
            }
    return out


class MlbStatsClient:
    """Wrapper de https://statsapi.mlb.com/api/v1 (gratis, sin API key)."""

    def __init__(self, timeout: int = 20) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MLB-Predictor/1.0"})
        self.timeout = timeout

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{API_BASE}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  ⚠️  API Error ({endpoint}): {e}")
            return {}

    def schedule(
        self,
        date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        team_id: int | None = None,
        season: int | None = None,
        hydrate: str = "probablePitcher,team,linescore,venue",
        game_types: str | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "sportId": 1,
            "hydrate": hydrate,
        }
        if date:
            params["date"] = date
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        if team_id is not None:
            params["teamId"] = team_id
        if season is not None:
            params["season"] = season
        if game_types:
            params["gameTypes"] = game_types
        return parse_schedule_games(self.get("schedule", params))

    def schedule_season(self, season: int) -> list[dict]:
        """Temporada regular + playoffs de un año, con abridor anunciado si la API lo trae."""
        hydrate = "probablePitcher,team,venue"
        games = self.schedule(
            season=season,
            hydrate=hydrate,
            game_types="R,F,D,L,W",
        )
        if games:
            return games
        return self.schedule(
            start_date=f"{season}-03-01",
            end_date=f"{season}-11-30",
            hydrate=hydrate,
            game_types="R,F,D,L,W",
        )

    def season_starters(self, season: int) -> list[dict]:
        """ERA de temporada de pitchers con al menos un juego iniciado."""
        out: list[dict] = []
        offset = 0
        while offset < 5000:
            data = self.get(
                "stats",
                {
                    "stats": "season",
                    "group": "pitching",
                    "season": season,
                    "sportIds": 1,
                    "playerPool": "all",
                    "limit": 1000,
                    "offset": offset,
                },
            )
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            if not splits:
                break
            for split in splits:
                player = split.get("player") or {}
                pid = player.get("id")
                stat = split.get("stat") or {}
                gs = _to_int(stat.get("gamesStarted"))
                if pid is None or gs < 1:
                    continue
                out.append(
                    {
                        "season": season,
                        "player_id": int(pid),
                        "name": player.get("fullName") or "",
                        "era": _to_float(stat.get("era"), 4.5),
                        "whip": _to_float(stat.get("whip"), 1.3),
                        "ip": _to_float(stat.get("inningsPitched")),
                        "games_started": gs,
                    }
                )
            if len(splits) < 1000:
                break
            offset += 1000
        return out

    def boxscore(self, game_pk: int) -> dict:
        return self.get(f"game/{game_pk}/boxscore")

    def season_group_stats(
        self,
        season: int,
        group: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[int, dict]:
        params: dict[str, Any] = {
            "group": group,
            "stats": "season",
            "season": season,
            "sportIds": 1,
        }
        if start_date and end_date:
            params["stats"] = "byDateRange"
            params["startDate"] = start_date
            params["endDate"] = end_date
        return splits_by_team_id(self.get("teams/stats", params))

    def pitcher_season_stats(self, person_id: int, season: int) -> dict:
        data = self.get(
            f"people/{person_id}/stats",
            {
                "stats": "season",
                "group": "pitching",
                "season": season,
                "sportId": 1,
            },
        )
        splits = (data.get("stats") or [{}])[0].get("splits") or []
        if not splits:
            return {}
        stat = splits[0].get("stat") or {}
        return {
            "id": person_id,
            "era": _to_float(stat.get("era")),
            "whip": _to_float(stat.get("whip")),
            "k9": _to_float(stat.get("strikeoutsPer9Inn")),
            "bb9": _to_float(stat.get("walksPer9Inn")),
            "wins": _to_int(stat.get("wins")),
            "losses": _to_int(stat.get("losses")),
            "ip": _to_float(stat.get("inningsPitched")),
            "games_started": _to_int(stat.get("gamesStarted")),
        }

    def standings(self, season: int) -> dict[int, dict]:
        return standings_by_team_id(
            self.get("standings", {"leagueId": "103,104", "season": season})
        )
