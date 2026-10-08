"""Calendario ESPN (sin clave): abridores confirmados + línea O/U y moneyline de mercado."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import requests

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
ESPN_SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"

# ESPN abbr -> id de Stats API MLB
ESPN_TO_MLB_ID = {
    "ari": 109, "az": 109, "atl": 144, "bal": 110, "bos": 111, "chc": 112,
    "cws": 145, "chw": 145, "cin": 113, "cle": 114, "col": 115, "det": 116,
    "hou": 117, "kc": 118, "laa": 108, "lad": 119, "mia": 146, "mil": 158,
    "min": 142, "nym": 121, "nyy": 147, "oak": 133, "ath": 133, "phi": 143,
    "pit": 134, "sd": 135, "sf": 137, "sea": 136, "stl": 138, "tb": 139,
    "tex": 140, "tor": 141, "was": 120, "wsh": 120,
}


def _tid(abbr: str | None) -> int | None:
    if not abbr:
        return None
    return ESPN_TO_MLB_ID.get(str(abbr).strip().lower())


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        text = str(value).lower().replace("o", "").replace("u", "").strip()
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).replace("+", "")))
    except (TypeError, ValueError):
        return None


def _parse_odds(comp: dict) -> dict[str, Any]:
    block = (comp.get("odds") or [None])[0] or {}
    ou = _to_float(block.get("overUnder"))
    if ou is None:
        tot = block.get("total") or {}
        line = ((tot.get("over") or {}).get("close") or {}).get("line")
        ou = _to_float(line)
    ml = block.get("moneyline") or {}
    home_ml = _to_int(((ml.get("home") or {}).get("close") or {}).get("odds"))
    away_ml = _to_int(((ml.get("away") or {}).get("close") or {}).get("odds"))
    if home_ml is None:
        home_ml = _to_int((block.get("homeTeamOdds") or {}).get("moneyLine"))
    if away_ml is None:
        away_ml = _to_int((block.get("awayTeamOdds") or {}).get("moneyLine"))
    return {
        "ou_line": ou,
        "ml_home": home_ml,
        "ml_away": away_ml,
        "spread": _to_float(block.get("spread")),
        "odds_details": block.get("details") or "",
    }


def _starter_name(competitor: dict) -> str:
    for p in competitor.get("probables") or []:
        name = ((p.get("athlete") or {}).get("displayName")) or p.get("displayName")
        if name:
            return str(name)
    return ""


def parse_scoreboard(payload: dict) -> list[dict]:
    out: list[dict] = []
    for event in payload.get("events") or []:
        comp = (event.get("competitions") or [None])[0] or {}
        odds = _parse_odds(comp)
        home = away = None
        for c in comp.get("competitors") or []:
            if c.get("homeAway") == "home":
                home = c
            elif c.get("homeAway") == "away":
                away = c
        if not home or not away:
            continue
        hid = _tid((home.get("team") or {}).get("abbreviation"))
        aid = _tid((away.get("team") or {}).get("abbreviation"))
        if hid is None or aid is None:
            continue
        status = ((comp.get("status") or {}).get("type") or {})
        out.append(
            {
                "espn_id": event.get("id"),
                "name": event.get("shortName") or event.get("name") or "",
                "home_team_id": hid,
                "away_team_id": aid,
                "home_abbr": (home.get("team") or {}).get("abbreviation") or "",
                "away_abbr": (away.get("team") or {}).get("abbreviation") or "",
                "home_score": _to_int(home.get("score")),
                "away_score": _to_int(away.get("score")),
                "home_pitcher_name": _starter_name(home),
                "away_pitcher_name": _starter_name(away),
                "status_espn": status.get("description") or "",
                "state_espn": status.get("state") or "",
                **odds,
            }
        )
    return out


class EspnMlbClient:
    def __init__(self, timeout: int = 15) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; MLB-Predictor/1.0)"}
        )
        self.timeout = timeout

    def scoreboard(self, date: str | None = None) -> list[dict]:
        params: dict[str, str] = {}
        if date:
            params["dates"] = date.replace("-", "")
        try:
            r = self.session.get(ESPN_SCOREBOARD, params=params or None, timeout=self.timeout)
            r.raise_for_status()
            return parse_scoreboard(r.json())
        except Exception as exc:
            print(f"  ⚠️  ESPN scoreboard: {exc}")
            return []

    def _fill_odds_from_summary(self, row: dict) -> dict:
        if row.get("ou_line") is not None and row.get("ml_home") is not None:
            return row
        eid = row.get("espn_id")
        if not eid:
            return row
        try:
            r = self.session.get(ESPN_SUMMARY, params={"event": eid}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return row
        odds_list = data.get("odds") or data.get("pickcenter") or []
        if not odds_list:
            return row
        extra = _parse_odds({"odds": odds_list})
        for key, val in extra.items():
            if val in (None, "") and row.get(key) not in (None, ""):
                continue
            if row.get(key) in (None, "") and val not in (None, ""):
                row[key] = val
        return row

    def scoreboard_around(self, date: str) -> list[dict]:
        """UTC de ESPN a veces cae en el día siguiente; junta d-1, d y d+1."""
        try:
            d0 = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return self.scoreboard(date)
        seen: set[tuple[int, int]] = set()
        merged: list[dict] = []
        for delta in (-1, 0, 1):
            day = (d0 + timedelta(days=delta)).strftime("%Y-%m-%d")
            for row in self.scoreboard(day):
                key = (int(row["away_team_id"]), int(row["home_team_id"]))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(self._fill_odds_from_summary(row))
        return merged


def merge_espn_into_games(games: list[dict], espn_rows: list[dict]) -> list[dict]:
    by_pair = {
        (int(r["away_team_id"]), int(r["home_team_id"])): r for r in espn_rows
    }
    for game in games:
        try:
            key = (int(game["away_team_id"]), int(game["home_team_id"]))
        except (TypeError, ValueError, KeyError):
            continue
        row = by_pair.get(key)
        if not row:
            continue
        game["odds_source"] = "ESPN"
        if row.get("ou_line") is not None:
            game["ou_line"] = float(row["ou_line"])
            game["ou_market"] = True
        if row.get("ml_home") is not None:
            game["ml_home"] = row["ml_home"]
        if row.get("ml_away") is not None:
            game["ml_away"] = row["ml_away"]
        if row.get("spread") is not None:
            game["spread"] = row["spread"]
        if row.get("odds_details"):
            game["odds_details"] = row["odds_details"]
        if not game.get("home_pitcher_name") and row.get("home_pitcher_name"):
            game["home_pitcher_name"] = row["home_pitcher_name"]
        if not game.get("away_pitcher_name") and row.get("away_pitcher_name"):
            game["away_pitcher_name"] = row["away_pitcher_name"]
    return games
