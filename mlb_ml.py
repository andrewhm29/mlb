"""Modelo as-of: features sin futuro + logística sobre games_history.csv."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

GAMES_CSV = Path(__file__).parent / "data" / "games_history.csv"
PITCHER_CSV = Path(__file__).parent / "data" / "pitcher_seasons.csv"
MODEL_PATH = Path(__file__).parent / "models" / "home_win.joblib"
ROLL = 20
ELO_HOME = 30.0
ELO_K = 20.0
LEAGUE_ERA = 4.20
FEATURE_COLS = [
    "elo_diff",
    "rest_home",
    "rest_away",
    "roll_wp_home",
    "roll_wp_away",
    "roll_rs_home",
    "roll_rs_away",
    "roll_ra_home",
    "roll_ra_away",
    "season_wp_home",
    "season_wp_away",
    "season_rd_pg_home",
    "season_rd_pg_away",
    "home_wp_at_home",
    "away_wp_on_road",
    "park_rpg",
    "h2h_home_wp",
    "is_playoff",
    "sp_era_home",
    "sp_era_away",
    "sp_era_diff",
    "sp_known_home",
    "sp_known_away",
    "sp_roll_ra_home",
    "sp_roll_ra_away",
]


@dataclass
class TeamForm:
    elo: float = 1500.0
    last_date: str | None = None
    season: str | None = None
    w: int = 0
    l: int = 0
    rs: int = 0
    ra: int = 0
    games: deque = None
    home_games: deque = None
    away_games: deque = None

    def __post_init__(self) -> None:
        self.games = deque(maxlen=ROLL)
        self.home_games = deque(maxlen=ROLL)
        self.away_games = deque(maxlen=ROLL)

    def new_season(self, season: str) -> None:
        if self.season == season:
            return
        self.season = season
        self.w = self.l = 0
        self.rs = self.ra = 0

    def rest_days(self, date_str: str) -> float:
        if not self.last_date:
            return 3.0
        try:
            d1 = datetime.strptime(self.last_date, "%Y-%m-%d")
            d2 = datetime.strptime(date_str, "%Y-%m-%d")
            return float(min(10, max(0, (d2 - d1).days)))
        except ValueError:
            return 3.0

    def wp(self, bucket: deque | None = None) -> float:
        g = bucket if bucket is not None else self.games
        if not g:
            return 0.5
        return sum(x["win"] for x in g) / len(g)

    def avg(self, key: str) -> float:
        if not self.games:
            return 4.5
        return sum(x[key] for x in self.games) / len(self.games)

    def season_wp(self) -> float:
        n = self.w + self.l
        return self.w / n if n else 0.5

    def season_rd_pg(self) -> float:
        n = self.w + self.l
        return (self.rs - self.ra) / n if n else 0.0


def _tid(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_final_games(path: Path | None = None) -> list[dict]:
    import csv

    csv_path = path or GAMES_CSV
    rows: list[dict] = []
    if not csv_path.exists():
        return rows
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("status") or "") != "F":
                continue
            hid = _tid(row.get("home_team_id"))
            aid = _tid(row.get("away_team_id"))
            try:
                hs = int(float(row.get("home_score") or ""))
                aws = int(float(row.get("away_score") or ""))
            except (TypeError, ValueError):
                continue
            if hid is None or aid is None:
                continue
            rows.append(
                {
                    "game_pk": str(row.get("game_pk") or ""),
                    "date": str(row.get("date") or ""),
                    "season": str(row.get("season") or (row.get("date") or "")[:4]),
                    "game_type": str(row.get("game_type") or "R"),
                    "home_id": hid,
                    "away_id": aid,
                    "home_score": hs,
                    "away_score": aws,
                    "home_pitcher_id": _tid(row.get("home_pitcher_id")),
                    "away_pitcher_id": _tid(row.get("away_pitcher_id")),
                    "venue": str(row.get("venue") or ""),
                    "home_win": 1 if hs > aws else 0,
                }
            )
    rows.sort(key=lambda r: (r["date"], r["game_pk"]))
    return rows


def _h2h_wp(meetings: deque, home_id: int) -> float:
    if not meetings:
        return 0.5
    return sum(1 for w in meetings if w == home_id) / len(meetings)


def load_prior_eras(path: Path | None = None) -> dict[tuple[int, int], float]:
    import csv

    csv_path = path or PITCHER_CSV
    out: dict[tuple[int, int], float] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                year = int(row["season"])
                pid = int(row["player_id"])
                era = float(row["era"])
            except (KeyError, TypeError, ValueError):
                continue
            if era > 0:
                out[(year, pid)] = era
    return out


def _sp_era(priors: dict[tuple[int, int], float], season: str, pid: int | None, live_era: float | None = None) -> tuple[float, float]:
    if live_era is not None and live_era > 0:
        return float(live_era), 1.0
    if not pid:
        return LEAGUE_ERA, 0.0
    try:
        year = int(season)
    except (TypeError, ValueError):
        return LEAGUE_ERA, 0.0
    era = priors.get((year - 1, pid))
    if era is None:
        return LEAGUE_ERA, 0.0
    return float(era), 1.0


def _sp_roll_ra(starts: dict[int, deque], pid: int | None) -> float:
    if not pid:
        return 4.5
    dq = starts.get(pid)
    if not dq:
        return 4.5
    return sum(dq) / len(dq)


def features_from_state(
    home: TeamForm,
    away: TeamForm,
    date: str,
    venue: str,
    game_type: str,
    park_rpg: float,
    h2h_wp: float,
    sp: dict[str, float] | None = None,
) -> dict[str, float]:
    extra = sp or {}
    return {
        "elo_diff": home.elo - away.elo,
        "rest_home": home.rest_days(date),
        "rest_away": away.rest_days(date),
        "roll_wp_home": home.wp(),
        "roll_wp_away": away.wp(),
        "roll_rs_home": home.avg("rs"),
        "roll_rs_away": away.avg("rs"),
        "roll_ra_home": home.avg("ra"),
        "roll_ra_away": away.avg("ra"),
        "season_wp_home": home.season_wp(),
        "season_wp_away": away.season_wp(),
        "season_rd_pg_home": home.season_rd_pg(),
        "season_rd_pg_away": away.season_rd_pg(),
        "home_wp_at_home": home.wp(home.home_games),
        "away_wp_on_road": away.wp(away.away_games),
        "park_rpg": park_rpg,
        "h2h_home_wp": h2h_wp,
        "is_playoff": 0.0 if game_type == "R" else 1.0,
        "sp_era_home": extra.get("sp_era_home", LEAGUE_ERA),
        "sp_era_away": extra.get("sp_era_away", LEAGUE_ERA),
        "sp_era_diff": extra.get("sp_era_diff", 0.0),
        "sp_known_home": extra.get("sp_known_home", 0.0),
        "sp_known_away": extra.get("sp_known_away", 0.0),
        "sp_roll_ra_home": extra.get("sp_roll_ra_home", 4.5),
        "sp_roll_ra_away": extra.get("sp_roll_ra_away", 4.5),
    }


def _apply_result(home: TeamForm, away: TeamForm, game: dict) -> None:
    hw = game["home_win"]
    hs, aws = game["home_score"], game["away_score"]
    exp = 1.0 / (1.0 + 10 ** ((away.elo - home.elo - ELO_HOME) / 400.0))
    home.elo += ELO_K * (hw - exp)
    away.elo += ELO_K * ((1 - hw) - (1 - exp))
    rec_h = {"win": hw, "rs": hs, "ra": aws}
    rec_a = {"win": 1 - hw, "rs": aws, "ra": hs}
    home.games.append(rec_h)
    away.games.append(rec_a)
    home.home_games.append(rec_h)
    away.away_games.append(rec_a)
    if hw:
        home.w += 1
        away.l += 1
    else:
        home.l += 1
        away.w += 1
    home.rs += hs
    home.ra += aws
    away.rs += aws
    away.ra += hs
    home.last_date = away.last_date = game["date"]


def _sp_features(
    season: str,
    home_pid: int | None,
    away_pid: int | None,
    priors: dict[tuple[int, int], float],
    starts: dict[int, deque],
    live_home_era: float | None = None,
    live_away_era: float | None = None,
) -> dict[str, float]:
    he, hk = _sp_era(priors, season, home_pid, live_home_era)
    ae, ak = _sp_era(priors, season, away_pid, live_away_era)
    return {
        "sp_era_home": he,
        "sp_era_away": ae,
        "sp_era_diff": ae - he,
        "sp_known_home": hk,
        "sp_known_away": ak,
        "sp_roll_ra_home": _sp_roll_ra(starts, home_pid),
        "sp_roll_ra_away": _sp_roll_ra(starts, away_pid),
    }


def _note_start(starts: dict[int, deque], pid: int | None, runs_allowed: int) -> None:
    if not pid:
        return
    if pid not in starts:
        starts[pid] = deque(maxlen=8)
    starts[pid].append(float(runs_allowed))


def build_dataset(games: list[dict] | None = None) -> tuple[list[dict], list[int], list[dict]]:
    """Una pasada: features SOLO con partidos anteriores. Devuelve X rows, y, meta."""
    games = games if games is not None else load_final_games()
    priors = load_prior_eras()
    teams: dict[int, TeamForm] = defaultdict(TeamForm)
    venue_runs: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    h2h: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=10))
    starts: dict[int, deque] = {}
    league_runs = 0.0
    league_n = 0
    X_rows: list[dict] = []
    y: list[int] = []
    meta: list[dict] = []

    for g in games:
        hid, aid = g["home_id"], g["away_id"]
        home, away = teams[hid], teams[aid]
        home.new_season(g["season"])
        away.new_season(g["season"])
        vr = venue_runs[g["venue"]]
        park = (vr[0] / vr[1]) if vr[1] else (league_runs / league_n if league_n else 9.0)
        pair = (min(hid, aid), max(hid, aid))
        sp = _sp_features(
            g["season"],
            g.get("home_pitcher_id"),
            g.get("away_pitcher_id"),
            priors,
            starts,
        )
        feats = features_from_state(
            home,
            away,
            g["date"],
            g["venue"],
            g["game_type"],
            park,
            _h2h_wp(h2h[pair], hid),
            sp,
        )
        X_rows.append(feats)
        y.append(g["home_win"])
        meta.append(
            {
                "date": g["date"],
                "season": g["season"],
                "home_id": hid,
                "away_id": aid,
                "home_win": g["home_win"],
                "venue": g["venue"],
            }
        )
        _apply_result(home, away, g)
        _note_start(starts, g.get("home_pitcher_id"), g["away_score"])
        _note_start(starts, g.get("away_pitcher_id"), g["home_score"])
        total = g["home_score"] + g["away_score"]
        vr[0] += total
        vr[1] += 1
        league_runs += total
        league_n += 1
        winner = hid if g["home_win"] else aid
        h2h[pair].append(winner)

    return X_rows, y, meta


def snapshot_before(as_of: str, games: list[dict] | None = None) -> dict[str, Any]:
    """Estado de equipos/parques usando solo juegos con date < as_of."""
    games = games if games is not None else load_final_games()
    teams: dict[int, TeamForm] = defaultdict(TeamForm)
    venue_runs: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    h2h: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=10))
    starts: dict[int, deque] = {}
    league_runs = 0.0
    league_n = 0
    for g in games:
        if g["date"] >= as_of:
            break
        hid, aid = g["home_id"], g["away_id"]
        home, away = teams[hid], teams[aid]
        home.new_season(g["season"])
        away.new_season(g["season"])
        _apply_result(home, away, g)
        _note_start(starts, g.get("home_pitcher_id"), g["away_score"])
        _note_start(starts, g.get("away_pitcher_id"), g["home_score"])
        total = g["home_score"] + g["away_score"]
        venue_runs[g["venue"]][0] += total
        venue_runs[g["venue"]][1] += 1
        league_runs += total
        league_n += 1
        h2h[(min(hid, aid), max(hid, aid))].append(hid if g["home_win"] else aid)
    return {
        "teams": teams,
        "venue_runs": venue_runs,
        "h2h": h2h,
        "starts": starts,
        "priors": load_prior_eras(),
        "league_rpg": (league_runs / league_n) if league_n else 9.0,
    }


def matchup_features(
    snap: dict[str, Any],
    home_id: int,
    away_id: int,
    date: str,
    venue: str,
    game_type: str = "R",
    home_pitcher_id: int | None = None,
    away_pitcher_id: int | None = None,
    live_home_era: float | None = None,
    live_away_era: float | None = None,
) -> dict[str, float]:
    teams: dict[int, TeamForm] = snap["teams"]
    home = teams.get(home_id) or TeamForm()
    away = teams.get(away_id) or TeamForm()
    vr = snap["venue_runs"].get(venue) or [0.0, 0.0]
    park = (vr[0] / vr[1]) if vr[1] else snap["league_rpg"]
    pair = (min(home_id, away_id), max(home_id, away_id))
    season = date[:4]
    sp = _sp_features(
        season,
        home_pitcher_id,
        away_pitcher_id,
        snap.get("priors") or {},
        snap.get("starts") or {},
        live_home_era,
        live_away_era,
    )
    return features_from_state(
        home,
        away,
        date,
        venue,
        game_type,
        park,
        _h2h_wp(snap["h2h"].get(pair, deque()), home_id),
        sp,
    )


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows], dtype="float64")


def train_and_save(
    holdout_season: str = "2026",
    path: Path | None = None,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    import joblib

    X_rows, y, meta = build_dataset()
    if len(y) < 500:
        raise RuntimeError("Hace falta historial: python3 scripts/01_download.py --from-year 2010")

    y_arr = np.array(y, dtype=int)
    train_idx = [i for i, m in enumerate(meta) if m["season"] < holdout_season]
    test_idx = [i for i, m in enumerate(meta) if m["season"] == holdout_season]
    if len(train_idx) < 500:
        cut = int(len(y) * 0.85)
        train_idx = list(range(cut))
        test_idx = list(range(cut, len(y)))

    X = _matrix(X_rows)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    weights = np.array(
        [
            0.45 + 0.55 * (int(meta[i]["season"]) - 2010) / max(1, datetime.now().year - 2010)
            for i in train_idx
        ],
        dtype="float64",
    )
    model = LogisticRegression(max_iter=400, solver="lbfgs")
    model.fit(X_train, y_arr[train_idx], sample_weight=weights)

    def _metrics(idx: list[int]) -> dict[str, float]:
        if not idx:
            return {}
        p = model.predict_proba(scaler.transform(X[idx]))[:, 1]
        pred = (p >= 0.5).astype(int)
        yt = y_arr[idx]
        home_base = float(yt.mean())
        return {
            "n": float(len(idx)),
            "accuracy": float(accuracy_score(yt, pred)),
            "baseline_home": home_base,
            "brier": float(brier_score_loss(yt, p)),
            "log_loss": float(log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6))),
        }

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "holdout_season": holdout_season,
        "train_metrics": _metrics(train_idx),
        "holdout_metrics": _metrics(test_idx),
    }
    out = path or MODEL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    bundle["path"] = str(out)
    return bundle


def load_bundle(path: Path | None = None) -> dict[str, Any] | None:
    import joblib

    p = path or MODEL_PATH
    if not p.exists():
        return None
    return joblib.load(p)


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return float(np.log(p / (1 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def p_home_win(
    bundle: dict[str, Any],
    feats: dict[str, float],
    home_era: float | None = None,
    away_era: float | None = None,
) -> float:
    x = np.array([[float(feats[c]) for c in bundle["feature_cols"]]], dtype="float64")
    xs = bundle["scaler"].transform(x)
    p = float(bundle["model"].predict_proba(xs)[0, 1])
    if "sp_era_diff" not in bundle.get("feature_cols", []) and home_era is not None and away_era is not None:
        era_gap = float(np.clip(away_era - home_era, -2.5, 2.5))
        p = _sigmoid(_logit(p) + 0.18 * era_gap)
    return float(np.clip(p, 0.05, 0.95))


def print_metrics(bundle: dict[str, Any]) -> None:
    def _line(title: str, m: dict[str, float]) -> None:
        if not m:
            print(f"  {title}: (sin juegos)")
            return
        print(
            f"  {title}: n={int(m['n'])}  accuracy={m['accuracy']:.3f}  "
            f"baseline local={m['baseline_home']:.3f}  brier={m['brier']:.3f}  "
            f"log_loss={m['log_loss']:.3f}"
        )

    print("\n📊 Modelo victoria del local (as-of, sin futuro)")
    _line("Train  (< holdout)", bundle.get("train_metrics") or {})
    hold = bundle.get("holdout_season", "")
    _line(f"Holdout ({hold})", bundle.get("holdout_metrics") or {})
    print(f"  Archivo: {bundle.get('path', MODEL_PATH)}\n")
