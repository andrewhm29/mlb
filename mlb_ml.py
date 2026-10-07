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
DEFAULT_OU_LINE = 8.5
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
TOTALS_COLS = [
    "exp_total",
    "park_rpg",
    "roll_rs_home",
    "roll_rs_away",
    "roll_ra_home",
    "roll_ra_away",
    "sp_era_home",
    "sp_era_away",
    "sp_roll_ra_home",
    "sp_roll_ra_away",
    "sp_known_home",
    "sp_known_away",
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

    def last_n(self, n: int = 5, bucket: deque | None = None) -> list:
        g = list(bucket if bucket is not None else self.games)
        return g[-n:]

    def last_n_wp(self, n: int = 5, bucket: deque | None = None) -> float:
        g = self.last_n(n, bucket)
        if not g:
            return 0.5
        return sum(x["win"] for x in g) / len(g)

    def last_n_avg(self, key: str, n: int = 5, bucket: deque | None = None) -> float:
        g = self.last_n(n, bucket)
        if not g:
            return 4.5
        return sum(x[key] for x in g) / len(g)

    def context(self, n: int = 5) -> dict[str, float]:
        g = self.last_n(n)
        hg = self.last_n(n, self.home_games)
        ag = self.last_n(n, self.away_games)
        return {
            "n": float(len(g)),
            "wp": self.last_n_wp(n),
            "rs": self.last_n_avg("rs", n),
            "ra": self.last_n_avg("ra", n),
            "wins": float(sum(x["win"] for x in g)),
            "home_n": float(len(hg)),
            "home_wins": float(sum(x["win"] for x in hg)),
            "home_rs": sum(x["rs"] for x in hg) / len(hg) if hg else 4.5,
            "home_ra": sum(x["ra"] for x in hg) / len(hg) if hg else 4.5,
            "away_n": float(len(ag)),
            "away_wins": float(sum(x["win"] for x in ag)),
            "away_rs": sum(x["rs"] for x in ag) / len(ag) if ag else 4.5,
            "away_ra": sum(x["ra"] for x in ag) / len(ag) if ag else 4.5,
            "season_wp": self.season_wp(),
            "season_w": float(self.w),
            "season_l": float(self.l),
        }


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
    rs_h, ra_h = home.avg("rs"), home.avg("ra")
    rs_a, ra_a = away.avg("rs"), away.avg("ra")
    return {
        "elo_diff": home.elo - away.elo,
        "rest_home": home.rest_days(date),
        "rest_away": away.rest_days(date),
        "roll_wp_home": home.wp(),
        "roll_wp_away": away.wp(),
        "roll_rs_home": rs_h,
        "roll_rs_away": rs_a,
        "roll_ra_home": ra_h,
        "roll_ra_away": ra_a,
        "exp_total": 0.5 * (rs_h + ra_a) + 0.5 * (rs_a + ra_h),
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
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "total": g["home_score"] + g["away_score"],
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


def _matrix(rows: list[dict], cols: list[str] | None = None) -> np.ndarray:
    use = cols or FEATURE_COLS
    return np.array([[float(r.get(c, 0.0)) for c in use] for r in rows], dtype="float64")


def _exp_sides(feats: dict[str, float]) -> tuple[float, float]:
    home_r = 0.5 * float(feats.get("roll_rs_home") or 4.5) + 0.5 * float(feats.get("roll_ra_away") or 4.5)
    away_r = 0.5 * float(feats.get("roll_rs_away") or 4.5) + 0.5 * float(feats.get("roll_ra_home") or 4.5)
    return home_r, away_r


def _sp_run_adj(feats: dict[str, float], side: str) -> float:
    known = float(feats.get(f"sp_known_{side}") or 0.0)
    era = float(feats.get(f"sp_era_{side}") or LEAGUE_ERA)
    roll = float(feats.get(f"sp_roll_ra_{side}") or 4.5)
    era_s = (0.5 * era + 0.5 * LEAGUE_ERA) if known else LEAGUE_ERA
    roll_s = 0.35 * roll + 0.65 * 4.5
    return 0.32 * (era_s - LEAGUE_ERA) + 0.20 * (roll_s - 4.5)


def structural_total(feats: dict[str, float], league_total: float = 8.83) -> dict[str, float]:
    """Carreras esperadas por lado: ataque vs pitcheo rival + abridor contrario + parque."""
    home_r, away_r = _exp_sides(feats)
    park_each = 0.20 * (float(feats.get("park_rpg") or league_total) - league_total)
    # El abridor local enfrenta a la ofensiva visitante (y al revés).
    home_r = home_r + _sp_run_adj(feats, "away") + park_each
    away_r = away_r + _sp_run_adj(feats, "home") + park_each
    home_r = float(np.clip(home_r, 2.2, 8.0))
    away_r = float(np.clip(away_r, 2.2, 8.0))
    total = float(np.clip(home_r + away_r, 6.0, 13.0))
    return {"home_runs": home_r, "away_runs": away_r, "total": total}


def train_and_save(
    holdout_season: str = "2026",
    path: Path | None = None,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
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

    y_total = np.array([m["total"] for m in meta], dtype="float64")
    y_home_r = np.array([m["home_score"] for m in meta], dtype="float64")
    y_away_r = np.array([m["away_score"] for m in meta], dtype="float64")
    league_rpg = float(y_total[train_idx].mean()) if train_idx else 8.83
    Xt = _matrix(X_rows, TOTALS_COLS)
    tot_scaler = StandardScaler()
    Xt_train = tot_scaler.fit_transform(Xt[train_idx])
    ridge_total = Ridge(alpha=2.0)
    ridge_home = Ridge(alpha=2.0)
    ridge_away = Ridge(alpha=2.0)
    ridge_total.fit(Xt_train, y_total[train_idx], sample_weight=weights)
    ridge_home.fit(Xt_train, y_home_r[train_idx], sample_weight=weights)
    ridge_away.fit(Xt_train, y_away_r[train_idx], sample_weight=weights)
    totals_blend = 0.28
    struct_tr = np.array(
        [structural_total(X_rows[i], league_rpg)["total"] for i in train_idx],
        dtype="float64",
    )
    blend_tr = totals_blend * ridge_total.predict(Xt_train) + (1.0 - totals_blend) * struct_tr
    resid_std = float(np.std(y_total[train_idx] - blend_tr)) or 2.8
    ou_sigma = float(np.clip(0.70 * resid_std, 2.55, 3.20))

    def _blend_pred(idx: list[int]) -> np.ndarray:
        ridge_hat = ridge_total.predict(tot_scaler.transform(Xt[idx]))
        struct_hat = np.array(
            [structural_total(X_rows[i], league_rpg)["total"] for i in idx],
            dtype="float64",
        )
        return totals_blend * ridge_hat + (1.0 - totals_blend) * struct_hat

    def _totals_metrics(idx: list[int]) -> dict[str, float]:
        if not idx:
            return {}
        pred_t = _blend_pred(idx)
        yt = y_total[idx]
        line = np.full_like(pred_t, DEFAULT_OU_LINE)
        ou_hat = (pred_t > line).astype(int)
        ou_true = (yt > line).astype(int)
        return {
            "n": float(len(idx)),
            "mae_total": float(mean_absolute_error(yt, pred_t)),
            "mean_pred": float(pred_t.mean()),
            "mean_real": float(yt.mean()),
            "pred_std": float(pred_t.std()),
            "pct_over_8_5": float((pred_t > line).mean()),
            "ou_acc_8_5": float((ou_hat == ou_true).mean()),
        }

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "holdout_season": holdout_season,
        "train_metrics": _metrics(train_idx),
        "holdout_metrics": _metrics(test_idx),
        "ridge_total": ridge_total,
        "ridge_home": ridge_home,
        "ridge_away": ridge_away,
        "totals_scaler": tot_scaler,
        "totals_cols": TOTALS_COLS,
        "totals_blend": totals_blend,
        "league_rpg": league_rpg,
        "resid_std": resid_std,
        "ou_sigma": ou_sigma,
        "totals_train": _totals_metrics(train_idx),
        "totals_holdout": _totals_metrics(test_idx),
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


def _phi(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def predict_totals(bundle: dict[str, Any], feats: dict[str, float]) -> dict[str, float]:
    league = float(bundle.get("league_rpg") or 8.83)
    struct = structural_total(feats, league)
    total = struct["total"]
    cols = bundle.get("totals_cols")
    scaler = bundle.get("totals_scaler")
    ridge = bundle.get("ridge_total")
    if cols and scaler is not None and ridge is not None:
        x = np.array([[float(feats.get(c, 0.0)) for c in cols]], dtype="float64")
        xs = scaler.transform(x)
        ridge_t = float(ridge.predict(xs)[0])
        w = float(bundle.get("totals_blend", 0.28))
        total = w * ridge_t + (1.0 - w) * struct["total"]
    total = float(np.clip(total, 6.0, 13.0))
    split = max(struct["home_runs"] + struct["away_runs"], 0.1)
    home_r = total * (struct["home_runs"] / split)
    away_r = total - home_r
    sigma = float(bundle.get("ou_sigma") or bundle.get("resid_std") or 2.8)
    sigma = float(np.clip(sigma, 2.55, 3.20))
    return {
        "home_runs": home_r,
        "away_runs": away_r,
        "total": total,
        "sigma": sigma,
    }


def projected_score(home_exp: float, away_exp: float, p_home: float) -> tuple[int, int]:
    """Marcador entero coherente con favorito y carreras esperadas (nunca empate)."""
    home_exp = max(0.8, float(home_exp))
    away_exp = max(0.8, float(away_exp))
    home_i = max(1, int(round(home_exp)))
    away_i = max(1, int(round(away_exp)))
    fav_home = float(p_home) >= 0.5
    if fav_home:
        if home_i < away_i:
            if away_i > 1:
                home_i += 1
                away_i -= 1
            if home_i <= away_i:
                home_i = away_i + 1
        elif home_i == away_i:
            home_i += 1
    else:
        if away_i < home_i:
            if home_i > 1:
                away_i += 1
                home_i -= 1
            if away_i <= home_i:
                away_i = home_i + 1
        elif away_i == home_i:
            away_i += 1
    return int(home_i), int(away_i)


def ou_factor_rows(
    feats: dict[str, float],
    totals: dict[str, float],
    line: float,
    league: float | None = None,
) -> list[str]:
    home_r, away_r = _exp_sides(feats)
    park = float(feats.get("park_rpg") or 0)
    lg = float(league) if league else 8.83
    rows = [
        f"Forma reciente: local ~{home_r:.1f} R esperadas, visitante ~{away_r:.1f} R "
        f"(RS/RA últimos {ROLL} juegos)",
        f"Parque: {park:.1f} R/juego de ambiente (no se usa como línea)",
    ]
    eh = float(feats.get("sp_era_home") or LEAGUE_ERA)
    ea = float(feats.get("sp_era_away") or LEAGUE_ERA)
    if feats.get("sp_known_home") or feats.get("sp_known_away"):
        rows.append(f"Abridores (ERA, encogida hacia la liga): {eh:.2f} vs {ea:.2f}")
    tot = float(totals["total"])
    decide = ou_decide_at(line, lg)
    if decide > line + 0.05:
        rows.append(
            f"Línea {line:g} está bajo la media MLB ({lg:.1f}); "
            f"OVER solo si el total ({tot:.1f}) supera un juego típico"
        )
    rows.append(f"Total del modelo {tot:.1f} vs línea {line:g} ({tot - line:+.1f})")
    return rows


def ou_decide_at(line: float, league: float | None = None) -> float:
    """Si la línea está bajo la media de liga (~8.9), 8.5 sale OVER en casi todo.

    El lado se decide contra max(línea, media): un juego típico queda 50/50, no OVER.
    """
    lg = float(league) if league and league > 0 else 8.83
    return float(max(line, lg))


def over_under(
    pred_total: float,
    line: float,
    sigma: float = 2.8,
    league: float | None = None,
) -> dict[str, Any]:
    decide = ou_decide_at(line, league)
    z = (decide - pred_total) / max(sigma, 0.6)
    p_over = 1.0 - _phi(z)
    p_over = float(np.clip(p_over, 0.05, 0.95))
    pick = "OVER" if p_over >= 0.5 else "UNDER"
    p_pick = p_over if pick == "OVER" else 1.0 - p_over
    return {
        "line": float(line),
        "decide_at": decide,
        "pick": pick,
        "p_over": p_over,
        "p_under": 1.0 - p_over,
        "p_pick": p_pick,
        "diff": pred_total - line,
        "edge_vs_typical": pred_total - decide,
    }


def round_line(value: float) -> float:
    return round(value * 2.0) / 2.0


def team_context(snap: dict[str, Any], team_id: int, n: int = 5) -> dict[str, float]:
    team = (snap.get("teams") or {}).get(team_id) or TeamForm()
    return team.context(n)


def win_factor_rows(
    bundle: dict[str, Any],
    feats: dict[str, float],
    home_abbr: str,
    away_abbr: str,
    p_home: float,
) -> tuple[list[str], list[str]]:
    """Factores del modelo (pp) + contexto. pp ≈ p*(1-p)*contribución al logit."""
    cols = bundle["feature_cols"]
    x = np.array([[float(feats.get(c, 0.0)) for c in cols]], dtype="float64")
    xs = bundle["scaler"].transform(x)[0]
    coef = bundle["model"].coef_[0]
    scale = float(p_home * (1.0 - p_home) * 100.0)
    labeled = []
    texts = {
        "elo_diff": ("Elo / nivel reciente", True),
        "rest_home": (f"Descanso de {home_abbr}", True),
        "rest_away": (f"Descanso de {away_abbr}", False),
        "roll_wp_home": (f"Forma reciente de {home_abbr} (win%)", True),
        "roll_wp_away": (f"Forma reciente de {away_abbr} (win%)", False),
        "roll_rs_home": (f"Ataque de {home_abbr}: {feats.get('roll_rs_home', 0):.1f} R/juego", True),
        "roll_rs_away": (f"Ataque de {away_abbr}: {feats.get('roll_rs_away', 0):.1f} R/juego", False),
        "roll_ra_home": (f"Pitcheo de {home_abbr}: permite {feats.get('roll_ra_home', 0):.1f} R/juego", True),
        "roll_ra_away": (f"Pitcheo de {away_abbr}: permite {feats.get('roll_ra_away', 0):.1f} R/juego", False),
        "season_wp_home": (f"Récord de temporada {home_abbr}", True),
        "season_wp_away": (f"Récord de temporada {away_abbr}", False),
        "season_rd_pg_home": (f"Diferencial de carreras {home_abbr}", True),
        "season_rd_pg_away": (f"Diferencial de carreras {away_abbr}", False),
        "sp_known_home": (f"Abridor {home_abbr} identificado", True),
        "sp_known_away": (f"Abridor {away_abbr} identificado", False),
        "home_wp_at_home": (f"{home_abbr} en casa (forma local)", True),
        "away_wp_on_road": (f"{away_abbr} de visitante", False),
        "park_rpg": ("Parque (carreras esperadas de ambiente)", True),
        "h2h_home_wp": ("Head-to-head histórico", True),
        "sp_era_home": (f"Abridor {home_abbr} (ERA previa)", True),
        "sp_era_away": (f"Abridor {away_abbr} (ERA previa)", False),
        "sp_era_diff": ("Choque de abridores (ERA)", True),
        "sp_roll_ra_home": (f"Abridor {home_abbr} en starts recientes", True),
        "sp_roll_ra_away": (f"Abridor {away_abbr} en starts recientes", False),
        "is_playoff": ("Contexto playoff", True),
    }
    for i, col in enumerate(cols):
        pp = float(coef[i] * xs[i] * scale)
        if abs(pp) < 0.35:
            continue
        label, home_positive = texts.get(col, (col, True))
        favors_home = pp > 0
        labeled.append((abs(pp), favors_home, pp, label))
    labeled.sort(reverse=True)
    home_rows: list[str] = []
    away_rows: list[str] = []
    n_h = n_a = 0
    for _mag, favors_home, pp, label in labeled:
        sign = f"[+{abs(pp):.1f} pp]"
        line = f"{label}  {sign}"
        if favors_home:
            if n_h < 5:
                home_rows.append(line)
                n_h += 1
        else:
            if n_a < 5:
                away_rows.append(line)
                n_a += 1
    return home_rows, away_rows


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
    tt = bundle.get("totals_holdout") or {}
    if tt:
        print(
            f"  Totales {hold}: MAE={tt.get('mae_total', 0):.2f} carreras  "
            f"media pred {tt.get('mean_pred', 0):.1f} vs real {tt.get('mean_real', 0):.1f}  "
            f"O/U vs línea 8.5: {tt.get('ou_acc_8_5', 0):.3f}  "
            f"std pred {tt.get('pred_std', 0):.2f}  "
            f"%OVER {100 * tt.get('pct_over_8_5', 0):.0f}"
        )
    print(f"  Archivo: {bundle.get('path', MODEL_PATH)}\n")
