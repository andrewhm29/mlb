# MLB Prediction Model

Modelo predictivo para partidos de MLB con análisis detallado.

## Características

- Estadísticas de temporada actual e histórica
- Head-to-head entre equipos
- Comparativas: ERA, WHIP, K/9, OPS, Run Differential
- Momentum (últimos 5 partidos)
- Record casa/visita
- Racha actual
- Historial por temporada (últimos años)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Predicciones

Igual que en NHL: un script de calendario **sin menú**. Ejecútalo desde la raíz del proyecto (la carpeta que contiene `requirements.txt` y `scripts/`).

```bash
pip install -r requirements.txt   # solo la primera vez (o si cambian dependencias)
python3 scripts/01_download.py --from-year 2010    # historial + abridores + ERA por temporada
python3 scripts/04_train.py                       # entrena (as-of, con pitcher) y mide 2026
python3 scripts/05_predict.py
python3 scripts/05_predict.py
```

Por defecto usa **hoy** (fecha local del sistema). Para otra fecha:

```bash
python3 scripts/05_predict.py --date 2026-05-05
```

(`YYYY-MM-DD`)

Imprime las predicciones en consola y escribe `predictions_<fecha>.csv` en la raíz del repo.

**Menú interactivo** (hoy / ayer / fecha / sync / exportar CSV):

```bash
python3 scripts/00_menu.py
```

#### Opciones del menú

| Opción | Qué hace |
|--------|-----------|
| **1** | Lista predicciones para todos los partidos **de hoy** (fecha local del sistema). |
| **2** | Igual, pero para **ayer**. |
| **3** | Pide una fecha en consola; usa **`YYYY-MM-DD`** (ej. `2026-05-05`). Enter sin texto = hoy. |
| **4** | Sincroniza datos (partidos finalizados de ayer + estadísticas de equipos). |
| **5** | Misma lógica que **1**, mostrando mensaje de exportación; enfocado en CSV de hoy. |
| **6** | Misma lógica que **2**, para CSV de ayer. |
| **A** | Descarga historial **2010 → hoy** a `data/games_history.csv`. |
| **T** | Entrena el modelo (features sin futuro; holdout = temporada actual). |
| **0** | Salir. |

Tras cada acción el programa puede pedir que pulses **Enter** para volver al menú.

#### Salida (consola y CSV)

- En pantalla verás el detalle por partido (probabilidades, comparativas, etc.).
- Además se escribe un archivo en la **raíz del repo**: `predictions_<fecha>.csv`, donde `<fecha>` es la misma fecha que elegiste (mismo formato `YYYY-MM-DD`). Ejemplo: predicciones del 5 de mayo de 2026 → `predictions_2026-05-05.csv`.

## Equipos disponibles

ARI, ATL, BAL, BOS, CHC, CIN, CLE, COL, CWS, DET, HOU, KC, LAA, LAD, MIA, MIL, MIN, NYM, NYY, OAK, PHI, PIT, SD, SEA, SF, STL, TB, TEX, TOR, WAS

## Data Source (MLB Stats API)

Datos de la [MLB Stats API](https://statsapi.mlb.com/api/v1) oficial. **Gratis, sin API key.** El cliente está en `mlb_api.py`.

| Endpoint | Uso |
|----------|-----|
| `GET /api/v1/schedule?sportId=1&season=YYYY&gameTypes=R,F,D,L,W` | **Historial por temporada** (2010, 2011, …). Regular + playoffs |
| `GET /api/v1/schedule?...&hydrate=probablePitcher,team,linescore,venue` | Calendario del día **con abridores** |
| `GET /api/v1/people/{id}/stats?stats=season&group=pitching` | ERA / WHIP / K9 del abridor |
| `GET /api/v1/teams/stats?...&stats=byDateRange` | Forma reciente (últimos 30 días) |
| `GET /api/v1/teams/stats?group=hitting&stats=season` | OPS, OBP, SLG, carreras de temporada |
| `GET /api/v1/teams/stats?group=pitching&stats=season` | ERA, WHIP de staff |
| `GET /api/v1/standings?leagueId=103,104` | W-L, casa/visita, racha, run differential |
| `GET /api/v1/game/{gamePk}/boxscore` | Boxscore si el partido ya terminó |

Igual que NHL (`scripts/01_download.py`):

```bash
# Historial de temporadas (la misma Stats API; hay datos desde ~1901).
# Desde 2010 hasta el año actual:
python3 scripts/01_download.py --from-year 2010

# Solo ayer + stats de esta temporada:
python3 scripts/01_download.py
```

El historial queda en `data/games_history.csv` (partidos finalizados, regular + playoffs). El head-to-head de las predicciones usa ese archivo, recortado a **antes** de la fecha del partido.

Luego predicciones:

```bash
python3 scripts/05_predict.py
```
