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

```bash
python3 mlb_prediction_model.py
```

Selecciona:
1. Temporada (1-5)
2. Equipo local (ej: bos, nyy, lad)
3. Equipo visitante (ej: nyy, bos, lad)

## Equipos disponibles

ARI, ATL, BAL, BOS, CHC, CIN, CLE, COL, CWS, DET, HOU, KC, LAA, LAD, MIA, MIL, MIN, NYM, NYY, OAK, PHI, PIT, SD, SEA, SF, STL, TB, TEX, TOR, WAS

## Data Source

Datos proporcionados por la MLB Stats API oficial (gratis, sin clave requerida).
