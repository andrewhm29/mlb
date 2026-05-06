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

**Dónde ejecutarlo:** en la carpeta raíz del proyecto (la que contiene `requirements.txt`, `mlb_prediction_model.py` y la carpeta `scripts/`).

```bash
pip install -r requirements.txt   # solo la primera vez (o si cambian dependencias)
python3 scripts/00_menu.py
```

**Comando recomendado:** `python3 scripts/00_menu.py` — añade la raíz del proyecto a `sys.path` y abre el menú interactivo.

**Alternativa equivalente** (solo si ya estás en la raíz del repo):

```bash
python3 mlb_prediction_model.py
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
| **0** | Salir. |

Tras cada acción el programa puede pedir que pulses **Enter** para volver al menú.

#### Salida (consola y CSV)

- En pantalla verás el detalle por partido (probabilidades, comparativas, etc.).
- Además se escribe un archivo en la **raíz del repo**: `predictions_<fecha>.csv`, donde `<fecha>` es la misma fecha que elegiste (mismo formato `YYYY-MM-DD`). Ejemplo: predicciones del 5 de mayo de 2026 → `predictions_2026-05-05.csv`.

## Equipos disponibles

ARI, ATL, BAL, BOS, CHC, CIN, CLE, COL, CWS, DET, HOU, KC, LAA, LAD, MIA, MIL, MIN, NYM, NYY, OAK, PHI, PIT, SD, SEA, SF, STL, TB, TEX, TOR, WAS

## Data Source

Datos proporcionados por la MLB Stats API oficial (gratis, sin clave requerida).
