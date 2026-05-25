"""
pcap_feature_extractor.py
=========================
Extraction vectorisée de features rythmiques et dimensionnelles
depuis des fichiers PCAP/PCAPNG.

"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import dpkt
from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────
#  HYPERPARAMÈTRES
# ──────────────────────────────────────────────
WINDOW_SECONDS  = 1.0    # largeur de la fenêtre locale (secondes)
BURST_GAP       = 0.1    # seuil de silence pour délimiter les rafales (s)
MIN_PACKETS     = 5      # sessions avec moins de N paquets → rejetées
CAUSAL_WINDOW   = True   # True = fenêtre causale [t-W, t] ; False = symétrique
NORMALIZE       = False  # normaliser les features par session (optionnel)

# Regex de validation du nom de fichier : <prefix>_<id>_<label>_<suffix>.pcapng
FILENAME_RE = re.compile(r'^[^_]+_[^_]+_([^_]+)_.+\.pcapng$')


# ══════════════════════════════════════════════
#  LECTURE PCAP
# ══════════════════════════════════════════════

def read_pcap_file(filepath: str) -> list[dict]:
    """
    Lit un fichier PCAP ou PCAPNG et retourne la liste des paquets IP
    sous forme de dicts {ts, pkt_len, ip_len, src_ip, dst_ip}.

    Retourne une liste vide en cas d'erreur.
    """
    packets = []
    try:
        with open(filepath, 'rb') as f:
            try:
                capture = dpkt.pcap.Reader(f)
            except Exception:
                f.seek(0)
                capture = dpkt.pcapng.Reader(f)

            for ts, buf in capture:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    packets.append({
                        'ts':      float(ts),
                        'pkt_len': len(buf),          # taille trame Ethernet
                        'ip_len':  int(ip.len),        # taille payload IP
                        'src_ip':  str(ip.src),        # bytes → str pour comparaison
                        'dst_ip':  str(ip.dst),
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"  [error] {filepath}: {e}")
    return packets


# ══════════════════════════════════════════════
#  INFÉRENCE DE LA DIRECTION (UP / DOWN)
# ══════════════════════════════════════════════

def infer_direction(packets: list[dict]) -> np.ndarray:
    """
    Détermine la direction de chaque paquet (1 = UP, 0 = DOWN)
    en identifiant l'IP source majoritaire comme l'hôte local.

    Hypothèse : dans une session de capture locale, l'hôte qui
    émet le plus de paquets est le client (direction UP).
    """
    src_ips = [p['src_ip'] for p in packets]
    if not src_ips:
        return np.array([], dtype=np.int8)

    # Compter les occurrences de chaque IP source
    unique, counts = np.unique(src_ips, return_counts=True)
    local_ip = unique[np.argmax(counts)]  # IP la plus fréquente = client

    return np.array([1 if p['src_ip'] == local_ip else 0
                     for p in packets], dtype=np.int8)


# ══════════════════════════════════════════════
#  EXTRACTION VECTORISÉE DES FEATURES
# ══════════════════════════════════════════════

def compute_rhythm_features(packets: list[dict],
                             window_sec: float = WINDOW_SECONDS,
                             burst_gap: float  = BURST_GAP,
                             causal: bool      = CAUSAL_WINDOW) -> np.ndarray:
    """
    Extrait 9 features par paquet de façon vectorisée (O(N log N)).

    Features retournées (colonnes) :
      0  T              : intervalle inter-paquets (s)
      1  local_rate     : paquets/s dans la fenêtre locale
      2  local_density  : fraction d'intervalles actifs dans la fenêtre
      3  burst_position : position relative dans la rafale [0, 1]
      4  silence_before : durée de silence avant ce paquet (s)
      5  pkt_len        : taille de la trame Ethernet (octets)
      6  ip_len         : taille du payload IP (octets)
      7  direction      : 1 = UP (client→serveur), 0 = DOWN
      8  up_ratio_local : ratio paquets UP dans la fenêtre locale
    """
    ts      = np.array([p['ts']      for p in packets], dtype=np.float64)
    pkt_len = np.array([p['pkt_len'] for p in packets], dtype=np.float32)
    ip_len  = np.array([p['ip_len']  for p in packets], dtype=np.float32)
    direction = infer_direction(packets)

    N = len(ts)
    T = np.empty(N, dtype=np.float64)
    T[0]  = 0.0
    T[1:] = np.diff(ts)

    features = np.zeros((N, 9), dtype=np.float64)
    features[:, 0] = T
    features[:, 5] = pkt_len
    features[:, 6] = ip_len
    features[:, 7] = direction

    # ── Détection des rafales ────────────────────────────────────────
    burst_ids    = np.zeros(N, dtype=np.int32)
    burst_ids[1:] = np.cumsum(T[1:] > burst_gap)

    # ── Bornes de fenêtre — O(N log N) via searchsorted ─────────────
    if causal:
        left  = np.searchsorted(ts, ts - window_sec, side='left')
        right = np.arange(N) + 1                      # fenêtre causale : [t-W, t]
    else:
        left  = np.searchsorted(ts, ts - window_sec, side='left')
        right = np.searchsorted(ts, ts + window_sec, side='right')

    window_size = right - left  # nombre de paquets dans chaque fenêtre

    # ── col 1 : local_rate ───────────────────────────────────────────
    elapsed = window_sec if causal else 2 * window_sec
    features[:, 1] = window_size / elapsed

    # ── col 2 : local_density & col 8 : up_ratio_local ──────────────
    for i in range(N):
        l, r = left[i], right[i]
        win_T = T[l:r]
        if len(win_T) > 1:
            features[i, 2] = np.sum(win_T[1:] < burst_gap) / (len(win_T) - 1)
        else:
            features[i, 2] = 0.0

        win_dir = direction[l:r]
        features[i, 8] = np.mean(win_dir) if len(win_dir) > 0 else 0.0

    # ── col 3 : burst_position ───────────────────────────────────────
    for bid in np.unique(burst_ids):
        idx = np.where(burst_ids == bid)[0]
        n   = len(idx)
        if n > 1:
            features[idx, 3] = np.linspace(0.0, 1.0, n)
        # si n == 1 → position reste 0.0

    # ── col 4 : silence_before ───────────────────────────────────────
    features[:, 4] = np.where(T > burst_gap, T, 0.0)

    return features


# ══════════════════════════════════════════════
#  NORMALISATION PAR SESSION (optionnelle)
# ══════════════════════════════════════════════

# Colonnes continues à normaliser (exclure direction binaire col 7)
_COLS_TO_NORMALIZE = [0, 1, 2, 3, 4, 5, 6, 8]

def normalize_session(features: np.ndarray) -> np.ndarray:
    """MinMaxScaler appliqué par session sur les features continues."""
    out = features.copy()
    scaler = MinMaxScaler()
    out[:, _COLS_TO_NORMALIZE] = scaler.fit_transform(
        features[:, _COLS_TO_NORMALIZE]
    )
    return out


# ══════════════════════════════════════════════
#  RAPPORT DE QUALITÉ PAR SESSION
# ══════════════════════════════════════════════

def session_quality_report(packets: list[dict],
                            features: np.ndarray,
                            burst_gap: float = BURST_GAP) -> dict:
    """Calcule des métriques de qualité pour une session."""
    ts = np.array([p['ts'] for p in packets])
    T  = features[:, 0]

    burst_ids = np.zeros(len(ts), dtype=np.int32)
    burst_ids[1:] = np.cumsum(T[1:] > burst_gap)
    n_bursts = int(burst_ids[-1]) + 1 if len(ts) > 0 else 0

    duration    = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    up_count    = int(np.sum(features[:, 7] == 1))
    down_count  = int(np.sum(features[:, 7] == 0))
    mean_pkt_len = float(np.mean(features[:, 5]))

    return {
        'n_packets':    len(packets),
        'duration_s':   round(duration, 4),
        'n_bursts':     n_bursts,
        'up_packets':   up_count,
        'down_packets': down_count,
        'mean_pkt_len': round(mean_pkt_len, 2),
    }


# ══════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

FEATURE_COLS = [
    'T', 'local_rate', 'local_density',
    'burst_position', 'silence_before',
    'pkt_len', 'ip_len', 'direction', 'up_ratio_local'
]

def extract_to_csv(session_dir:  str,
                   output_csv:   str,
                   output_parquet: str | None = None) -> pd.DataFrame:
    """
    Parcourt session_dir, extrait les features de chaque fichier PCAPNG
    et exporte le résultat en CSV (et optionnellement en Parquet).
    """
    rows     = []
    quality_rows = []
    skipped  = 0
    processed = 0

    files = sorted(f for f in os.listdir(session_dir) if f.endswith('.pcapng'))
    print(f"Fichiers trouvés : {len(files)}\n")

    for pcap_file in files:
        filepath   = os.path.join(session_dir, pcap_file)
        session_id = pcap_file.replace('.pcapng', '')

        # ── Validation de la convention de nommage ───────────────────
        match = FILENAME_RE.match(pcap_file)
        if not match:
            warnings.warn(
                f"  [skip] Convention de nommage invalide : {pcap_file}\n"
                f"         Attendu : <prefix>_<id>_<label>_<suffix>.pcapng"
            )
            skipped += 1
            continue
        label = match.group(1)

        print(f"  Lecture {pcap_file} ...", end=' ', flush=True)
        packets = read_pcap_file(filepath)

        if len(packets) < MIN_PACKETS:
            print(f"ignoré ({len(packets)} paquets < seuil {MIN_PACKETS})")
            skipped += 1
            continue

        # ── Extraction des features ──────────────────────────────────
        features = compute_rhythm_features(packets)

        if NORMALIZE:
            features = normalize_session(features)

        # ── Rapport qualité ──────────────────────────────────────────
        qr = session_quality_report(packets, features)
        qr.update({'session_id': session_id, 'label': label})
        quality_rows.append(qr)

        print(
            f"{qr['n_packets']} paquets | "
            f"{qr['duration_s']:.2f}s | "
            f"{qr['n_bursts']} rafales | "
            f"UP={qr['up_packets']} DOWN={qr['down_packets']}"
        )

        # ── Construction des lignes CSV ──────────────────────────────
        for i, feat in enumerate(features):
            row = {
                'session_id':     session_id,
                'label':          label,
                'timestep':       i,
            }
            for col, val in zip(FEATURE_COLS, feat):
                row[col] = round(float(val), 9 if col == 'T' else 6)
            rows.append(row)

        processed += 1

    # ── Assemblage du DataFrame final ────────────────────────────────
    all_cols = ['session_id', 'label', 'timestep'] + FEATURE_COLS
    df = pd.DataFrame(rows, columns=all_cols)

    # Optimisation mémoire : types appropriés
    df['timestep']      = df['timestep'].astype(np.int32)
    df['direction']     = df['direction'].astype(np.int8)
    df['pkt_len']       = df['pkt_len'].astype(np.int32)
    df['ip_len']        = df['ip_len'].astype(np.int32)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    if output_parquet:
        df.to_parquet(output_parquet, index=False)
        print(f"\nParquet exporté : {output_parquet}")

    # ── Rapport de qualité global ────────────────────────────────────
    df_quality = pd.DataFrame(quality_rows)
    quality_path = output_csv.replace('.csv', '_quality_report.csv')
    df_quality.to_csv(quality_path, index=False)

    print(f"\n{'═'*55}")
    print(f"  Sessions traitées : {processed}")
    print(f"  Sessions ignorées : {skipped}")
    print(f"  Lignes totales    : {len(df):,}")
    print(f"  Classes trouvées  : {sorted(df['label'].unique()) if processed else []}")
    print(f"  CSV exporté       : {output_csv}")
    print(f"  Rapport qualité   : {quality_path}")
    print(f"{'═'*55}")
    print("\nAperçu (10 premières lignes) :")
    print(df.head(10).to_string(index=False))

    return df


# ══════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════

if __name__ == "__main__":
    INPUT_DIR        = r'.\datafile\session_data'
    OUTPUT_CSV       = r'.\datafile\extracted\sequences.csv'
    OUTPUT_PARQUET   = r'.\datafile\extracted\sequences.parquet'   # None pour désactiver

    df = extract_to_csv(
        session_dir    = INPUT_DIR,
        output_csv     = OUTPUT_CSV,
        output_parquet = OUTPUT_PARQUET,
    )
