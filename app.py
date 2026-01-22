from __future__ import annotations

"""
Rural BTS Coverage — ITU-R P.1812 + (opcjonalnie) ITU-R P.2108 (clutter statystyczny)

Aplikacja Streamlit do:
- wyboru BTS (Tx) oraz punktu Rx na mapie,
- obliczenia budżetu łącza (Prx) dla jednej relacji Tx→Rx,
- wygenerowania heatmapy (raster PNG overlay) w promieniu R km wokół Tx.

Model propagacyjny:
- P.1812 (Py1812) jako podstawowe tłumienie Lb na bazie profilu terenu (DEM) i statystyk (p czasu, pL lokalizacji),
- (opcjonalnie) P.2108 jako dodatkowa strata "clutter" Lctt w dB.

Źródło profilu terenu:
- SRTM (biblioteka `srtm`), z cache na dysku.
"""

import base64
import io
import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Dict

import numpy as np
import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from pyproj import Geod
import srtm
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


# Wycisz ostrzeżenia numpy generowane przez Py1812 (czasem pojawiają się ostrzeżenia runtime przy ekstremach)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"Py1812\..*")

# Obiekt geodezyjny do obliczeń odległości/azymutu po elipsoidzie WGS84
WGS84_GEOD = Geod(ellps="WGS84")


# -----------------------------
# MODELE / KONFIGURACJA
# -----------------------------
@dataclass
class LatLon:
    """Prosty nośnik współrzędnych geograficznych."""
    lat: float
    lon: float


@dataclass
class AntennaConfig:
    """
    Uproszczony model anteny sektorowej (nie-ITU):
    - azimuth_deg: kierunek głównej wiązki (0=N, 90=E)
    - beamwidth_deg: szerokość wiązki 3 dB (tu traktowana geometrycznie jako próg)
    - max_gain_dbi: zysk w osi
    - front_to_back_db: maksymalne tłumienie w tył
    """
    azimuth_deg: float = 0.0
    beamwidth_deg: float = 65.0
    max_gain_dbi: float = 15.0
    front_to_back_db: float = 25.0


@dataclass
class RadioConfig:
    """
    Parametry radiowe i modelowe.

    P.1812:
    - f_ghz: częstotliwość [GHz]
    - p_time: % czasu (dla którego poziom jest przekroczony / zależnie od definicji w Py1812)
    - p_loc: % lokalizacji (pL)
    - DN, N0: parametry refrakcji
    - inland_zone: kod strefy (tu: 4 inland)
    - polarization: vertical/horizontal (mapowane na kod dla Py1812)
    - htg_m, hrg_m: wysokości anten nad gruntem AGL [m]

    Budżet:
    - eirp_dbm: EIRP [dBm]
    - gr_dbi: zysk Rx [dBi]

    P.2108:
    - use_p2108: czy dodawać Lctt
    - p2108_p_loc: % lokalizacji dla clutter w P.2108

    DEM:
    - profile_step_m: co ile metrów próbkować profil
    - cache_dir: katalog cache dla SRTM
    """
    mode: str = "LTE"  # LTE / 5G
    f_ghz: float = 0.8
    eirp_dbm: float = 46.0
    gr_dbi: float = 0.0
    htg_m: float = 30.0
    hrg_m: float = 1.5
    polarization: str = "vertical"  # vertical/horizontal
    p_time: float = 50.0
    p_loc: float = 50.0
    DN: float = 45.0
    N0: float = 325.0
    inland_zone: int = 4  # 4 = inland
    use_p2108: bool = True
    p2108_p_loc: float = 50.0
    profile_step_m: float = 90.0
    cache_dir: Optional[str] = "./.srtm_cache"


@dataclass
class HeatmapConfig:
    """
    Parametry siatki heatmapy:
    - radius_km: promień wokół Tx
    - grid_step_m: rozdzielczość siatki (im mniejsza, tym dużo wolniej)
    """
    radius_km: float = 10.0
    grid_step_m: float = 200.0


@dataclass
class HeatRenderConfig:
    """
    Konfiguracja renderowania rastra:
    - vmin_dbm / vmax_dbm: mapowanie dBm -> [0..1]
    - style_mode: colormap / single
    - colormap_name: nazwa colormap matplotlib
    - single_color_hex: kolor dla trybu single
    - opacity: globalny mnożnik alpha
    """
    vmin_dbm: float = -130.0
    vmax_dbm: float = -70.0
    style_mode: str = "colormap"  # "colormap" lub "single"
    colormap_name: str = "turbo"
    single_color_hex: str = "#ff0000"
    opacity: float = 0.85


# -----------------------------
# GEOMETRIA: bearing & zysk sektora
# -----------------------------
def bearing_deg(a: LatLon, b: LatLon) -> float:
    """
    Azymut (bearing) z punktu a do b w stopniach geograficznych (0=N, 90=E).
    Liczone klasycznie z trygonometrii sferycznej.
    """
    phi1 = math.radians(a.lat)
    phi2 = math.radians(b.lat)
    dlam = math.radians(b.lon - a.lon)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def wrap_angle_deg(x: float) -> float:
    """Normalizacja kąta do zakresu (-180, +180]."""
    return (x + 180.0) % 360.0 - 180.0


def tx_gain_sector(tx: LatLon, rx: LatLon, ant: AntennaConfig) -> float:
    """
    Uproszczony zysk anteny sektorowej:
    - w wiązce (|delta| <= BW/2): max_gain
    - poza wiązką: liniowe narastanie tłumienia aż do front_to_back_db

    To NIE jest model ITU, tylko praktyczny "pokazowy" model kierunkowości.
    """
    brng = bearing_deg(tx, rx)
    delta = wrap_angle_deg(brng - ant.azimuth_deg)
    half_bw = ant.beamwidth_deg / 2.0

    if abs(delta) <= half_bw:
        att = 0.0
    else:
        att = min(ant.front_to_back_db, ant.front_to_back_db * (abs(delta) - half_bw) / (180.0 - half_bw))

    return ant.max_gain_dbi - att


# -----------------------------
# DEM: SRTM profil + cache
# -----------------------------
def configure_srtm_cache(cache_dir: Optional[str]) -> None:
    """
    Ustawienie cache SRTM przez zmienną środowiskową wykorzystywaną przez bibliotekę `srtm`.
    Dzięki temu przy heatmapie nie pobierasz kafli w kółko.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["SRTM_CACHE_DIR"] = cache_dir


@lru_cache(maxsize=8)
def _get_srtm_data(cache_dir_key: str):
    """
    Trzymamy `srtm.get_data()` w LRU cache, żeby nie inicjalizować pobierania wielokrotnie.
    cache_dir_key jest w kluczu cache (zmiana folderu = nowy wpis).
    """
    configure_srtm_cache(cache_dir_key if cache_dir_key else None)
    return srtm.get_data()


@st.cache_data(show_spinner=False)
def build_profile_srtm(
    tx: LatLon, rx: LatLon, step_m: float, cache_dir: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Buduje profil terenu Tx→Rx z DEM (SRTM).

    Zwraca:
      di_km: odległości od Tx [km]
      h_asl: wysokości terenu [m n.p.m.]

    WAŻNE:
    - Py1812 wymaga > 4 punktów profilu => minimum 5.
    - Przy bardzo krótkich dystansach (dist_m < 1 m) robimy wymuszone 5 punktów.
    """
    data = _get_srtm_data(cache_dir or "")

    az12, _, dist_m = WGS84_GEOD.inv(tx.lon, tx.lat, rx.lon, rx.lat)
    dist_m = float(dist_m)

    if dist_m < 1.0:
        # ekstremalnie krótki dystans: wymuś 5 punktów, aby Py1812 nie poleciał wyjątkiem
        frac = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        lats = tx.lat + (rx.lat - tx.lat) * frac
        lons = tx.lon + (rx.lon - tx.lon) * frac
        di_km = (frac * dist_m) / 1000.0
    else:
        # ile punktów potrzebujemy dla kroku step_m + 1 punkt końcowy
        n = max(5, int(math.ceil(dist_m / step_m)) + 1)
        frac = np.linspace(0.0, 1.0, n)

        # Punkty po wielkim kole od Tx, azymutem az12, dla dystansu frac * dist_m
        lons, lats = WGS84_GEOD.fwd(
            np.full(n, tx.lon),
            np.full(n, tx.lat),
            np.full(n, az12),
            frac * dist_m,
        )[0:2]
        lats = np.array(lats, dtype=float)
        lons = np.array(lons, dtype=float)
        di_km = (frac * dist_m) / 1000.0

    # Pobranie wysokości z SRTM (brak danych -> 0.0)
    h = []
    for lat, lon in zip(lats, lons):
        elev = data.get_elevation(float(lat), float(lon))
        h.append(float(elev) if elev is not None else 0.0)

    return np.asarray(di_km, dtype=float), np.asarray(h, dtype=float)


# -----------------------------
# ITU-R P.2108: clutter loss (statystyczny)
# -----------------------------
def qinv(p: float) -> float:
    """
    Q^{-1}(p) dla rozkładu normalnego: inverse survival function.
    W praktyce: norm.isf(p).
    """
    return float(norm.isf(p))


def p2108_lctt_db(f_ghz: float, d_km: float, p_loc_percent: float) -> float:
    """
    Statystyczna strata clutter (terrestrial) z ITU-R P.2108.

    Parametry:
    - f_ghz: częstotliwość w GHz
    - d_km: długość ścieżki w km (Tx→Rx)
    - p_loc_percent: procent lokalizacji (0..100) dla statystyki

    Zwraca: Lctt [dB]

    Uwaga: w P.2108 dla d > 2 km stosujemy ograniczenie min(L(d), L(2 km)).
    """
    if not (0.0 < p_loc_percent < 100.0):
        raise ValueError("P.2108: percentage locations must be 0 < p < 100")
    if d_km <= 0:
        return 0.0

    f = float(f_ghz)
    d = float(d_km)
    p = float(p_loc_percent) / 100.0

    # Składowa "low loss" (Ll) i "suburban/short" (Ls) + odchylenia standardowe
    Ll = -2.0 * math.log10((10 ** (-5.0 * math.log10(f) - 12.5)) + (10 ** (-16.5)))
    sig_l = 4.0

    Ls = 32.98 + 23.9 * math.log10(d) + 3.0 * math.log10(f)
    sig_s = 6.0

    # Kombinacja w domenie liniowej (10^(-0.2L))
    a = 10 ** (-0.2 * Ll)
    b = 10 ** (-0.2 * Ls)

    sig_cb = math.sqrt((sig_l ** 2) * a + (sig_s ** 2) * b) / (a + b)
    Lctt = -5.0 * math.log10(a + b) - sig_cb * qinv(p)

    # Ograniczenie dla d > 2 km
    if d > 2.0:
        Lctt2 = p2108_lctt_db(f_ghz, 2.0, p_loc_percent)
        Lctt = min(Lctt, Lctt2)

    return float(Lctt)


# -----------------------------
# ITU-R P.1812 przez Py1812
# -----------------------------
def pol_to_int(pol: str) -> int:
    """
    Mapowanie polaryzacji na kod Py1812:
    - horizontal -> 1
    - vertical   -> 2
    """
    return 1 if pol == "horizontal" else 2


def p1812_basic_loss_db(
    f_ghz: float,
    p_time: float,
    p_loc: float,
    di_km: np.ndarray,
    h_asl: np.ndarray,
    tx: LatLon,
    rx: LatLon,
    htg_m: float,
    hrg_m: float,
    pol: str,
    DN: float,
    N0: float,
    inland_zone: int,
) -> float:
    """
    Oblicza Lb (basic transmission loss) wg P.1812, korzystając z implementacji Py1812.

    Wejście:
    - profil: di_km i h_asl (ASL)
    - htg/hrg: wysokości AGL po stronie Tx/Rx
    - p_time i p_loc: statystyki czasu i lokalizacji
    - DN, N0: parametry refrakcji
    - inland_zone: kod strefy

    Uwaga praktyczna:
    - Py1812 wymaga min. 5 punktów profilu.
    """
    from Py1812 import P1812

    d = np.asarray(di_km, dtype=float)
    h = np.asarray(h_asl, dtype=float)

    if len(d) <= 4:
        raise ValueError("Py1812: The number of points in path profile should be larger than 4")

    # R, Ct, zone: wektory tej samej długości co profil (tu: uproszczenie)
    R = np.zeros_like(h)               # rural/open
    Ct = np.zeros_like(h, dtype=int)
    zone = np.full_like(h, inland_zone, dtype=int)

    Lb, _Ep = P1812.bt_loss(
        float(f_ghz),
        float(p_time),
        d,
        h,
        R,
        Ct,
        zone,
        float(htg_m),
        float(hrg_m),
        int(pol_to_int(pol)),
        float(tx.lat),
        float(rx.lat),
        float(tx.lon),
        float(rx.lon),
        pL=float(p_loc),
        DN=float(DN),
        N0=float(N0),
    )
    return float(Lb)


# -----------------------------
# Obliczenie punktu Tx→Rx
# -----------------------------
def compute_point(tx: LatLon, rx: LatLon, ant: AntennaConfig, rc: RadioConfig) -> Dict[str, object]:
    """
    Liczy wszystkie składowe dla pojedynczego punktu Rx:
    - profil DEM
    - Lb z P.1812
    - (opcjonalnie) Lctt z P.2108
    - zysk sektora Tx (model aplikacji)
    - końcowo Prx (dBm)

    Dodatkowo przygotowuje dane do wykresu:
    - teren (ASL)
    - prosta LOS łącząca wysokości anten (ASL) — “wizualna linia” Tx→Rx
    """
    di_km, h_asl = build_profile_srtm(tx, rx, rc.profile_step_m, rc.cache_dir)

    lb = p1812_basic_loss_db(
        f_ghz=rc.f_ghz,
        p_time=rc.p_time,
        p_loc=rc.p_loc,
        di_km=di_km,
        h_asl=h_asl,
        tx=tx,
        rx=rx,
        htg_m=rc.htg_m,
        hrg_m=rc.hrg_m,
        pol=rc.polarization,
        DN=rc.DN,
        N0=rc.N0,
        inland_zone=rc.inland_zone,
    )

    # Dodatkowa strata clutter (statystyczna), jeśli włączona
    lctt = 0.0
    if rc.use_p2108:
        lctt = p2108_lctt_db(rc.f_ghz, float(di_km[-1]), rc.p2108_p_loc)

    # Zysk Tx zależny od kierunku (sektor)
    gt = tx_gain_sector(tx, rx, ant)

    # Budżet łącza (upraszczamy: EIRP + Gt + Gr - Lb - Lctt)
    prx = rc.eirp_dbm + gt + rc.gr_dbi - lb - lctt

    # Prosta łącząca wysokości anten (ASL) — tylko do poglądowego wykresu
    tx_asl = float(h_asl[0] + rc.htg_m)
    rx_asl = float(h_asl[-1] + rc.hrg_m)
    los_line = tx_asl + (rx_asl - tx_asl) * (di_km / float(di_km[-1] if di_km[-1] > 0 else 1.0))

    return {
        "prx_dbm": float(prx),
        "basic_loss_db": float(lb),
        "clutter_loss_db": float(lctt),
        "tx_gain_dbi": float(gt),
        "rx_gain_dbi": float(rc.gr_dbi),
        "distance_km": float(di_km[-1]),
        "n_profile_points": int(len(di_km)),
        "profile_di_km": di_km,
        "profile_h_asl": h_asl,
        "profile_los_asl": los_line,
        "tx_ant_asl": tx_asl,
        "rx_ant_asl": rx_asl,
    }


# -----------------------------
# Siatka heatmapy
# -----------------------------
def build_grid_bounds(
    tx: LatLon, radius_km: float, step_m: float
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Buduje siatkę (lat/lon) w kwadracie obejmującym okrąg o promieniu radius_km.

    Uwaga:
    - konwersja metrów -> stopnie jest przybliżona (dla małych obszarów OK),
    - potem i tak maskujemy punkty do koła (inside_circle).
    """
    lat0, lon0 = tx.lat, tx.lon
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * math.cos(math.radians(lat0)))

    r_m = radius_km * 1000.0
    dlat = step_m * deg_per_m_lat
    dlon = step_m * deg_per_m_lon
    half_lat = r_m * deg_per_m_lat
    half_lon = r_m * deg_per_m_lon

    lat_min = lat0 - half_lat
    lat_max = lat0 + half_lat
    lon_min = lon0 - half_lon
    lon_max = lon0 + half_lon

    lats = np.arange(lat_min, lat_max + dlat, dlat)
    lons = np.arange(lon_min, lon_max + dlon, dlon)
    return lats, lons, lat_min, lat_max, lon_min, lon_max


def inside_circle(tx: LatLon, lat: float, lon: float, radius_m: float) -> bool:
    """
    Proste sprawdzenie czy punkt mieści się w kole (płaska aproksymacja lokalna).
    Wystarcza do maskowania siatki heatmapy.
    """
    dy = (lat - tx.lat) * 111_320.0
    dx = (lon - tx.lon) * 111_320.0 * math.cos(math.radians(tx.lat))
    return (dx * dx + dy * dy) <= (radius_m * radius_m)


# -----------------------------
# Render rastra (PNG) dla folium overlay
# -----------------------------
def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    """Konwersja #RRGGBB -> (r,g,b) w skali 0..1."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def make_colormap(render: HeatRenderConfig):
    """
    Buduje colormap:
    - tryb 'single': jeden kolor, alpha budowana osobno
    - tryb 'colormap': standardowe mapy matplotlib
    """
    if render.style_mode == "single":
        r, g, b = hex_to_rgb01(render.single_color_hex)
        return LinearSegmentedColormap.from_list(
            "single_color",
            [(r, g, b, 0.0), (r, g, b, 1.0)],
            N=256,
        )
    return cm.get_cmap(render.colormap_name)


def raster_png_from_grid(grid_dbm: np.ndarray, render: HeatRenderConfig) -> bytes:
    """
    Tworzy raster PNG z macierzy dBm.

    Zasada:
    - dBm mapujemy do [0..1] wg vmin/vmax
    - kolor pochodzi z colormap lub single-color
    - alpha zależy od intensywności (normv) i globalnego opacity
    - NaN -> przezroczyste (alpha=0)

    grid_dbm: (H, W), NaN = brak piksela
    """
    vmin, vmax = render.vmin_dbm, render.vmax_dbm
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0

    normv = (grid_dbm - vmin) / denom
    normv = np.clip(normv, 0.0, 1.0)
    normv = np.where(np.isfinite(grid_dbm), normv, np.nan)

    cmap = make_colormap(render)
    rgba = cmap(np.nan_to_num(normv, nan=0.0))  # (H,W,4)

    # Alpha = intensywność * opacity
    alpha = np.where(np.isfinite(normv), normv, 0.0) * float(render.opacity)
    rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

    # Render do PNG przez matplotlib (transparent background)
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


def colorbar_png(render: HeatRenderConfig) -> bytes:
    """
    Tworzy mały pasek legendy (PNG) dla colormap/single-color.
    """
    cmap = make_colormap(render)
    fig, ax = plt.subplots(figsize=(5, 0.7), dpi=200)
    fig.subplots_adjust(bottom=0.5)

    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=cmap)
    ax.set_axis_off()

    ax2 = fig.add_axes([0.1, 0.0, 0.8, 0.3])
    ax2.axis("off")
    ax2.text(0.0, 0.1, f"{render.vmin_dbm:.0f} dBm", fontsize=9, va="bottom")
    ax2.text(1.0, 0.1, f"{render.vmax_dbm:.0f} dBm", fontsize=9, va="bottom", ha="right")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


# -----------------------------
# Worker dla wątków heatmapy
# -----------------------------
def _heat_worker(
    tx: LatLon,
    ant: AntennaConfig,
    rc: RadioConfig,
    i: int,
    j: int,
    lat: float,
    lon: float,
) -> Tuple[int, int, float]:
    """
    Jedno zadanie dla ThreadPool:
    - liczy Prx w punkcie (lat, lon)
    - zwraca indeks (i,j), żeby wypełnić macierz wynikową grid[H,W]
    """
    rx = LatLon(lat, lon)
    r = compute_point(tx, rx, ant, rc)
    return i, j, float(r["prx_dbm"])


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Rural BTS Coverage (P.1812 + P.2108)", layout="wide")
st.title("Rural BTS Coverage — ITU-R P.1812 + ITU-R P.2108 (statistical clutter)")

# Stan aplikacji (kliknięcia na mapie, cache heatmapy itd.)
if "tx" not in st.session_state:
    st.session_state.tx = None
if "rx" not in st.session_state:
    st.session_state.rx = None
if "heat_png_b64" not in st.session_state:
    st.session_state.heat_png_b64 = None
if "heat_bounds" not in st.session_state:
    st.session_state.heat_bounds = None
if "last_point_result" not in st.session_state:
    st.session_state.last_point_result = None


# ---------------
# Sidebar: parametry modelu i renderu
# ---------------
with st.sidebar:
    st.header("Parametry")

    st.markdown(
        """
        **Jak to czytać:**
        - P.1812 liczy tłumienie na podstawie **profilu terenu** + statystyk (**p czasu**, **pL lokalizacji**) i refrakcji (**DN**, **N0**).
        - P.2108 (opcjonalnie) dodaje **statystyczną stratę clutter** jako narzut w dB.
        - Antena (sektor) w tej aplikacji to prosty model kierunkowości (nie norma ITU).
        """
    )

    mode = st.selectbox(
        "Tryb",
        ["LTE", "5G (sub-6)"],
        help=(
            "Preset ułatwiający dobór częstotliwości startowej. "
            "Model propagacji jest ten sam; różni się głównie f (GHz)."
        ),
    )

    if mode.startswith("LTE"):
        f_ghz = st.number_input(
            "Częstotliwość (GHz)",
            value=0.8,
            step=0.1,
            format="%.3f",
            help=(
                "Częstotliwość nośna. W P.1812 wpływa na tłumienie: "
                "wyższa f zwykle oznacza większe straty."
            ),
        )
    else:
        f_ghz = st.number_input(
            "Częstotliwość (GHz)",
            value=3.5,
            step=0.1,
            format="%.3f",
            help="Częstotliwość nośna dla 5G sub-6. Wyższa f zwykle zmniejsza zasięg.",
        )

    eirp_dbm = st.number_input(
        "EIRP (dBm)",
        value=46.0,
        step=0.5,
        help=(
            "EIRP = efektywna moc promieniowana izotropowo. "
            "+3 dB ≈ 2× większa moc."
        ),
    )
    gr_dbi = st.number_input(
        "Gr (dBi)",
        value=0.0,
        step=0.5,
        help="Zysk anteny Rx. Telefon ~0 dBi, CPE może mieć 8–16 dBi.",
    )

    st.subheader("Wysokości AGL")
    htg_m = st.number_input("htg Tx (m)", value=30.0, step=1.0)
    hrg_m = st.number_input("hrg Rx (m)", value=1.5, step=0.5)

    st.subheader("Antena (model sektorowy)")
    az = st.number_input("Azymut (deg)", value=0.0, step=1.0)
    bw = st.number_input("Beamwidth (deg)", value=65.0, step=1.0)
    gt = st.number_input("Max gain (dBi)", value=15.0, step=0.5)
    ftb = st.number_input("Front-to-back (dB)", value=25.0, step=0.5)

    st.subheader("P.1812")
    pol = st.selectbox("Polaryzacja", ["vertical", "horizontal"])
    p_time = st.number_input("p (czas %)", value=50.0, min_value=1.0, max_value=50.0, step=1.0)
    p_loc = st.number_input("pL (lokalizacje %)", value=50.0, min_value=1.0, max_value=99.0, step=1.0)

    st.subheader("Refrakcja (DN, N0)")
    DN = st.number_input("DN (N-units/km)", value=45.0, step=1.0)
    N0 = st.number_input("N0 (N-units)", value=325.0, step=1.0)

    st.subheader("P.2108 (clutter statystyczny)")
    use_p2108 = st.toggle("Włącz P.2108 clutter", value=True)
    p2108_p = st.number_input("p (%) dla P.2108", value=50.0, min_value=1.0, max_value=99.0, step=1.0)

    st.subheader("Profil terenu / cache")
    profile_step_m = st.number_input("Krok profilu (m)", value=90.0, min_value=30.0, max_value=1000.0, step=10.0)
    cache_dir = st.text_input("Cache SRTM (katalog)", value="./.srtm_cache")

    st.subheader("Heatmapa – siatka")
    radius_km = st.number_input("Promień (km)", value=10.0, min_value=0.5, max_value=50.0, step=0.5)
    grid_step_m = st.number_input("Krok siatki (m)", value=200.0, min_value=25.0, max_value=1000.0, step=25.0)

    st.subheader("Heatmapa – wygląd (raster)")
    vmin_dbm = st.number_input("Zakres: vmin (dBm)", value=-130.0, step=1.0)
    vmax_dbm = st.number_input("Zakres: vmax (dBm)", value=-70.0, step=1.0)
    opacity = st.slider("Opacity", 0.0, 1.0, 0.85, 0.01)

    style_mode = st.selectbox("Kolorowanie", ["colormap", "single"])
    if style_mode == "colormap":
        colormap_name = st.selectbox("Kolormap", ["turbo", "viridis", "plasma", "inferno", "magma", "cividis"])
        single_color_hex = "#ff0000"
    else:
        single_color_hex = st.color_picker("Kolor heatmapy", value="#ff0000")
        colormap_name = "turbo"

    st.subheader("Przyspieszenie heatmapy")
    max_workers = st.slider(
        "Wątki (workers)",
        min_value=1,
        max_value=max(1, (os.cpu_count() or 4)),
        value=min(8, max(1, (os.cpu_count() or 4))),
        step=1,
    )
    show_eta = st.toggle("Pokazuj ETA (pozostały czas)", value=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Wyczyść BTS/Rx"):
            st.session_state.tx = None
            st.session_state.rx = None
            st.session_state.heat_png_b64 = None
            st.session_state.heat_bounds = None
            st.session_state.last_point_result = None
            st.rerun()
    with col_b:
        if st.button("Wyczyść heatmapę"):
            st.session_state.heat_png_b64 = None
            st.session_state.heat_bounds = None
            st.rerun()


# Złożenie konfiguracji na podstawie UI
ant = AntennaConfig(azimuth_deg=az, beamwidth_deg=bw, max_gain_dbi=gt, front_to_back_db=ftb)
rc = RadioConfig(
    mode="LTE" if mode.startswith("LTE") else "5G",
    f_ghz=f_ghz,
    eirp_dbm=eirp_dbm,
    gr_dbi=gr_dbi,
    htg_m=htg_m,
    hrg_m=hrg_m,
    polarization=pol,
    p_time=p_time,
    p_loc=p_loc,
    DN=DN,
    N0=N0,
    inland_zone=4,
    use_p2108=use_p2108,
    p2108_p_loc=p2108_p,
    profile_step_m=profile_step_m,
    cache_dir=cache_dir if cache_dir.strip() else None,
)
hm = HeatmapConfig(radius_km=radius_km, grid_step_m=grid_step_m)
hr = HeatRenderConfig(
    vmin_dbm=float(vmin_dbm),
    vmax_dbm=float(vmax_dbm),
    style_mode=style_mode,
    colormap_name=colormap_name,
    single_color_hex=single_color_hex,
    opacity=float(opacity),
)

left, right = st.columns([1.25, 1])

# ---------------
# Lewa kolumna: mapa + kliknięcia
# ---------------
with left:
    st.subheader("Mapa (kliknij: BTS, potem Rx)")
    center = [52.2, 21.0]
    if st.session_state.tx:
        center = [st.session_state.tx.lat, st.session_state.tx.lon]

    m = folium.Map(location=center, zoom_start=10, control_scale=True, tiles="OpenStreetMap")

    # Jeśli heatmapa wygenerowana: nakładamy raster jako ImageOverlay
    if st.session_state.heat_png_b64 and st.session_state.heat_bounds:
        img_url = "data:image/png;base64," + st.session_state.heat_png_b64
        bounds = st.session_state.heat_bounds
        ImageOverlay(
            image=img_url,
            bounds=bounds,
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=10,
        ).add_to(m)

    # Markery Tx/Rx
    if st.session_state.tx:
        folium.Marker(
            [st.session_state.tx.lat, st.session_state.tx.lon],
            tooltip="BTS (Tx)",
            icon=folium.Icon(color="red", icon="signal", prefix="fa"),
        ).add_to(m)

    if st.session_state.rx:
        folium.Marker(
            [st.session_state.rx.lat, st.session_state.rx.lon],
            tooltip="Rx",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    out = st_folium(m, width=None, height=640)

    # Logika kliknięć:
    # - pierwszy klik ustawia Tx,
    # - kolejny klik ustawia Rx,
    # - zmiana Tx resetuje heatmapę i wynik punktu.
    clicked = out.get("last_clicked")
    if clicked:
        p = LatLon(clicked["lat"], clicked["lng"])
        if st.session_state.tx is None:
            st.session_state.tx = p
            st.session_state.rx = None
            st.session_state.heat_png_b64 = None
            st.session_state.heat_bounds = None
            st.session_state.last_point_result = None
            st.rerun()
        else:
            st.session_state.rx = p
            st.rerun()

# ---------------
# Prawa kolumna: wyniki punktu + generacja heatmapy
# ---------------
with right:
    st.subheader("Wyniki")

    if st.session_state.tx is None:
        st.info("Kliknij na mapie, aby ustawić BTS (Tx).")
    else:
        st.success(f"Tx: {st.session_state.tx.lat:.6f}, {st.session_state.tx.lon:.6f}")

    # Wynik punktowy (Tx->Rx)
    if st.session_state.tx and st.session_state.rx:
        st.write(f"Rx: {st.session_state.rx.lat:.6f}, {st.session_state.rx.lon:.6f}")

        try:
            with st.spinner("Liczenie punktu (P.1812 + P.2108 + DEM)..."):
                res = compute_point(st.session_state.tx, st.session_state.rx, ant, rc)
            st.session_state.last_point_result = res

            st.metric("Prx (dBm)", f"{res['prx_dbm']:.1f}")
            st.write(
                {
                    "Lb_P1812_dB": round(res["basic_loss_db"], 2),
                    "Lclutter_P2108_dB": round(res["clutter_loss_db"], 2),
                    "Gt_sector_dBi": round(res["tx_gain_dbi"], 2),
                    "Gr_dBi": round(res["rx_gain_dbi"], 2),
                    "distance_km": round(res["distance_km"], 3),
                    "profile_points": res["n_profile_points"],
                }
            )

            # Profil terenu + linia łącząca anteny (ASL)
            st.subheader("Profil terenu + linia Tx→Rx (ASL)")
            di_km = res["profile_di_km"]
            h_asl = res["profile_h_asl"]
            los_asl = res["profile_los_asl"]

            fig = plt.figure(figsize=(7, 3), dpi=140)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(di_km, h_asl, label="Teren (m ASL)")
            ax.plot(di_km, los_asl, label="Prosta Tx→Rx (anteny ASL)")
            ax.set_xlabel("Odległość od Tx [km]")
            ax.set_ylabel("Wysokość [m n.p.m.]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            st.pyplot(fig, clear_figure=True)

        except Exception as e:
            st.error(str(e))

    # Heatmapa
    st.subheader("Heatmapa (raster overlay)")

    if st.session_state.tx is None:
        st.info("Ustaw BTS (Tx), żeby generować heatmapę.")
    else:
        if st.button("Generuj heatmapę"):
            try:
                tx = st.session_state.tx

                # Siatka w kwadracie, maska do koła
                lats, lons, lat_min, lat_max, lon_min, lon_max = build_grid_bounds(tx, hm.radius_km, hm.grid_step_m)
                H = len(lats)
                W = len(lons)

                grid = np.full((H, W), np.nan, dtype=float)

                radius_m = hm.radius_km * 1000.0
                mask = np.zeros((H, W), dtype=bool)
                for i, lat in enumerate(lats):
                    for j, lon in enumerate(lons):
                        mask[i, j] = inside_circle(tx, float(lat), float(lon), radius_m)

                # Lista zadań tylko dla punktów w kole
                tasks = []
                for i, lat in enumerate(lats):
                    for j, lon in enumerate(lons):
                        if mask[i, j]:
                            tasks.append((i, j, float(lat), float(lon)))

                total = len(tasks)
                if total == 0:
                    raise ValueError("Za mało punktów w siatce (spróbuj zwiększyć promień albo zmniejszyć krok).")

                # Progress + ETA
                prog = st.progress(0, text=f"Liczenie heatmapy: 0 / {total}")
                done = 0
                t0 = time.time()
                last_update = 0.0

                # Równoległe liczenie punktów heatmapy
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [
                        ex.submit(_heat_worker, tx, ant, rc, i, j, lat, lon)
                        for (i, j, lat, lon) in tasks
                    ]

                    for fut in as_completed(futures):
                        i, j, prx_dbm = fut.result()
                        grid[i, j] = prx_dbm
                        done += 1

                        now = time.time()
                        if now - last_update >= 0.2 or done == total:
                            frac = done / total
                            if show_eta and done > 0:
                                elapsed = now - t0
                                sec_per = elapsed / done
                                eta = sec_per * (total - done)
                                eta_txt = f"{int(eta//60)}m {int(eta%60)}s" if eta >= 60 else f"{int(eta)}s"
                                prog.progress(frac, text=f"Liczenie heatmapy: {done} / {total} | ETA: {eta_txt}")
                            else:
                                prog.progress(frac, text=f"Liczenie heatmapy: {done} / {total}")
                            last_update = now

                # Render rastra -> base64 -> overlay na folium
                png = raster_png_from_grid(grid, hr)
                st.session_state.heat_png_b64 = base64.b64encode(png).decode("ascii")
                st.session_state.heat_bounds = [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]

                prog.progress(1.0, text=f"Gotowe: {total} / {total}")
                st.success("Heatmapa wygenerowana (raster).")

                cb = colorbar_png(hr)
                st.image(cb, caption="Legenda (kolorowanie wg vmin/vmax)", use_container_width=True)

            except Exception as e:
                st.error(str(e))

    st.caption(
        "Heatmapa jest rastrem PNG z przezroczystością zależną od poziomu sygnału. "
        "Ustaw vmin/vmax w dBm, żeby dopasować kontrast."
    )
