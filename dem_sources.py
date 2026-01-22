# dem_sources.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import rasterio
from rasterio.warp import transform
from shapely.geometry import box, mapping
from pystac_client import Client
import planetary_computer as pc



@dataclass(frozen=True)
class DemDebug:
    source: str
    crs: Optional[str]
    nodata: Optional[float]
    used_assets: List[str]
    last_bbox_wgs84: Optional[Tuple[float, float, float, float]]  # minlon,minlat,maxlon,maxlat
    notes: Dict[str, Any]


class IDemSource:
    def height_at(self, lat: float, lon: float, *, strict: bool = False) -> Optional[float]:
        raise NotImplementedError

    def heights_at(self, lats: np.ndarray, lons: np.ndarray, *, strict: bool = False) -> np.ndarray:
        out = np.full(lats.shape, np.nan, dtype=np.float32)
        it = np.nditer(lats, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            h = self.height_at(float(lats[idx]), float(lons[idx]), strict=strict)
            out[idx] = np.nan if h is None else np.float32(h)
            it.iternext()
        return out

    def debug_info(self) -> DemDebug:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalGeoTiffDem(IDemSource):
    """
    Lokalne DEM GeoTIFF, jak wcześniej.
    """
    def __init__(self, path: str):
        self.path = path
        self.ds = rasterio.open(path)
        self.band = 1
        self._debug = DemDebug(
            source=f"local:{path}",
            crs=str(self.ds.crs) if self.ds.crs else None,
            nodata=self.ds.nodata,
            used_assets=[path],
            last_bbox_wgs84=None,
            notes={
                "width": self.ds.width,
                "height": self.ds.height,
                "bounds_dem_crs": (self.ds.bounds.left, self.ds.bounds.bottom, self.ds.bounds.right, self.ds.bounds.top),
                "res": self.ds.res,
                "dtype": str(self.ds.dtypes[0]),
            }
        )

    def _to_dem_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        if self.ds.crs is None:
            raise RuntimeError("DEM nie ma CRS.")
        xs, ys = transform("EPSG:4326", self.ds.crs, [lon], [lat])
        return float(xs[0]), float(ys[0])

    def height_at(self, lat: float, lon: float, *, strict: bool = False) -> Optional[float]:
        x, y = self._to_dem_xy(lat, lon)
        left, bottom, right, top = self.ds.bounds
        inside = (left <= x <= right) and (bottom <= y <= top)
        if not inside:
            if strict:
                raise ValueError("Punkt poza DEM (local).")
            return None

        r, c = rasterio.transform.rowcol(self.ds.transform, x, y)
        if r < 0 or r >= self.ds.height or c < 0 or c >= self.ds.width:
            if strict:
                raise ValueError("Indeks poza rastrą (local).")
            return None

        val = self.ds.read(self.band, window=((r, r + 1), (c, c + 1)))[0, 0]
        if self.ds.nodata is not None and np.isclose(val, self.ds.nodata):
            return None
        return float(val)

    def debug_info(self) -> DemDebug:
        return self._debug

    def close(self) -> None:
        try:
            self.ds.close()
        except Exception:
            pass
# dem_sources.py (albo nowy plik)
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, List

from pyhigh import get_elevation, get_elevation_batch  # pyhigh

@dataclass
class DemDebugInfo:
    source: str
    crs: str = "WGS84"
    nodata: Optional[float] = None
    last_bbox_wgs84: Optional[Tuple[float, float, float, float]] = None
    used_assets: Optional[list] = None
    notes: str = ""

class PyHighDem:
    """
    Zamiennik DEM: API kompatybilne z Twoim dem.height_at(...)
    + opcjonalny batch.
    """
    def __init__(self):
        self._last_bbox = None

    def height_at(self, lat: float, lon: float, strict: bool = False) -> Optional[float]:
        try:
            h = get_elevation(lat=lat, lon=lon)   # zwraca metry
            if h is None:
                if strict:
                    raise ValueError("pyhigh: brak danych wysokości")
                return None
            return float(h)
        except Exception:
            if strict:
                raise
            return None

    def height_at_batch(self, points: List[Tuple[float, float]], strict: bool = False) -> List[Optional[float]]:
        """
        points: [(lat, lon), ...]
        """
        try:
            # pyhigh ma batch API (znacznie lepsze niż tysiące single-call)
            hs = get_elevation_batch(points)
            out = []
            for h in hs:
                if h is None:
                    out.append(None if not strict else float("nan"))
                else:
                    out.append(float(h))
            return out
        except Exception:
            if strict:
                raise
            return [None] * len(points)

    def debug_info(self) -> DemDebugInfo:
        return DemDebugInfo(
            source="pyhigh (auto-download + cache HGT)",
            notes="pyhigh pobiera i cachuje dane wysokości automatycznie."
        )


class PlanetaryCopDem(IDemSource):
    """
    Zdalny Copernicus DEM GLO-30 przez Microsoft Planetary Computer (STAC + COG).
    Strategia:
      - dla punktu budujemy mały bbox (bufor),
      - STAC search: kolekcja cop-dem-glo-30,
      - otwieramy 1..N COG-ów (rasterio.open na HTTP),
      - próbkujemy wysokość.
    Z cache:
      - trzymamy otwarte dataset’y per "grid id" bbox, żeby wielokrotne zapytania nie robiły nowych searchy.
    """

    PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "cop-dem-glo-30"

    def __init__(self, *, point_buffer_deg: float = 0.15, max_items: int = 20):
        self.point_buffer_deg = float(point_buffer_deg)
        self.max_items = int(max_items)
        self._catalog = Client.open(self.PC_STAC)

        self._ds_cache: Dict[str, List[rasterio.io.DatasetReader]] = {}
        self._debug = DemDebug(
            source="planetary:cop-dem-glo-30",
            crs=None,
            nodata=None,
            used_assets=[],
            last_bbox_wgs84=None,
            notes={"cache_keys": []},
        )

    def _bbox_key(self, bbox: Tuple[float, float, float, float]) -> str:
        # Zaokrąglamy bbox, żeby cache zadziałał “po okolicy”
        return ",".join(f"{x:.3f}" for x in bbox)

    def _query_bbox_for_point(self, lat: float, lon: float) -> Tuple[float, float, float, float]:
        b = self.point_buffer_deg
        return (lon - b, lat - b, lon + b, lat + b)

    def _ensure_datasets_for_bbox(self, bbox_wgs84: Tuple[float, float, float, float]) -> List[rasterio.io.DatasetReader]:
        key = self._bbox_key(bbox_wgs84)
        if key in self._ds_cache:
            return self._ds_cache[key]

        # STAC search
        search = self._catalog.search(
            collections=[self.COLLECTION],
            intersects=mapping(box(*bbox_wgs84)),
            limit=self.max_items,
        )
        items = list(search.items())

        if not items:
            self._ds_cache[key] = []
            self._debug = DemDebug(
                source="planetary:cop-dem-glo-30",
                crs=None,
                nodata=None,
                used_assets=[],
                last_bbox_wgs84=bbox_wgs84,
                notes={"error": "Brak DEM dla bbox (STAC returned 0 items)."},
            )
            return []

        datasets: List[rasterio.io.DatasetReader] = []
        used_assets: List[str] = []

        for it in items:
            signed = pc.sign(it)
            # zazwyczaj jest asset 'data', ale iterujemy po wszystkich
            for _, a in signed.assets.items():
                href = a.href
                if href.lower().endswith(".tif") or (a.media_type and "geotiff" in a.media_type.lower()):
                    try:
                        ds = rasterio.open(href)
                        datasets.append(ds)
                        used_assets.append(href)
                        # z reguły 1 asset starczy; ale zostawiamy możliwość wielu
                        break
                    except Exception:
                        continue

        crs = str(datasets[0].crs) if datasets and datasets[0].crs else None
        nodata = datasets[0].nodata if datasets else None

        self._ds_cache[key] = datasets
        self._debug = DemDebug(
            source="planetary:cop-dem-glo-30",
            crs=crs,
            nodata=nodata,
            used_assets=used_assets[:5],  # dla debug ograniczamy widok
            last_bbox_wgs84=bbox_wgs84,
            notes={"n_items": len(items), "n_opened": len(datasets), "cache_keys": list(self._ds_cache.keys())[-5:]},
        )
        return datasets

    def height_at(self, lat: float, lon: float, *, strict: bool = False) -> Optional[float]:
        bbox = self._query_bbox_for_point(lat, lon)
        datasets = self._ensure_datasets_for_bbox(bbox)

        if not datasets:
            if strict:
                raise ValueError("Brak danych DEM dla tego punktu (planetary).")
            return None

        for ds in datasets:
            try:
                # sample przyjmuje współrzędne w CRS ds; jeśli ds jest EPSG:4326 -> OK.
                # Jeśli ds ma inny CRS, rasterio.sample nadal oczekuje w CRS datasetu,
                # więc trzeba transformować (robimy to).
                if ds.crs and str(ds.crs).upper() not in ("EPSG:4326", "WGS84"):
                    xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
                    x, y = float(xs[0]), float(ys[0])
                    val = next(ds.sample([(x, y)]))[0]
                else:
                    val = next(ds.sample([(lon, lat)]))[0]

                if ds.nodata is not None and np.isclose(val, ds.nodata):
                    continue

                return float(val)
            except Exception:
                continue

        if strict:
            raise ValueError("Nie udało się odczytać wysokości (planetary).")
        return None

    def debug_info(self) -> DemDebug:
        return self._debug

    def close(self) -> None:
        # zamykamy wszystkie otwarte dataset'y
        for key, dss in self._ds_cache.items():
            for ds in dss:
                try:
                    ds.close()
                except Exception:
                    pass
        self._ds_cache.clear()
