
# -*- coding: utf-8 -*-
import re
import datetime as dt
from pathlib import Path

import numpy as np
import streamlit as st

# Optional GeoTIFF support
try:
    import rasterio
except Exception:
    rasterio = None

from PIL import Image

# Optional for single-band coloring
try:
    import matplotlib.cm as cm
except Exception:
    cm = None

st.set_page_config(page_title="Remote Sensing for Watershed", layout="wide")
st.title("🌊 Remote Sensing for Watershed")

st.markdown(
    """
Browse date-based images for two study areas: **Snow Melt Progression** and **Island Lake Images**.  
Dates must appear in filenames: `YYYY-MM-DD`, `YYYY_MM_DD`, or `YYYYMMDD`.
"""
)

DATE_PATTERNS = [
    r"(?P<y>20\d{2}|19\d{2})[-_](?P<m>0[1-9]|1[0-2])[-_](?P<d>0[1-9]|[12]\\d|3[01])",
    r"(?P<y>20\d{2}|19\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\\d|3[01])",
]
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

def extract_date_from_name(name: str):
    for pat in DATE_PATTERNS:
        m = re.search(pat, name)
        if m:
            y, mth, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
            try:
                return dt.date(y, mth, d)
            except ValueError:
                return None
    return None

@st.cache_data(show_spinner=False)
def index_images(folder: str):
    p = Path(folder)
    if not p.exists():
        return {}
    idx = {}
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            d = extract_date_from_name(f.name)
            if d is not None:
                idx.setdefault(d, []).append(f)
    return idx

def _linear_rescale(arr, pmin=2, pmax=98, clip=True):
    arr = arr.astype("float32")
    lo = np.nanpercentile(arr, pmin)
    hi = np.nanpercentile(arr, pmax)
    if hi <= lo:
        hi = lo + 1e-9
    arr = (arr - lo) / (hi - lo)
    if clip: arr = np.clip(arr, 0, 1)
    return (arr * 255).astype("uint8")

def _single_band_to_rgb(band_u8, colormap_name="viridis"):
    if cm is None:
        import numpy as np
        return np.stack([band_u8, band_u8, band_u8], axis=-1)
    cmap = cm.get_cmap(colormap_name)
    colored = cmap(band_u8.astype(np.float32) / 255.0)
    return (colored[..., :3] * 255).astype("uint8")

def _read_tiff_as_rgb(path: Path, rgb_bands=(1,2,3), out_max_dim=1200, pmin=2, pmax=98, single_band_colormap="viridis"):
    if rasterio is None:
        st.error("GeoTIFF support requires `rasterio`. Sample uses PNG so you're good; add rasterio for TIFFs.")
        return None, None
    import rasterio, numpy as np
    from PIL import Image
    with rasterio.open(path) as ds:
        meta = {
            "driver": ds.driver,
            "crs": str(ds.crs) if ds.crs else "None",
            "transform": tuple(ds.transform) if ds.transform else None,
            "width": ds.width, "height": ds.height, "count": ds.count,
            "dtype": str(ds.dtypes[0]), "bounds": tuple(ds.bounds) if ds.bounds else None,
            "res": tuple(ds.res) if ds.res else None,
        }
        h, w = ds.height, ds.width
        scale = out_max_dim / max(h, w) if max(h, w) > out_max_dim else 1.0
        out_h, out_w = max(1, int(h*scale)), max(1, int(w*scale))
        if ds.count >= 3:
            bands = []
            for b in rgb_bands:
                b = min(max(1, b), ds.count)
                band = ds.read(b, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
                if ds.nodata is not None: band[band == ds.nodata] = np.nan
                bands.append(_linear_rescale(band, pmin, pmax))
            rgb = np.stack(bands, axis=-1)
        else:
            band = ds.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
            if ds.nodata is not None: band[band == ds.nodata] = np.nan
            rgb = _single_band_to_rgb(_linear_rescale(band, pmin, pmax), single_band_colormap)
    return Image.fromarray(rgb), meta

def load_image(path: Path, out_max_dim=1200, pmin=2, pmax=98, colormap="viridis", rgb_bands=(1,2,3)):
    ext = path.suffix.lower()
    if ext in {".tif",".tiff"}:
        return _read_tiff_as_rgb(path, rgb_bands=rgb_bands, out_max_dim=out_max_dim, pmin=pmin, pmax=pmax, single_band_colormap=colormap)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > out_max_dim:
        s = out_max_dim / max(w, h); img = img.resize((int(w*s), int(h*s)))
    return img, None

def section_view(title: str, default_folder: str, key_prefix: str):
    st.markdown("---")
    st.header(title)
    cols = st.columns([2,1,1,1])
    folder = cols[0].text_input(f"{title} — folder", value=default_folder, key=f"{key_prefix}_folder")
    out_max_dim = cols[1].slider("Max px", 600, 3000, 1200, 100, key=f"{key_prefix}_max")
    pmin = cols[2].slider("Pmin", 0, 20, 2, key=f"{key_prefix}_pmin")
    pmax = cols[3].slider("Pmax", 80, 100, 98, key=f"{key_prefix}_pmax")
    cmap = st.selectbox("Colormap (1-band)", ["viridis","plasma","magma","inferno","gray"], key=f"{key_prefix}_cmap")

    idx = index_images(folder)
    if not idx:
        st.warning(f"No dated images found in `{folder}`.")
        return
    dates = sorted(idx.keys())
    min_d, max_d = dates[0], dates[-1]

    mode = st.radio("Mode", ["Single date", "Compare two dates"], horizontal=True, key=f"{key_prefix}_mode")

    def choose_file(d):
        files = idx[d]
        if len(files)==1: return files[0]
        name = st.selectbox(f"Multiple on {d}", [f.name for f in files], key=f"{key_prefix}_{d}_pick")
        return next(f for f in files if f.name == name)

    def show(d):
        f = choose_file(d)
        rgb_bands = (1,2,3)
        if f.suffix.lower() in {".tif",".tiff"} and rasterio is not None:
            import rasterio
            with rasterio.open(f) as ds:
                st.caption(f"**{f.name}** — {ds.width}×{ds.height}px • bands:{ds.count}")
                if ds.count >= 3:
                    c1,c2,c3 = st.columns(3)
                    r = c1.number_input("R",1,ds.count,1,key=f"{key_prefix}_r")
                    g = c2.number_input("G",1,ds.count,min(2,ds.count),key=f"{key_prefix}_g")
                    b = c3.number_input("B",1,ds.count,min(3,ds.count),key=f"{key_prefix}_b")
                    rgb_bands = (int(r),int(g),int(b))
        img, meta = load_image(f, out_max_dim, pmin, pmax, cmap, rgb_bands)
        if img is None: st.error(f"Could not load {f.name}"); return
        st.image(img, use_column_width=True)
        if meta:
            with st.expander("GeoTIFF metadata"): st.json(meta)

    if mode == "Single date":
        d = st.date_input("Pick date", value=max_d, min_value=min_d, max_value=max_d, key=f"{key_prefix}_date")
        target = d if d in idx else min(dates, key=lambda x: abs((x-d).days))
        if d not in idx: st.info(f"No image on {d}. Showing nearest: **{target}**")
        show(target)
    else:
        c1,c2 = st.columns(2)
        d1 = c1.date_input("Left", value=max(min_d, min_d+dt.timedelta(days=1)), min_value=min_d, max_value=max_d, key=f"{key_prefix}_L")
        d2 = c2.date_input("Right", value=max_d, min_value=min_d, max_value=max_d, key=f"{key_prefix}_R")
        nearest = lambda d: d if d in idx else min(dates, key=lambda x: abs((x-d).days))
        d1p, d2p = nearest(d1), nearest(d2)
        if d1p!=d1: st.info(f"Left: no {d1}; using **{d1p}**")
        if d2p!=d2: st.info(f"Right: no {d2}; using **{d2p}**")
        a,b = st.columns(2)
        with a: st.subheader(f"{d1p}"); show(d1p)
        with b: st.subheader(f"{d2p}"); show(d2p)

section_view("❄️ Snow Melt Progression", "Data/Snow melt progression", "snow")
section_view("🏞️ Island Lake Images",   "Data/Island Lake", "island")

st.sidebar.markdown("---")
st.sidebar.write("**Tips**")
st.sidebar.caption("- PNG samples are bundled so the app runs without rasterio. Add rasterio to requirements if you use GeoTIFFs.")
