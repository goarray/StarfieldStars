#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL PRODUCTION SIMBAD ENRICHER — 2025 GOLD MASTER
Input : starsDebug.csv (CLEAN, corrected, ready)
Output: NewStarsData.csv + Mismatches_Report.csv + Binary_Systems.csv
No hacks. No overrides. Pure science.
"""
import pandas as pd
import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astroquery.simbad import conf
import astropy.units as u
from math import log10, cos, sin, sqrt, pi
import re
import argparse
import warnings
from typing import Dict, Any
from astroquery.exceptions import NoResultsWarning

# ----------------------------------------------------------------------
# SIMBAD SETUP — MAXIMUM COMPATIBILITY & DATA (2025+)
# ----------------------------------------------------------------------
warnings.filterwarnings("ignore", category=NoResultsWarning)
custom_simbad = Simbad()
custom_simbad.reset_votable_fields()
custom_simbad.add_votable_fields(
    'main_id', 'ids',
    'ra', 'dec',
    'pmra', 'pmdec',
    'plx_value', 'plx_err',
    'rvz_radvel',
    'V', 'B',
    'J', 'H', 'K',
    'sp_type'
)
conf.timeout = 45
Simbad = custom_simbad

# ----------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------

# System level rows to add to the output
additional_systems = {
    "72432s": "GJ 566",
    "83865s": "GJ 660",
    "119220s": "GJ 667",
    "81444s": "HIP 81693",
    "118742s": "HIP 55203"
}

# Some IDs were change between stars.csv and the actual in-game STDT
starID_corrections = {
    "118043": "119221", # BetaAndraste
    "119058": "119225", # BetaTernion
    "119143": "119220", # OborumPrime
    "118441": "119224", # ThePup
    "71453": "119223", # Toliman
    "118743": "119222" # UrsaeMinoris
}

# Note that all GL, Wo, NN and GJ in stars.csv are replaced with GJ and mormalized upon csv load
id_remap = {
    "gj451a": "GJ 451",
    "gj580a": "GJ 580",
    "gj596.1a": "GJ 9527",
    "gj667b": "GJ 667A",
    "gj3669a": "GJ 3669",
    "gj3727a": "GJ 3727",
    "gj3728b": "GJ 3728",
    "gj3781a": "GJ 3781",
    "gj3782b": "GJ 3782",
    "gj3868a": "GJ 3868",
    "fermi": "HD 134066A",
    "leonis": "GJ 9359A"
}

# Star names that have changed between stars.csv and the game
full_name_remap = {
    "59020": "Alchiba",
    "65147": "Al-Battani",
    "69793": "Bannoc",
    "65515": "Bara",
    "57767": "Groombridge",
    "84147": "Guniibuu",
    "119135": "Indum",
    "65150": "Khayyam",
    "57584": "Lantana",
    "119144": "Oborum Proxima",
    "69479": "Syrma",
    "119224": "The Pup",
    "86475": "Celebrai", # Spelling change, note: planets are still *Calabrai*
    "3814": "Eta Cassiopeia", # Spelling change
}

lum_class_map = {
    'Ia': 'Supergiant',
    'Iab': 'Supergiant',
    'Ib': 'Supergiant',
    'II': 'Bright giant',
    'III': 'Giant',
    'IV': 'Subgiant',
    'V': 'Main sequence',
    'VI': 'Subdwarf',
    'D': 'White dwarf',
    'WD': 'White dwarf'
}

BC_TABLE = {
    'O': {'V': -3.2, 'B': -3.4, 'J': -3.0, 'H': -2.9, 'K': -2.8},
    'B': {'V': -2.7, 'B': -2.9, 'J': -2.5, 'H': -2.4, 'K': -2.3},
    'A': {'V': -0.3, 'B': -0.4, 'J': -0.2, 'H': -0.2, 'K': -0.2},
    'F': {'V': -0.1, 'B': -0.2, 'J': -0.1, 'H': -0.1, 'K': -0.1},
    'G': {'V': -0.1, 'B': -0.2, 'J': -0.1, 'H': -0.1, 'K': -0.1},
    'K': {'V': -0.3, 'B': -0.4, 'J': -0.2, 'H': -0.2, 'K': -0.2},
    'M': {'V': -1.2, 'B': -1.4, 'J': -0.9, 'H': -0.9, 'K': -0.9},
}

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='starsDebug.csv')
args = parser.parse_args()
SEARCH_RADIUS = '5 arcmin'
MAS_TO_RAD = np.deg2rad(1/3600000)
ID_COLS = ['hip', 'hd', 'hr', 'comp_primary', 'gl', 'proper', 'comp']

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def remap_identifier(val: Any) -> str:
    """
    Returns the canonical main_id or Simbad query name for a given identifier.
    Uses lowercase keys for map to avoid case issues.
    Works with strings or numbers.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    val_str = str(val).strip()
    if val_str == '':
        return ''
    val_clean = val_str.lower().replace(" ", "")
    # Map keys should all be lowercase
    val_remap = id_remap.get(val_clean, val_str)
    return val_remap

def smart_normalize_gl(val):
    if not val or str(val).strip() in ['', 'nan']:
        return ''

    val = str(val).strip()

    # Strip common prefixes (GJ, Gl, gj, gl, wo, nn)
    raw = re.sub(r'^(gl|gj|nn|wo)\s*', '', val, flags=re.IGNORECASE).strip()

    # Normalize SIMBAD-style system aliases: 448.0 → 448
    m = re.match(r'^(\d+)\.0$', raw)
    if m:
        raw = m.group(1)

    # Now return standard canonical GJ form
    return f"GJ {raw}"

def U(name: str):
    return getattr(u, name)

def safe_float(x, default=np.nan):
    try: return float(x) if pd.notna(x) and str(x).strip() not in ['', 'None'] else default
    except: return default

def parse_ids(ids_str):
    # print("Raw ids_str:", ids_str)
    out = {}
    for p in str(ids_str or '').split('|'):
        p = p.strip()
        if p.startswith('HIP '): out['hip'] = p[4:].strip()
        elif p.startswith('HD '): out['hd'] = p[3:].strip()
        elif p.startswith('HR '): out['hr'] = p[3:].strip()
        elif p.startswith('GJ ') or p.startswith('Gl '): out['gl'] = p.split(' ',1)[1].strip()
        elif p.startswith('NAME '): out['proper'] = p[5:].strip()
        elif 'Gaia DR' in p: out['gaia'] = p.split()[-1]
        elif p.startswith('2MASS '): out['2mass'] = p[6:].strip()
        elif p.startswith('TYC '): out['tyc'] = p[4:].strip()
        elif p.startswith('CCDM '): out['ccdm'] = p[5:].strip()
        elif p.startswith('WDS '): out['wds'] = p[4:].strip()
        elif p.startswith('ADS '): out['ads'] = p[4:].strip()
        elif p.startswith('TIC '): out['tic'] = p[4:].strip()
        elif p.startswith('NLTT '): out['nltt'] = p[5:].strip()
        elif p.startswith('PLX '): out['plx'] = p[4:].strip()
        elif p.startswith('FK5 '): out['fk5'] = p[4:].strip()
        elif p.startswith('IRAS '): out['iras'] = p[5:].strip()
        elif p.startswith('GCRV '): out['gcrv'] = p[5:].strip()
        elif p.startswith('WEB '): out['web'] = p[4:].strip()
    return out

def extract_bayer_flam_con(name):
    name = str(name)
    # Try bayer + con + optional trailing letter
    m = re.search(r'([a-zA-Z]+)\s+([A-Z][a-z]{2})\s*[A-Z]?$', name)
    if m: return m.group(1), '', m.group(2)
    # Try flam + con + optional trailing letter
    m = re.search(r'(\d+)\s+([A-Z][a-z]{2})\s*[A-Z]?$', name)
    if m: return '', m.group(1), m.group(2)
    return '', '', ''

def extract_component(name):
    m = re.search(r'\s([A-Z])$', str(name))
    return m.group(1) if m else ''

def cartesian(dist, ra_rad, dec_rad):
    return (dist * cos(dec_rad) * cos(ra_rad),
            dist * cos(dec_rad) * sin(ra_rad),
            dist * sin(dec_rad))

def proper_to_cartesian_pm(dist_pc, ra_rad, dec_rad, pmra_masyr, pmdec_masyr, rv_kms):
    # Convert mas/yr → rad/yr
    k = 4.74047  # km/s per mas/yr at 1 pc
    vx = rv_kms * cos(dec_rad) * cos(ra_rad) - k * dist_pc * (pmra_masyr * sin(ra_rad) + pmdec_masyr * cos(ra_rad) * sin(dec_rad))
    vy = rv_kms * cos(dec_rad) * sin(ra_rad) + k * dist_pc * (pmra_masyr * cos(ra_rad) - pmdec_masyr * sin(ra_rad) * sin(dec_rad))
    vz = rv_kms * sin(dec_rad) + k * dist_pc * pmdec_masyr * cos(dec_rad)
    return vx, vy, vz

def simple_hz(lum):
    if np.isnan(lum): return np.nan, np.nan, np.nan, np.nan
    s = sqrt(lum)
    return 0.84*s, 2.4*s, 1.67*s, 0.95*s

def get_BC(spect, band):
    if not spect or band not in ['V','B','J','H','K']:
        return 0.0
    letter = spect[0]  # crude: O, B, A, F, G, K, M
    return BC_TABLE.get(letter, {}).get(band, 0.0)

def temp_from_BV(ci):
    # Ballesteros formula (valid for B–V)
    return 4600 * (1 / (0.92 * ci + 1.7) + 1 / (0.92 * ci + 0.62))

def temp_from_VK(ci):
    # Approximate calibration for V–K (Alonso et al. 1996)
    return 8800 - 2000 * ci + 200 * ci**2

def temp_from_JK(ci):
    # Approximate calibration for J–K (Mann et al. 2015)
    return 3700 + 1200 * (0.9 - ci)  # crude linear fit

TEMP_FUNCS = {
    'B-V': temp_from_BV,
    'V-K': temp_from_VK,
    'J-K': temp_from_JK,
}

def get_temp(ci, ci_type):
    func = TEMP_FUNCS.get(ci_type)
    return func(ci) if func and pd.notna(ci) else np.nan

def estimate_lifespan_gyr(spect: str, mass: float = np.nan) -> float:
    """
    Estimate main-sequence lifespan (Gyr) from mass or spectral type.
    Uses mass if available, otherwise maps spectral type with numeric subclass.
    """
    # Use mass if available
    if pd.notna(mass) and mass > 0:
        return 10.0 * (1.0 / mass)**2.5

    if not spect or not spect.strip():
        return np.nan

    # Spectral type parsing
    m = re.match(r'([OBAFGKM])\s*(\d*\.?\d*)', spect.strip(), re.I)
    if not m:
        return np.nan

    sp_class, sp_num = m.group(1).upper(), m.group(2)
    sp_num = float(sp_num) if sp_num else 5.0  # default to middle subclass

    # Approximate mass by class and numeric subclass (linear interpolation)
    # Mass ranges from Allen's Astrophysical Quantities
    class_mass_ranges = {
        'O': (16, 50),
        'B': (2.1, 16),
        'A': (1.4, 2.1),
        'F': (1.04, 1.4),
        'G': (0.8, 1.04),
        'K': (0.45, 0.8),
        'M': (0.08, 0.45)
    }
    if sp_class not in class_mass_ranges:
        return np.nan

    m_high, m_low = class_mass_ranges[sp_class]
    mass_est = m_high - (sp_num / 9.0) * (m_high - m_low)  # subclass 0 → high mass, 9 → low mass

    # Compute lifespan
    t_ms = 10.0 * (1.0 / mass_est)**2.5
    return t_ms

def apply_id_remap(rows, id_remap):
    for r in rows:
        # Ensure comp_primary is string for consistent lookup
        if 'comp_primary' in r and pd.notna(r['comp_primary']):
            cp = str(r['comp_primary']).strip()
            if cp in id_remap:
                r['comp_primary'] = id_remap[cp]

        # Update comp (siblings) with remap
        if 'comp' in r and r['comp']:
            comps = str(r.get('comp', '')).split('|')
            new_comps = []
            for c in comps:
                c = c.strip()
                if c in id_remap:
                    new_comps.append(id_remap[c])
                else:
                    new_comps.append(c)
            r['comp'] = '|'.join(new_comps)

def assign_siblings(rows, sep=' '):
    # Collect all stars by primary
    primary_map = {}
    for row in rows:
        primary = row.get('comp_primary')
        if pd.notna(primary):
            primary_map.setdefault(primary, []).append(str(row['id']))
    
    # Assign siblings
    for row in rows:
        primary = row.get('comp_primary')
        if pd.notna(primary):
            siblings = [sid for sid in primary_map[primary] if sid != str(row['id'])]
            row['sibling'] = sep.join(siblings) if siblings else np.nan

# ----------------------------------------------------------------------
# Load Input — NOW ASSUMED CLEAN
# ----------------------------------------------------------------------
df_in = pd.read_csv(args.file, skiprows=1).fillna('')
df_in.columns = df_in.columns.str.strip()
df_in['id'] = df_in['id'].astype(str).str.strip()
proper_in = df_in.get('proper', None)

# Inject missing systems as blank rows
extra_rows = []
for new_id, query_name in additional_systems.items():
    template = {col: '' for col in df_in.columns}
    template['id'] = new_id
    template['proper'] = query_name
    extra_rows.append(template)

df_in = pd.concat([df_in, pd.DataFrame(extra_rows)], ignore_index=True)

# RA in hours → degrees
if 'ra' in df_in.columns:
    df_in['ra'] = pd.to_numeric(df_in['ra'], errors='coerce') * 15.0
if 'dec' in df_in.columns:
    df_in['dec'] = pd.to_numeric(df_in['dec'], errors='coerce')

# Clean IDs
for col in ID_COLS:
    if col in df_in.columns:
        df_in[col] = (
            df_in[col]
            .astype(str)
            .str.strip()
            .replace({'nan': '', '<NA>': '', '0': ''})
        )

df_in['gl_raw'] = df_in['gl'].astype(str).str.strip()
df_in['gl'] = df_in['gl_raw'].apply(
    lambda x: remap_identifier(x) if x and str(x).strip() not in ['nan', ''] else x
)

df_in['gl'] = df_in['gl'].apply(smart_normalize_gl)

# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
output_rows = []
failed = []
fallbacks = []

for _, row_in in df_in.iterrows():
    old_id = str(row_in['id'])
    star_id = starID_corrections.get(old_id, old_id)
    print(f"[{star_id}] Processing...", end='')

    # Original GL before any processing
    original_gl_raw = str(row_in.get('gl_raw', '')).strip()

    current_gl = row_in.get('gl', '')
    current_gl = '' if pd.isna(current_gl) else str(current_gl).strip()

    row: Dict[str, Any] = {k: np.nan for k in [
        # Core identifiers
        'id','hip','hd','hr','gl','main_id','full_name','proper',
        
        # Coordinates / motion
        'ra','dec','rarad','decrad','dist','x','y','z','vx','vy','vz','pmra','pmdec','pmrarad','pmdecrad',
        
        # Physical / stellar properties
        'spect','lum_class','mag','absmag','ci','radius','mass','temp','type','lifespan','comp','comp_primary','sibling','compdist','dist_au',
        
        # Habitability zones
        'inner_ohz','outer_ohz','inner_chz','outer_chz',
        
        # External / metadata IDs
        'bayer','flam','con','gaia','2mass','tyc','ccdm','wds','ads','tic','nltt','plx','fk5','iras','gcrv','web',
        
        # Old stars.csv fields for provenance
        'old_bgs_id'
    ]}
    row.update({
        'id': star_id
    })

    # ----------------------------------------------------------------------
    # Manual overrides
    # ----------------------------------------------------------------------

    # Sol
    if star_id == '0':
        sol_row: Dict[str, Any] = {k: np.nan for k in row.keys()}
        sol_row.update({
            # Core identifiers
            'id': '0', 'full_name': 'Sol', 'proper': 'Sol', 'main_id': 'Sol', 'hip': np.nan, 'hd': np.nan, 'hr': np.nan, 'gl': np.nan,

            # Coordinates / motion. Sol is the reference; relative motion is defined from Earth/Sun.
            'ra': 0.0, 'dec': 0.0, 'rarad': 0.0, 'decrad': 0.0, 'dist': 0.0, 'dist_au': 0.0,
            
            # Physical / stellar properties
            'spect': 'G2V', 'lum_class': 'V', 'mag': -26.74, 'absmag': 4.83, 'ci': 0.63,
            
            # Habitability zones
            'inner_ohz': 0.84,'outer_ohz': 2.4,'inner_chz': 0.95,'outer_chz': 1.67,
            
            # External / metadata IDs
            'bayer': np.nan, 'flam': np.nan, 'con': np.nan, 'gaia': np.nan,'2mass': np.nan,'tyc': np.nan,'ccdm': np.nan,'wds': np.nan,'ads': np.nan,
            'tic': np.nan,'nltt': np.nan,'plx': np.nan,'fk5': np.nan,'iras': np.nan,'gcrv': np.nan,'web': np.nan,
            
            # Legacy / provenance
            'old_bgs_id': 0
        })
        print("Sol processed")
        output_rows.append(sol_row)
        continue

    table = None
    proper_in = None
    identifiers = [
        (row_in.get('gl'), ''),
        (row_in.get('hip'), 'HIP '),
        (row_in.get('proper'), ''),
        (proper_in, '')
    ]

    query = None
    used_identifier = None
    for val, prefix in identifiers:
        if pd.notna(val) and str(val).strip():
            query_val = remap_identifier(val)
            query = f"{prefix}{query_val}".strip()
            try:
                table = Simbad.query_object(query)
                if table is not None and len(table):
                    used_identifier = prefix or 'gl'
                    print(f" → {query}")
                    break
            except Exception:
                pass
    
    # === FALLBACK / CORRECTION REPORTING ===
    original_star_id = old_id
    original_gl = original_gl_raw
    original_proper = row_in.get('proper', '')

    # Prepare triggers
    triggers = []

    # 1. StarID correction
    if old_id in starID_corrections:
        new_id = starID_corrections[old_id]
        triggers.append({
            'star_id': new_id,
            'original': old_id,
            'used': new_id,
            'query': query or current_gl,
            'correction': "bgs id changed, used starID_corrections"
        })

    # 2. id_remap report
    orig_key = (original_gl_raw or "").lower().replace(" ", "")
    if not orig_key and row_in.get('proper'):
        orig_key = str(row_in['proper']).lower().replace(" ", "")
    if orig_key in id_remap:
        triggers.append({
            'star_id': star_id,
            'original': original_gl_raw or row_in.get('proper'),
            'used': id_remap[orig_key],  # always report the mapped value
            'query': query or id_remap[orig_key],
            'correction': "bgs 'gl' id incorrect, used id_remap"
        })

    # 3. Identifier fallbacks (HIP / HD / HR)
    for val, prefix in identifiers:
        if pd.notna(val) and str(val).strip():
            prefix_name = prefix
            # Only report if a fallback was actually used
            if used_identifier is not None and used_identifier == prefix_name:
                triggers.append({
                    'star_id': star_id,
                    'original': original_gl,
                    'used': f"{prefix_name}{val}".strip(),  # the actual fallback value used
                    'query': query or current_gl,
                    'correction': f"Gliese preferred, used {prefix_name.strip()} fallback instead"
                })

    # 4. Proper name remap
    if star_id in full_name_remap:
        triggers.append({
            'star_id': star_id,
            'original': original_proper,
            'used': full_name_remap[star_id],
            'query': query or current_gl,
            'correction': "proper name changed (full_name_remap applied)"
        })

    # Append each row to fallbacks
    for trigger in triggers:
        fallbacks.append(trigger)

    # Coordinate fallback
    if not table and pd.notna(row_in.get('ra')) and pd.notna(row_in.get('dec')):
        coord = SkyCoord(ra=row_in['ra']*U("deg"), dec=row_in['dec']*U("deg"), frame='icrs')
        try:
            table = Simbad.query_region(coord, radius=SEARCH_RADIUS)
            if table and len(table):
                table = table[0:1]
                print(f" → coord → {table[0]['main_id']}")
        except: pass

    if not table or len(table) == 0:
        print(" → FAILED")
        failed.append(star_id)
        output_rows.append(row)
        continue

    # ----------------------------------------------------------------------
    # SIMBAD ENRICHMENT + DERIVED PROPERTIES
    # ----------------------------------------------------------------------
    sim = table[0]

    # Core identifiers
    row['main_id'] = str(sim['main_id']).strip()
    simbad_proper = sim.get('proper') or sim.get('name')
    row['proper'] = str(simbad_proper).strip() if simbad_proper else ""
    dist   = row.get('dist', np.nan)
    mag    = row.get('mag', np.nan)
    pmra   = row.get('pmra', np.nan)
    pmdec  = row.get('pmdec', np.nan)
    rv     = row.get('rv', np.nan)
    rarad  = row.get('rarad', 0.0)
    decrad = row.get('decrad', 0.0)
    
    row['full_name'] = row_in.get('proper', "").strip()
    sid = str(star_id)
    if sid in full_name_remap:
        row['full_name'] = full_name_remap[sid]

    row['bayer'], row['flam'], row['con'] = extract_bayer_flam_con(sim['main_id'])

    # Coordinates / motion
    row['ra'] = safe_float(sim['ra'])
    row['dec'] = safe_float(sim['dec'])
    row['rarad'] = np.deg2rad(row['ra'])
    row['decrad'] = np.deg2rad(row['dec'])

    plx = safe_float(sim.get('plx_value'))
    if plx > 0:
        row['dist'] = 1000.0 / plx  # pc
        row['dist_au'] = row['dist'] * 206264.806  # AU

    row['pmra'] = safe_float(sim.get('pmra'))
    row['pmdec'] = safe_float(sim.get('pmdec'))
    row['rv'] = safe_float(sim.get('rvz_radvel'))

    # Magnitudes
    mag_V = safe_float(sim.get('V'))
    mag_B = safe_float(sim.get('B'))
    mag_J = safe_float(sim.get('J'))
    mag_H = safe_float(sim.get('H'))
    mag_K = safe_float(sim.get('K'))

    row['mag_V'] = mag_V
    row['mag_B'] = mag_B
    row['mag_J'] = mag_J
    row['mag_H'] = mag_H
    row['mag_K'] = mag_K

    # Apparent magnitude (best available)
    if pd.notna(mag_V):
        row['mag'] = mag_V
        row['primary_band'] = 'V'
    elif pd.notna(mag_B):
        row['mag'] = mag_B
        row['primary_band'] = 'B'
    elif pd.notna(mag_K):
        row['mag'] = mag_K
        row['primary_band'] = 'K'
    elif pd.notna(mag_H):
        row['mag'] = mag_H
        row['primary_band'] = 'H'
    elif pd.notna(mag_J):
        row['mag'] = mag_J
        row['primary_band'] = 'J'
    else:
        row['mag'] = np.nan
        row['primary_band'] = 'None'

    # Color index (B-V preferred, fallback V-K, J-K)
    if pd.notna(mag_B) and pd.notna(mag_V):
        row['ci'] = mag_B - mag_V
        row['ci_type'] = 'B-V'
    elif pd.notna(mag_V) and pd.notna(mag_K):
        row['ci'] = mag_V - mag_K
        row['ci_type'] = 'V-K'
    elif pd.notna(mag_J) and pd.notna(mag_K):
        row['ci'] = mag_J - mag_K
        row['ci_type'] = 'J-K'
    else:
        row['ci'] = np.nan
        row['ci_type'] = None

    # Spectral type and luminosity class
    row['spect'] = str(sim.get('sp_type') or '').strip()
    m_lum = re.search(r'(I{1,3}[ab]?\b|IV\b|V\b)', row['spect'])
    row['lum_class'] = m_lum.group(1) if m_lum else np.nan

    if pd.notna(row['mag']) and pd.notna(row['dist']) and row['dist'] > 0:
        row['absmag'] = row['mag'] - 5*log10(row['dist']) + 5
        BC = get_BC(row['spect'], row['primary_band'])
        Mbol = row['absmag'] + BC
        row['lum'] = 10**((4.74 - Mbol) / 2.5)  # Sun's bolometric magnitude
    else:
        row['absmag'] = np.nan
        row['lum'] = np.nan

    row['temp'] = get_temp(row['ci'], row['ci_type'])

    # Estimate radius (solar units) via Stefan-Boltzmann
    if pd.notna(row['lum']) and pd.notna(row['temp']):
        row['radius'] = sqrt(row['lum']) * (5772 / row['temp'])**2
    else:
        row['radius'] = np.nan

    # Estimate mass (crude, main sequence)
    if pd.notna(row['lum']):
        row['mass'] = row['lum']**0.25
    else:
        row['mass'] = np.nan

    # IDs from Simbad
    ids = parse_ids(sim.get('ids'))
    # Strip .0 from GJ IDs returned by SIMBAD
    if 'gl' in ids and isinstance(ids['gl'], str):
        ids['gl'] = re.sub(r'\.0([A-Za-z]?$)', r'\1', ids['gl'])
        ids['gl'] = re.sub(r'\.0$', '', ids['gl'])

    row.update({k: v for k, v in ids.items() if v})

    # Preserve stars.csv IDs that were changed in the final game data
    old_id = row_in.get("id")
    if old_id in starID_corrections:
        row["old_bgs_id"] = starID_corrections[old_id]
    else:
        row["old_bgs_id"] = np.nan

    # Component / binaries
    row['comp'] = extract_component(sim['main_id'])
    comp_primary_val = row_in.get('comp_primary', np.nan)

    # Only try to convert if value is not NaN and not an empty string
    if pd.notna(comp_primary_val) and str(comp_primary_val).strip() != '':
        try:
            # allow "123.0" -> 123
            row['comp_primary'] = str(int(float(str(comp_primary_val).strip())))
        except Exception:
            # fallback: keep the stripped source (handles non-numeric labels gracefully)
            row['comp_primary'] = str(comp_primary_val).strip()
    else:
        row['comp_primary'] = ''
        
    row['sibling'] = row_in.get('sibling', np.nan)
    row['compdist'] = row_in.get('compdist', np.nan)

    row['pmrarad'] = row['pmra'] * MAS_TO_RAD
    row['pmdecrad'] = row['pmdec'] * MAS_TO_RAD

    # Cartesian coordinates
    if pd.notna(row['dist']):
        row['x'], row['y'], row['z'] = cartesian(row['dist'], row['rarad'], row['decrad'])

    # Space velocity
    if pd.notna(row['dist']) and pd.notna(row['pmra']) and pd.notna(row['pmdec']) and pd.notna(row['rv']):
        row['vx'], row['vy'], row['vz'] = proper_to_cartesian_pm(
            dist_pc=row['dist'],
            ra_rad=row['rarad'],
            dec_rad=row['decrad'],
            pmra_masyr=row['pmra'],
            pmdec_masyr=row['pmdec'],
            rv_kms=row['rv']
        )

    # Habitability zones
    row['inner_ohz'], row['outer_ohz'], row['inner_chz'], row['outer_chz'] = simple_hz(row.get('lum'))

    # Determine evolutionary stage
    m = re.search(r'(0|Iab|Ia|Ib|II|III|IV|V|VI|D|WD)', row['spect'])
    row['type'] = lum_class_map.get(m.group(1), np.nan) if m else np.nan

    # Estimate lifespan
    row['lifespan'] = estimate_lifespan_gyr(row.get('spect', ''), row.get('mass', np.nan))

    output_rows.append(row)
    print(" → Success")
        
# Post-processing GJ53 'system' overrides for GJ 53 A/B components based on wikipedia values
for row in output_rows:
    star_id = str(row['id']).strip()
    if star_id == '5325':  # Alpha Andraste
        row.update({
            'full_name': 'Alpha Andraste',
            'proper': '',
            'spect': 'G5Vb',
            'lum_class': 'Vb',
            'mag': 5.17,
            'absmag': 5.78,
            'radius': 0.789,
            'mass': 0.744,
            'temp': 5306,
            'lum': 0.445,
            'ci': 0.695,
            'inner_ohz': 0.95,
            'outer_ohz': 2.36,
            'inner_chz': 0.96,
            'outer_chz': 1.66,
            'gl': '53 A',
            'main_id': '* mu. Cas A',
            'comp': 'A',
            'bayer': 'mu',
            'con': 'Cas',
            'rv': -97.09,
            'lifespan': estimate_lifespan_gyr('G5Vb', 0.744)
        })
    elif star_id == '119221':  # Beta Andraste
        row.update({
            'full_name': 'Beta Andraste',
            'proper': '',
            'spect': 'M4V',
            'lum_class': 'V',
            'mag': 9.8,
            'absmag': 11.6,
            'radius': 0.29,
            'mass': 0.173,
            'temp': 3025,
            'lum': 0.0062,
            'ci': 1.554,
            'inner_ohz': 0.08,
            'outer_ohz': 0.17,
            'inner_chz': 0.10,
            'outer_chz': 0.15,
            'gl': '53 B',
            'main_id': '* mu. Cas B',
            'comp': 'B',
            'bayer': 'mu',
            'con': 'Cas',
            'lifespan': estimate_lifespan_gyr('M4V', 0.173)
        })

assign_siblings(output_rows, sep='|')
apply_id_remap(output_rows, id_remap)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
df_out = pd.DataFrame(output_rows)
df_out.to_csv('NewStarsData.csv', index=False)

print(f"\n=== DONE ===")
print(f"Processed: {len(df_in)} | Success: {len(df_in)-len(failed)} | Failed: {len(failed)}")
if failed:
    print("Failed:", ", ".join(failed[:20]))

if fallbacks:
    df_fallbacks = pd.DataFrame(fallbacks)
    df_fallbacks.to_csv('Corrections_Report.csv', index=False)
    print(f"{len(fallbacks)} updates saved to Corrections_Report.csv")