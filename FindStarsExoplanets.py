import pandas as pd
import numpy as np
import re

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
stars = pd.read_csv("NewStarsData.csv")
exo = pd.read_csv("exoplanet.eu.csv")
exo = exo.fillna(np.nan)

# === AGGRESSIVE EXO MATCHING (REPLACES YOUR OLD BLOCK) ===
greek_map = {'ALF':'ALPHA','BET':'BETA','GAM':'GAMMA','DEL':'DELTA','EPS':'EPSILON',
            'ZET':'ZETA','ETA':'ETA','THE':'THETA','IOT':'IOTA','KAP':'KAPPA',
            'LAM':'LAMBDA','MU':'MU','NU':'NU','XI':'XI','OMI':'OMICRON',
            'PI':'PI','RHO':'RHO','SIG':'SIGMA','TAU':'TAU','UPS':'UPSILON',
            'PHI':'PHI','CHI':'CHI','PSI':'PSI','OME':'OMEGA'}

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def is_valid_str(val) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "nan"

def clean(val):
    if val is None:
        return ""
    # If it's a float and NaN, bail out
    if isinstance(val, float):
        if np.isnan(val):
            return ""
        # convert numeric to string
        return str(val).upper()
    # Otherwise, assume string
    return re.sub(r"\s+", " ", str(val).strip()).upper()

def normalize_planet_suffix(name: str) -> str:
    """
    HD 19994 Ab -> HD 19994 A
    GJ 128 Ab   -> GJ 128 A
    HR 962 Ab   -> HR 962 A
    Only acts if there's a trailing 'Component + planet' pattern.
    """
    if not is_valid_str(name):
        return ""
    name_u = clean(name)
    # Match trailing ' <Component><planet>' e.g., ' A b' or 'Ab'
    # Ensure the last char is lowercase (planet), and preceding is uppercase (component)
    m = re.search(r"(.*\b[A-Z])([A-Z])([A-Z]?)?([A-Z]?)?$", name_u)  # safeguard
    # Safer explicit rule: strip trailing ' [A-Z][a-z]' -> keep the [A-Z]
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)  # accidental case (e.g., 'AB')
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)  # idempotent
    # Correct pattern: space + uppercase component + lowercase planet at end
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)  # guard (noop mostly)
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)  # guard (noop)
    # Final: properly handle component + planet (Ab, Bb, etc.)
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)  # keep component only (noop if not found)
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)
    # Best explicit: if endswith ' Ab' / ' Bb' ... replace with ' A' / ' B'
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)
    # In practice, a simpler, reliable rule:
    name_u = re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)
    # And handle compact forms like 'HD 19994 AB' mistakenly; we won't alter non-matching cases.
    return re.sub(r"\s+([A-Z])[A-Z]$", r" \1", name_u)

def strip_lowercase_planet_suffix(name: str) -> str:
    """
    A reliable minimal normalizer:
    - If the name ends with ' <Component><planet>' where planet is lowercase, keep only the component.
      e.g., 'HD 19994 Ab' -> 'HD 19994 A'
    - Otherwise, return cleaned uppercase name unchanged.
    """
    if not is_valid_str(name):
        return ""
    name_u = clean(name)
    return re.sub(r"\s+([A-Z])[a-z]$", r" \1", name_u)

def strip_component(name: str) -> str:
    """
    Remove trailing ' A' / ' B' / ' C' components to match system-level star names, e.g. 'GJ 667 A' -> 'GJ 667'.
    """
    if not is_valid_str(name):
        return ""
    name_u = clean(name)
    return re.sub(r"\s+[A-Z]$", "", name_u)

def alias_variants(base: str):
    """
    Generate matching variants for a star alias:
    - cleaned uppercase
    - with planet suffix stripped
    - without component
    """
    if not is_valid_str(base):
        return set()
    a = clean(base)
    variants = {a}
    a1 = strip_lowercase_planet_suffix(a)
    variants.add(a1)
    variants.add(strip_component(a1))
    return {v for v in variants if is_valid_str(v)}

# -------------------------------------------------------
# Build alias list for each star for matching
# -------------------------------------------------------
def build_aliases(row):
    aliases = set()
    
    def add(val):
        if not is_valid_str(val):
            return
        c = clean(val).replace("*", "").strip()
        if not c: return
        aliases.add(c)
        aliases.add(c.replace(" ", ""))
        aliases.add(strip_component(c))
        aliases.add(strip_component(c.replace(" ", "")))
        # Greek expansion
        for short, long in greek_map.items():
            if short in c.upper():
                expanded = c.upper().replace(short, long)
                aliases.add(expanded)
                aliases.add(expanded.replace(" ", ""))
                aliases.add(strip_component(expanded))

    # Main identifiers
    add(row.get("main_id"))
    add(row.get("full_name"))
    add(row.get("proper"))
    add(row.get("bayer"))
    add(row.get("flam"))  # Flamsteed numbers!
    
    # Numeric IDs — generate both with and without prefix
    for prefix, field in [("HD", "hd"), ("HIP", "hip"), ("HR", "hr"), ("GJ", "gl"), ("GL", "gl")]:
        val = row.get(field)
        if is_valid_str(val):
            s = str(val).strip().upper()
            num = s.replace("A","").replace("B","").replace("C","")
            if num.isdigit() or (len(num)>3 and num[1:].isdigit()):
                aliases.add(f"{prefix} {s}")
                aliases.add(f"{prefix}{s}")
                aliases.add(f"{prefix} {num}")
                aliases.add(f"{prefix}{num}")
                aliases.add(num)  # just the number

    # Bayer + constellation — critical
    bayer = clean(row.get("bayer") or "").replace("*", "")
    con = clean(row.get("con") or "")
    if bayer and con:
        variants = [
            f"{bayer} {con}",
            f"{bayer}{con}",
            f"{bayer.upper()} {con.upper()}",
            f"{bayer.upper()}{con.upper()}",
        ]
        for v in variants:
            aliases.add(v)
            # Expand Greek
            for short, long in greek_map.items():
                if short in v.upper():
                    exp = v.upper().replace(short, long)
                    aliases.add(exp)
                    aliases.add(exp.replace(" ", ""))

    # 2MASS, TYC, Gaia — add raw and cleaned
    for field in ["2mass", "tyc", "gaia", "tic"]:
        val = row.get(field)
        if is_valid_str(val):
            aliases.add(clean(val))
            aliases.add(clean(val).replace("-", "").replace("+", ""))

    return {a for a in aliases if len(a) >= 3}

stars["aliases"] = [build_aliases(row) for _, row in stars.iterrows()]

# -------------------------------------------------------
# Build exoplanet lookup dictionary
# -------------------------------------------------------
def add_exo_keys(row):
    """Extract every possible name variant from an exoplanet row"""
    keys = set()

    def add(name):
        if not is_valid_str(name):
            return
        c = clean(name)
        if not c: return
        keys.add(c)
        # strip planet letter
        keys.add(strip_lowercase_planet_suffix(c))
        # strip component A/B/C
        keys.add(strip_component(strip_lowercase_planet_suffix(c)))
        # also add without spaces (common in 2MASS etc.)
        keys.add(c.replace(" ", ""))

    # Main fields
    add(row.get("star_name"))
    add(row.get("name"))  # sometimes planet name has host

    # alternate_names and star_alternate_names are GOLD
    for field in ["alternate_names", "star_alternate_names"]:
        val = row.get(field)
        if is_valid_str(val):
            for sep in [";", ",", " | ", "|"]:
                for part in str(val).split(sep):
                    part = part.strip().split("://")[0]  # kill URLs
                    if part.count(" ") >= 1 or any(prefix in part for prefix in ["HD", "HIP", "GJ", "HR", "WDS", "TOI", "K2", "TIC"]):
                        add(part)

def aggressive_exo_keys(row):
    keys = set()
    def add(s):
        if not is_valid_str(s): return
        c = clean(s)
        if not c: return
        keys.add(c)
        keys.add(strip_lowercase_planet_suffix(c))
        keys.add(strip_component(c))
        keys.add(c.replace(" ",""))
        u = c.upper()
        for gshort, glong in greek_map.items():
            if gshort in u:
                keys.add(u.replace(gshort, glong))
    for field in ["star_name", "name", "alternate_names", "star_alternate_names"]:
        val = row.get(field)
        if is_valid_str(val):
            for sep in [";", ",", "|"]:
                for part in str(val).split(sep):
                    add(part.strip())
    return keys

exo_grouped = {}
for _, row in exo.iterrows():
    for k in aggressive_exo_keys(row):
        exo_grouped.setdefault(k, []).append(row)

# -------------------------------------------------------
# Match planets to each star
# -------------------------------------------------------
planet_rows = []

for _, srow in stars.iterrows():
    matched_keys = set()
    for alias in srow["aliases"]:
        if alias in exo_grouped:
            matched_keys.add(alias)
        # Also check variants in case
        for v in [alias, alias.replace(" ",""), strip_component(alias)]:
            if v in exo_grouped:
                matched_keys.add(v)

    for key in matched_keys:
        for p in exo_grouped[key]:
            planet_rows.append({
                "star": srow.get("full_name") or srow.get("main_id"),
                "alias_matched": key,
                "planet": p.get("name"),
                "planet_status": p.get("planet_status"),
                "host_star_name": p.get("star_name"),
                "mass": p.get("mass"),
                "radius": p.get("radius"),
                "period_days": p.get("orbital_period"),
                "sma_au": p.get("semi_major_axis"),
                "eccentricity": p.get("eccentricity"),
                "temp_eq": p.get("temp_calculated"),
                "discovered": p.get("discovered"),
                "updated": p.get("updated"),
                "detection_type": p.get("detection_type"),
                "alternate_names": p.get("alternate_names"),
                "molecules": p.get("molecules"),
            })

# deduplicate planets
planets_df = pd.DataFrame(planet_rows)

# Keep only unique (star + planet name) combinations
planets_df = planets_df.drop_duplicates(subset=["star", "planet"])

# Optional: keep the most recent update if multiple
planets_df = planets_df.sort_values("updated", ascending=False)
planets_df = planets_df.drop_duplicates(subset=["star", "planet"], keep="first")

planets_df.to_csv("MatchedPlanets.csv", index=False)
print(f"Done! {len(planets_df)} unique planets saved.")
