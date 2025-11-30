import re
import numpy as np
import pandas as pd
from typing import cast

# Constants
SOLAR_RADIUS_KM = 695700

# Hardcoded Bethesda-style overrides
spectral_overrides = {
    "DM6": "M6",
    "DQZ": "WD",
    "DZ7.5": "WD"
}

def round_for_comparison(df):
    # Define rounding groups by precision
    round4 = [
        "lum", "AU dist",
        "ra", "dec", "mag", "ci", "x", "y", "z", "dist",
    ]
    round3 = ["radius", "absmag"]
    round2 = ["mass", "Inner OHZ", "Outer OHZ"]
    round8 = ["rarad", "decrad", "pmrarad", "pmdecrad", "vx", "vy", "vz"]
    round10 = ["Temp"]

    # Convert all possibly numeric columns first
    cols = round4 + round3 + round2 + round8 + round10
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply grouped rounding
    for col in round4:
        if col in df.columns:
            df[col] = df[col].round(4)

    for col in round3:
        if col in df.columns:
            df[col] = df[col].round(3)

    for col in round2:
        if col in df.columns:
            df[col] = df[col].round(2)

    for col in round8:
        if col in df.columns:
            df[col] = df[col].round(8)

    # Temperature — round to nearest 10 K
    for col in round10:
        if col in df.columns:
            df[col] = df[col].round(-1)

    return df

def normalize_spectral_class(s):
    """Extract only core letter and subclass digit, ignoring luminosity and peculiarities."""
    if pd.isna(s):
        return None
    s = s.strip().upper()
    if s in spectral_overrides:
        return spectral_overrides[s]
    if any(x in s for x in ["WD", "DQ", "DA", "DZ", "DM"]):
        return "WD"
    match = re.search(r'([OBAFGKMW])\s*([0-9]?)', s)
    if not match:
        return s
    return f"{match.group(1)}{match.group(2)}".strip()

def spectral_class_equivalent(game_val, cat_val):
    g = normalize_spectral_class(game_val)
    c = normalize_spectral_class(cat_val)
    return g == c

# === Load CSVs ===
id_cols = ["hip","hd","hr","gliese","bf","proper","lp","ltt","giclas",
           "2mass","gaia","tyc","ccdm","wds","ads","nltt","plx","fk5",
           "iras","gcrv","tic","web"]
catalog_df = pd.read_csv("NewStarsData.csv", dtype={c: str for c in id_cols})
game_df = pd.read_csv("StarFormData.csv", delimiter="|", dtype={"StarID": str})

# === Round numeric values for comparison ===
catalog_df = round_for_comparison(catalog_df)

# print("Game CSV columns:", game_df.columns.tolist())

# === Normalize keys ===
game_df["StarID"] = game_df["StarID"].astype(str)
catalog_df["id"] = catalog_df["id"].astype(str)
catalog_df["gaia"] = catalog_df["gaia"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "")

# === Convert game CSV radius from km -> solar radii ===
game_df['StarRadius_Rsun'] = game_df['StarRadius_km'] / SOLAR_RADIUS_KM
game_df['StarRadius_Rsun'] = game_df['StarRadius_Rsun'].round(3)

# === Fields to compare ===
field_map = {
    "StarID": "id",
    "Name(ANAM)": "full_name",  # Compare game's Name(ANAM) (e.g., "Al-Battani") to catalog's full_name (e.g., "Al-Battani")
    "SpectralClass": "spect",
    "Magnitude": "absmag",
    "StarMass": "mass",
    "StarRadius_Rsun": "radius",
    "Temperature_K": "temp",
    "DistanceFromOrigin_pc": "dist",
    "InnerHZ": "inner_ohz",
    "OuterHZ": "outer_ohz"
}

mismatch_column_names = {
    "Name(ANAM)": "STDT_Mismatch_FullName",  # Updated key for the name mismatch column
    "SpectralClass": "STDT_Mismatch_Class",
    "Magnitude": "STDT_Mismatch_Magnitude",
    "StarMass": "STDT_Mismatch_Mass",
    "StarRadius_Rsun": "STDT_Mismatch_Radius",
    "Temperature_K": "STDT_Mismatch_Temperature",
    "DistanceFromOrigin_pc": "STDT_Mismatch_Distance",
    "InnerHZ": "STDT_Mismatch_InnerHZ",
    "OuterHZ": "STDT_Mismatch_OuterHZ"
}

# === Comparison logic ===
def compare_rows(game_row, catalog_row):
    mismatches = []
    for game_field, c_field in field_map.items():
        if game_field == "StarID":
            continue
        game_val = game_row.get(game_field)
        c_val = catalog_row.get(c_field)

        # Skip if None or NaN
        if game_val is None or c_val is None or pd.isna(game_val) or pd.isna(c_val):
            continue

        if game_field == "SpectralClass":
            game_val = normalize_spectral_class(game_val)
            c_val = normalize_spectral_class(c_val)
            if not spectral_class_equivalent(game_val, c_val):
                mismatches.append((game_field, game_val, c_val))
        else:
            # Numeric comparison (exact)
            try:
                gv = float(game_val)
                cv = float(c_val)
                if gv != cv:
                    display_game_val = int(round(gv)) if game_field == "Temperature_K" else game_val
                    display_c_val = int(round(cv)) if game_field == "Temperature_K" else c_val
                    mismatches.append((game_field, display_game_val, display_c_val))
            except (ValueError, TypeError):
                # Fallback to string comparison
                def normalize_str(s):
                    return re.sub(r"[\s']", "", str(s)).lower().strip()
                if normalize_str(game_val) != normalize_str(c_val):
                    mismatches.append((game_field, game_val, c_val))
    return mismatches

# === Build mismatch report ===
report = []
for _, game_row in game_df.iterrows():
    star_id = str(game_row["StarID"])
    match = catalog_df[catalog_df["id"] == star_id]
    if match.empty:
        report.append((star_id, game_row["Name(ANAM)"], "Missing in catalog", "", ""))  # Use "Name(ANAM)" here too
        continue
    catalog_row = match.iloc[0]
    mismatches = compare_rows(game_row, catalog_row)
    for field, game_val, c_val in mismatches:
        report.append((star_id, game_row["Name(ANAM)"], field, game_val, c_val))  # Use "Name(ANAM)"

# --- Normalize comp_primary ---
catalog_df["comp_primary"] = (
    catalog_df["comp_primary"]
    .fillna("")
    .astype(str)
    .str.strip()
    .replace("", np.nan)
)

# === Calculate companion distances (system-based) ===
# A star is a "primary" if its comp == 'A' or comp is NaN/blank
is_primary = catalog_df["comp"].astype(str).str.upper().str.strip().isin(["A", "", "NAN"])
primary_coords_map = (
    catalog_df[is_primary & catalog_df['comp_primary'].notna() & (catalog_df['comp_primary'] != '')]
    .groupby('comp_primary')[['x','y','z']]
    .first()  # take the first primary coordinates
    .to_dict(orient='index')
)
print("Sample primary map keys:", list(primary_coords_map.keys())[:10])
print(catalog_df["comp_primary"].unique()[:20])
print(catalog_df["comp_primary"].isna().sum(), "NaN entries")
print((catalog_df["comp_primary"] == "").sum(), "empty strings")
print("Sample primary map keys:", list(primary_coords_map.keys())[:10])
print("Sample comp_primary values:", catalog_df["comp_primary"].dropna().unique()[:10])

# --- Compute 3D distance ---
def compute_distance(row):
    primary_name = row["comp_primary"]
    if pd.isna(primary_name):
        return np.nan
    primary_name = primary_name.strip()
    if primary_name not in primary_coords_map:
        return np.nan
    p = primary_coords_map[primary_name]
    if any(pd.isna([row["x"], row["y"], row["z"], p["x"], p["y"], p["z"]])):
        return np.nan
    dx = row["x"] - p["x"]
    dy = row["y"] - p["y"]
    dz = row["z"] - p["z"]
    return np.sqrt(dx * dx + dy * dy + dz * dz)
   
catalog_df["compdist"] = catalog_df.apply(compute_distance, axis=1)
catalog_df["compdist"] = catalog_df["compdist"].round(6)

# FIX: Avoid FutureWarning by casting to object before setting ''
catalog_df["compdist"] = catalog_df["compdist"].astype(object)
catalog_df.loc[catalog_df["compdist"] == 0, "compdist"] = ""
def fix_bgs_fields(df):
    # 1. base
    df['base'] = df['full_name']
    # 2. comp_numeric
    comp_rank_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    df['comp_numeric'] = (df['comp'].astype(str).str.upper().map(comp_rank_map).fillna(1).astype(int))
    # 3. comp_primary = own id if primary, else as is
    df['comp_primary'] = df.apply(
        lambda row: row['id'] if row['comp_numeric'] == 1 or pd.isna(row['comp']) else row['comp_primary'],
        axis=1
    )
    # 4. Map primary bayer/flam/con
    # FIX: fillna before to_dict to handle NaNs as ''
    primaries = df[df['comp_numeric'] == 1].set_index('id')[['bayer','flam','con']].fillna('').to_dict(orient='index')
    def map_primary_bfc(row):
        pid = row['comp_primary']
        if pid in primaries:
            return pd.Series(primaries[pid])
        return pd.Series({'bayer':'','flam':'','con':''})
    df[['bayer','flam','con']] = df.apply(map_primary_bfc, axis=1)
    # FIX: Ensure str dtype after apply to avoid concat issues
    df['bayer'] = df['bayer'].astype(str)
    df['flam'] = df['flam'].astype(str)
    df['con'] = df['con'].astype(str)
    # 5. Build bf
    df['bf'] = (df['bayer'] + ' ' + df['flam'] + ' ' + df['con']).str.strip()
    # 6. Copy numeric rank to comp column
    df['comp'] = df['comp_numeric']
    return df
    
catalog_df = fix_bgs_fields(catalog_df)

# === Output ===
report_df = pd.DataFrame(report, columns=["StarID", "Name", "Field", "Game Value", "Catalog Value"])
report_df.to_csv("StarMismatchReport.csv", index=False)
print("Mismatch report saved to StarMismatchReport.csv")

# === Prepare catalog_df to receive mismatch info ===
for col in mismatch_column_names.values():
    if col not in catalog_df.columns:
        catalog_df[col] = None

# Build a lookup from StarID -> list of mismatches
mismatch_lookup = {}
for _, row in report_df.iterrows():
    star_id = str(row["StarID"])
    field = row["Field"]
    game_val = row["Game Value"]
    c_val = row["Catalog Value"]
    if field in ["Magnitude", "StarRadius_Rsun", "DistanceFromOrigin_pc", "InnerHZ", "OuterHZ"]:
        game_val_str = str(game_val)
        cat_val_str = str(c_val)
        try:
            game_val_num = float(game_val)
            cat_val_num = float(c_val)
            game_val_str = f"{game_val_num:.2f}"
            cat_val_str = f"{cat_val_num:.2f}"
        except (ValueError, TypeError):
            pass  # keep the string versions
        mismatch_text = f"{game_val_str} <-> {cat_val_str}"
    else:
        # For non-HZ fields (mass, radius, etc.), keep your existing logic
        display_game_val = int(round(float(game_val))) if field == "Temperature_K" else game_val
        display_c_val = int(round(float(c_val))) if field == "Temperature_K" else c_val
        mismatch_text = f"{display_game_val} <-> {display_c_val}"
    if star_id not in mismatch_lookup:
        mismatch_lookup[star_id] = {}
    mismatch_lookup[star_id][field] = mismatch_text

# Apply mismatches back to catalog_df
for idx, row in catalog_df.iterrows():
    star_id = str(row["id"])
    if star_id in mismatch_lookup:
        for field, mismatch_text in mismatch_lookup[star_id].items():
            col_name = mismatch_column_names[field]
            catalog_df.loc[idx, col_name] = mismatch_text # type: ignore

# Save the updated catalog with mismatches appended
catalog_df.to_csv("NewStarsData_WithMismatches.csv", index=False)
print("Updated catalog with mismatches saved to NewStarsData_WithMismatches.csv")