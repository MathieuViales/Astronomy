import base64
import html
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.io import fits


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_CSV = Path("manga_ids.csv")

OUTPUT_DIR = Path("output")
DOWNLOAD_DIR = OUTPUT_DIR / "fits"

# HTML table written at the end of the run
REPORT_FILE = OUTPUT_DIR / "spectra_table_generated.html"

CATALOG_URL = (
    "https://data.sdss.org/sas/dr17/manga/spectro/mastar/"
    "v3_1_1/v1_7_7/mastarall-v3_1_1-v1_7_7.fits"
)

CATALOG_FILE = DOWNLOAD_DIR / "mastarall-v3_1_1-v1_7_7.fits"

OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)


# ==========================================
# REFERENCE SPECTRAL LINES
# Vacuum wavelengths in Angstroms
# ==========================================

LINES = {
    "H I": [6564.61, 4862.68],
    "Na I": [5891.58, 5897.56],
    # H and K lines, then the near-infrared triplet
    "Ca II": [3934.78, 3970.00, 8500.36, 8544.44, 8664.52],
    "Mg I": [5174.13],
    "Fe I": [4384.09]
}

# Molecular absorption bands, shaded as wavelength ranges
# (Angstroms, from the practical's spectral line table)
BANDS = {
    "TiO": [
        (4900, 5200),
        (5400, 5700),
        (6200, 6300),
        (6700, 6900)
    ]
}

# Descriptions used in the index table
LINE_NOTES = {
    "H I": "Hydrogen (Hα & Hβ)",
    "Na I": "Sodium",
    "Ca II": "Calcium (H & K lines, near-infrared triplet)",
    "Mg I": "Magnesium",
    "Fe I": "Iron"
}

BAND_NOTES = {
    "TiO": "Titanium oxide (molecular bands)"
}


# ==========================================
# STAR INFORMATION FOR THE HTML TABLE
# Class, colour and temperature come from the
# practical; elements are the ones identified
# by eye in each spectrum. Edit freely.
# ==========================================

CLASS_ORDER = "OBAFGKM"

STAR_INFO = {
    "3-126639257": {
        "class": "O",
        "colour": "Blue",
        "temperature": "30,000",
        "elements": ["Hydrogen"]
    },
    "27-208": {
        "class": "B",
        "colour": "Blue-white",
        "temperature": "13,577",
        "elements": ["Hydrogen", "Calcium"]
    },
    "27-1802": {
        "class": "A",
        "colour": "White",
        "temperature": "8,228",
        "elements": ["Hydrogen", "Calcium"]
    },
    "13-5": {
        "class": "F",
        "colour": "Yellow-white",
        "temperature": "6,313",
        "elements": ["Hydrogen", "Calcium"],
        "note": (
            "What I'm confused by in the bottom right may just be "
            "noise? 13-6 & 13-7 only show an emission line in that "
            "area of the spectrum"
        )
    },
    "16-35": {
        "class": "G",
        "colour": "Yellow",
        "temperature": "5,964",
        "elements": ["Hydrogen", "Calcium", "Sodium", "Magnesium"]
    },
    "27-1622": {
        "class": "K",
        "colour": "Orange",
        "temperature": "3,920",
        "elements": [
            "Sodium", "Magnesium", "Calcium", "Titanium oxide"
        ]
    },
    "3-103789408": {
        "class": "M",
        "colour": "Red",
        "temperature": "3,594",
        "elements": [
            "Hydrogen (emission)", "Sodium", "Magnesium",
            "Calcium", "Titanium oxide"
        ]
    }
}

REPORT_INTRO = (
    "I've built a python script to download the spectra and add an "
    "overlay for some of common elements to make it easier to see :-)"
)


# ==========================================
# DOWNLOAD CATALOGUE
# ==========================================

def download_catalogue():

    if CATALOG_FILE.exists():
        print("Using cached MaStar catalogue.")
        return

    print("Downloading MaStar catalogue...")

    urllib.request.urlretrieve(
        CATALOG_URL,
        CATALOG_FILE
    )


# ==========================================
# NORMALISE MA NGA ID
# ==========================================

def normalise_id(value):

    if isinstance(value, bytes):
        value = value.decode("ascii")

    return str(value).strip()


# ==========================================
# DOWNLOAD INDIVIDUAL SPECTRUM
# ==========================================

def download_spectrum(plate, ifu):

    filename = f"mastar-LOG-{plate}-{ifu}.fits.gz"
    filepath = DOWNLOAD_DIR / filename

    if filepath.exists():
        return filepath

    url = (
        f"https://data.sdss.org/sas/dr17/manga/spectro/"
        f"redux/v3_1_1/{plate}/mastar/{filename}"
    )

    print(f"  Downloading {filename}")

    urllib.request.urlretrieve(url, filepath)

    return filepath


# ==========================================
# PLOT AND ANNOTATE SPECTRUM
# Returns the saved image path, or None
# ==========================================

def plot_spectrum(
    mangaid,
    plate,
    ifu,
    mjd,
    wavelength,
    flux,
    ivar
):

    # Create individual star folder
    star_dir = OUTPUT_DIR / mangaid
    star_dir.mkdir(exist_ok=True)

    # Valid pixels
    valid = (
        np.isfinite(wavelength) &
        np.isfinite(flux) &
        np.isfinite(ivar) &
        (ivar > 0)
    )

    if not np.any(valid):
        print("  No valid spectral data.")
        return None

    wave_min = wavelength[valid].min()
    wave_max = wavelength[valid].max()

    fig, ax = plt.subplots(figsize=(18, 7))

    ax.plot(
        wavelength[valid],
        flux[valid],
        color="black",
        linewidth=0.7
    )

    # Reference line annotations
    for element, positions in LINES.items():

        for position in positions:

            if wave_min <= position <= wave_max:

                ax.axvline(
                    position,
                    color="red",
                    alpha=0.35,
                    linestyle="--"
                )

                ax.text(
                    position,
                    0.95,
                    element,
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    color="darkred",
                    ha="right",
                    va="top"
                )

    # Molecular band annotations (shaded ranges)
    for molecule, ranges in BANDS.items():

        for low, high in ranges:

            if high < wave_min or low > wave_max:
                continue

            low = max(low, wave_min)
            high = min(high, wave_max)

            ax.axvspan(
                low,
                high,
                color="orange",
                alpha=0.15,
                linewidth=0
            )

            ax.text(
                (low + high) / 2,
                0.95,
                molecule,
                rotation=90,
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                color="darkorange",
                ha="center",
                va="top"
            )

    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux (10⁻¹⁷ erg/s/cm²/Å)")

    ax.set_title(
        f"MaStar {mangaid} | "
        f"Plate {plate} | IFU {ifu} | MJD {mjd}"
    )

    ax.grid(alpha=0.2)

    plt.tight_layout()

    output_file = (
        star_dir /
        f"spectrum-{plate}-{ifu}-{mjd}.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"  Saved {output_file}")

    return output_file


# ==========================================
# PROCESS ONE MA NGA ID
# Only the first good visit is used
# ==========================================

def process_star(mangaid, goodvisits):

    print(f"\nProcessing MaNGA ID: {mangaid}")

    matches = [
        row for row in goodvisits
        if normalise_id(row["MANGAID"]) == mangaid
    ]

    if not matches:
        print("  No good-quality observations found.")
        return []

    print(f"  Found {len(matches)} good visits.")

    if len(matches) > 1:
        print("  Using the first visit only.")

    visit = matches[0]

    plate = int(visit["PLATE"])
    ifu = int(visit["IFUDESIGN"])
    mjd = int(visit["MJD"])

    try:

        filepath = download_spectrum(plate, ifu)

        with fits.open(filepath) as hdul:

            spectra = hdul["MASTAR"].data

            visit_data = spectra[
                spectra["MJD"] == mjd
            ]

            if len(visit_data) == 0:
                print(f"  MJD {mjd} not found in FITS.")
                return [{
                    "mangaid": mangaid,
                    "plate": plate,
                    "ifu": ifu,
                    "mjd": mjd,
                    "status": "mjd_not_found",
                    "image": None
                }]

            spectrum = visit_data[0]

            wavelength = np.array(spectrum["WAVE"])
            flux = np.array(spectrum["FLUX"])
            ivar = np.array(spectrum["IVAR"])

        image = plot_spectrum(
            mangaid,
            plate,
            ifu,
            mjd,
            wavelength,
            flux,
            ivar
        )

        return [{
            "mangaid": mangaid,
            "plate": plate,
            "ifu": ifu,
            "mjd": mjd,
            "status": "success" if image else "no_valid_data",
            "image": image
        }]

    except Exception as error:

        print(f"  ERROR: {error}")

        return [{
            "mangaid": mangaid,
            "plate": plate,
            "ifu": ifu,
            "mjd": mjd,
            "status": f"error: {error}",
            "image": None
        }]


# ==========================================
# HTML REPORT
# All styling is inline (no <style> block)
# ==========================================

CELL = (
    "padding: 8px; border: 1px solid #cccccc; "
    "vertical-align: middle;"
)

HEAD = (
    "padding: 8px; border: 1px solid #cccccc; "
    "background-color: #f3f3f3;"
)


def esc(text):
    return html.escape(str(text))


def embed_image(path):

    data = base64.b64encode(
        Path(path).read_bytes()
    ).decode("ascii")

    return f"data:image/png;base64,{data}"


def build_index_table():

    rows = ""

    for label, positions in LINES.items():

        wavelengths = ", ".join(f"{p:.2f}" for p in positions)

        rows += (
            f'<tr><td style="{CELL}">{esc(label)}</td>'
            f'<td style="{CELL}">{wavelengths}</td>'
            f'<td style="{CELL}">'
            f'{esc(LINE_NOTES.get(label, ""))}</td></tr>\n'
        )

    for label, ranges in BANDS.items():

        wavelengths = ", ".join(
            f"{low:.0f}–{high:.0f}" for low, high in ranges
        )

        rows += (
            f'<tr><td style="{CELL}">{esc(label)}</td>'
            f'<td style="{CELL}">{wavelengths}</td>'
            f'<td style="{CELL}">'
            f'{esc(BAND_NOTES.get(label, ""))}</td></tr>\n'
        )

    return (
        '<table style="border-collapse: collapse; '
        'font-family: Arial, sans-serif; font-size: 13px;">\n'
        "<thead><tr>"
        f'<th scope="col" style="{HEAD}">Label</th>'
        f'<th scope="col" style="{HEAD}">'
        "Wavelengths (Å, vacuum)</th>"
        f'<th scope="col" style="{HEAD}">Notes</th>'
        "</tr></thead>\n"
        f"<tbody>\n{rows}</tbody>\n</table>\n"
    )


def sort_key(result_with_position):

    position, result = result_with_position

    info = STAR_INFO.get(result["mangaid"], {})
    star_class = info.get("class", "")

    if star_class in CLASS_ORDER:
        return (CLASS_ORDER.index(star_class), position)

    return (len(CLASS_ORDER), position)


def build_spectra_table(results):

    widths = ["7.1%", "6.4%", "9.2%", "7.3%", "44%", "26%"]
    headings = [
        "Class", "Colour", "Temperature (K)",
        "MaNGA ID", "Spectrum", "Elements"
    ]

    head_cells = ""

    for width, heading in zip(widths, headings):

        align = " text-align: left;" if heading == "Elements" else ""

        head_cells += (
            f'<th style="{HEAD} width: {width};{align}">'
            f"{heading}</th>"
        )

    ordered = sorted(
        enumerate(results),
        key=sort_key
    )

    rows = ""

    for _, result in ordered:

        mangaid = result["mangaid"]
        info = STAR_INFO.get(mangaid, {})

        star_class = esc(info.get("class", "-"))
        colour = esc(info.get("colour", "-"))
        temperature = esc(info.get("temperature", "-"))

        elements = "<br />".join(
            esc(name) for name in info.get("elements", [])
        )

        caption = (
            f"MaStar {mangaid} | Plate {result['plate']} | "
            f"IFU {result['ifu']} | MJD {result['mjd']}"
        )

        note = ""

        if info.get("note"):
            note = (
                '<p style="text-align: left; margin: 8px 0 0;">'
                f"{esc(info['note'])}</p>"
            )

        rows += (
            "<tr>\n"
            f'<td style="{CELL} width: {widths[0]};">'
            f'<strong style="font-size: 16px;">{star_class}'
            "</strong></td>\n"
            f'<td style="{CELL} width: {widths[1]};">{colour}</td>\n'
            f'<td style="{CELL} width: {widths[2]};">'
            f"{temperature}</td>\n"
            f'<td style="{CELL} width: {widths[3]};">'
            f"{esc(mangaid)}</td>\n"
            f'<td style="{CELL} width: {widths[4]};">'
            f'<img src="{embed_image(result["image"])}" '
            f'alt="MaStar {esc(mangaid)} spectrum" '
            'style="max-width: 100%; height: auto; '
            'display: block; margin: 0 auto;" />'
            '<p style="color: #555555; font-size: 12px; '
            f'margin: 6px 0 0;">{esc(caption)}</p>{note}</td>\n'
            f'<td style="{CELL} width: {widths[5]}; '
            f'text-align: left;">{elements}</td>\n'
            "</tr>\n"
        )

    return (
        '<table style="width: 100%; border-collapse: collapse; '
        "text-align: center; font-family: Arial, sans-serif; "
        'font-size: 13px;">\n'
        f"<thead><tr>{head_cells}</tr></thead>\n"
        f"<tbody>\n{rows}</tbody>\n</table>\n"
    )


def write_report(results):

    successes = [
        result for result in results
        if result["status"] == "success" and result["image"]
    ]

    if not successes:
        print("No spectra to put in the HTML table.")
        return

    document = (
        '<!doctype html>\n<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" '
        'content="width=device-width,initial-scale=1">'
        "<title>MaStar stellar spectra</title></head>\n"
        '<body style="font-family: Arial, sans-serif; '
        'font-size: 14px; margin: 24px; color: #222222;">\n'
        f"<p>{esc(REPORT_INTRO)}</p>\n"
        f"{build_index_table()}"
        "<p>&nbsp;</p>\n"
        f"{build_spectra_table(successes)}"
        "</body></html>\n"
    )

    REPORT_FILE.write_text(document, encoding="utf-8")

    print(f"HTML table: {REPORT_FILE.resolve()}")


# ==========================================
# MAIN
# ==========================================

def main():

    print("MaStar Batch Spectrum Analysis")
    print("================================")

    # Check input
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    stars = pd.read_csv(
        INPUT_CSV,
        dtype=str,
        encoding="utf-8-sig"
    )

    if "mangaid" not in stars.columns:
        raise ValueError(
            "CSV must contain a mangaid column."
        )

    mangaids = [
        normalise_id(value)
        for value in stars["mangaid"].dropna()
    ]

    print(f"Found {len(mangaids)} MaNGA IDs.")

    # Download catalogue
    download_catalogue()

    print("Reading MaStar catalogue...")

    with fits.open(CATALOG_FILE) as hdul:

        goodvisits = hdul["GOODVISITS"].data

        # Copy data into memory before closing FITS
        goodvisits = goodvisits.copy()

    # Process all requested stars
    all_results = []

    for mangaid in mangaids:

        results = process_star(
            mangaid,
            goodvisits
        )

        all_results.extend(results)

    # Export processing report
    if all_results:

        report = pd.DataFrame(all_results).drop(
            columns=["image"]
        )

        report.to_csv(
            OUTPUT_DIR / "processing_log.csv",
            index=False
        )

        write_report(all_results)

    print("\n================================")
    print("Batch processing complete.")
    print(f"Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
