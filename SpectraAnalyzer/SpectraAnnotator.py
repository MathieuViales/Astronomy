
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
    "Ca II": [3934.78, 3970.00],
    "Mg I": [5174.13],
    "Fe I": [4384.09]
}


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
        return False

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

            if (
                wavelength[valid].min()
                <= position
                <= wavelength[valid].max()
            ):

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

    return True


# ==========================================
# PROCESS ONE MA NGA ID
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

    results = []

    # Each visit has its own MJD
    for visit in matches:

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
                    continue

                spectrum = visit_data[0]

                wavelength = np.array(spectrum["WAVE"])
                flux = np.array(spectrum["FLUX"])
                ivar = np.array(spectrum["IVAR"])

            success = plot_spectrum(
                mangaid,
                plate,
                ifu,
                mjd,
                wavelength,
                flux,
                ivar
            )

            results.append({
                "mangaid": mangaid,
                "plate": plate,
                "ifu": ifu,
                "mjd": mjd,
                "status": "success" if success else "no_valid_data"
            })

        except Exception as error:

            print(f"  ERROR: {error}")

            results.append({
                "mangaid": mangaid,
                "plate": plate,
                "ifu": ifu,
                "mjd": mjd,
                "status": f"error: {error}"
            })

    return results


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
        dtype=str
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

        report = pd.DataFrame(all_results)

        report.to_csv(
            OUTPUT_DIR / "processing_log.csv",
            index=False
        )

    print("\n================================")
    print("Batch processing complete.")
    print(f"Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
