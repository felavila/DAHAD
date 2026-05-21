from sparcl.client import SparclClient

from astropy.table import Table
from astropy.io import fits
import os
import pandas as pd
from dl import queryClient as qc

client = SparclClient(connect_timeout=3.1)
file_path = 'VAC_BHmass_338_v1.7.fits'
hdul = fits.open(file_path)
table_data = Table(hdul[1].data)
table_pandas = table_data.to_pandas().sort_values("RMAG", ascending=True)

# Directory for output spectra
out_dir = 'spectra_fits'
os.makedirs(out_dir, exist_ok=True)

# Build set of already-downloaded target IDs from existing files
downloaded = {
    int(fname.split('_')[-1].split('.')[0])
    for fname in os.listdir(out_dir)
    if fname.startswith('spectrum_targetid_') and fname.endswith('.fits')
}

radius = 1.8 / 3600.0  # ~1.8″
incs = [
    "data_release", "datasetgroup", "dateobs", "dateobs_center", "dec",
    "exptime", "extra_files", "file", "flux", "instrument", "ivar", "mask",
    "model", "ra", "redshift", "redshift_err", "redshift_warning", "site",
    "sparcl_id", "specid", "specprimary", "spectype", "survey", "targetid",
    "telescope", "updated", "wave_sigma", "wavelength", "wavemax", "wavemin"
]

# Function to write spectrum as FITS

def make_spectrum_fits(filename, wave, flux, ivar, wave_sigma, header_meta):
    primary_hdu = fits.PrimaryHDU()
    for key, value in header_meta.items():
        primary_hdu.header[key] = value

    cols = [
        fits.Column(name='WAVELENGTH', format='D', unit='Angstrom', array=wave),
        fits.Column(name='FLUX', format='E', unit='1e-17 erg cm-2 s-1 AA-1', array=flux),
        fits.Column(name='IVAR', format='E', unit='1e+34 cm4 s2 AA2 erg-2', array=ivar),
        fits.Column(name='WAVE_SIGMA', format='E', unit='pixel', array=wave_sigma),
    ]
    table_hdu = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')

    hdulist = fits.HDUList([primary_hdu, table_hdu])
    hdulist.writeto(filename, overwrite=True)

# Loop through each target in the input table
for _, row in table_pandas.iterrows():
    print("_")
    targetid = int(row['TARGETID'])
    if targetid in downloaded:
        print(f"Skipping target {targetid}: already downloaded.")
        continue

    ra, dec = row['RA'], row['DEC']

    # Build sky constraint box + data release
    cons = {
        'ra':  [ra - radius, ra + radius],
        'dec': [dec - radius, dec + radius],
        'data_release': ['DESI-DR1']
    }
    outfields = ['sparcl_id', 'specid', 'ra', 'dec', 'redshift']

    # Query SPARCL for nearest spectrum
    found = client.find(outfields=outfields, constraints=cons, limit=1)
    if found.count == 0:
        print(f"No spectrum within {radius}° of ({ra}, {dec}) for target {targetid}")
        continue

    sparcl_id = found.ids[0]

    # Retrieve spectrum record
    retrieved = client.retrieve(uuid_list=[sparcl_id], include=incs)
    rec = retrieved.records[0]

    # Fetch photometry for this single target
    sql = f"""
    SELECT targetid, flux_g, flux_r, flux_z,
           mw_transmission_g, mw_transmission_r, mw_transmission_z
    FROM desi_dr1.photometry
    WHERE targetid = {targetid}
    """
    phot = qc.query(sql=sql, fmt='table').to_pandas().iloc[0]

    # Prepare header metadata
    header_meta = {
        'SPARCL_ID':      rec['sparcl_id'],
        'TARGETID':       rec['targetid'],
        'RA':             rec['ra'],
        'DEC':            rec['dec'],
        'REDSHIFT':       rec['redshift'],
        'SPECTYPE':       rec['spectype'],
        'REDSHIFT_ERR':   rec['redshift_err'],
        'REDSHIFT_WARN':  rec['redshift_warning'],
        'FLUX_G':         phot['flux_g'],
        'FLUX_R':         phot['flux_r'],
        'FLUX_Z':         phot['flux_z'],
        'MW_TRANS_G':     phot['mw_transmission_g'],
        'MW_TRANS_R':     phot['mw_transmission_r'],
        'MW_TRANS_Z':     phot['mw_transmission_z'],
    }

    # Write to FITS
    outfile = os.path.join(out_dir, f"spectrum_targetid_{targetid}.fits")
    make_spectrum_fits(outfile,
                        rec['wavelength'], rec['flux'], rec['ivar'], rec['wave_sigma'],
                        header_meta)
    print(f"Downloaded and saved spectrum for target {targetid} -> {outfile}")
    downloaded.add(targetid)
