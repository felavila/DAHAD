from pathlib import Path
from typing import Sequence, Tuple
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Observations

#to utils
def _to_coord(ra, dec) -> SkyCoord:
	"""
	Convert RA/Dec in degrees to SkyCoord.
	"""
	return SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")

def query_mast_missions(
	ra,
	dec,
	radius_arcsec: float = 5.0,
	missions: Sequence[str] = ("HST", "JWST"),
) -> Tuple[pd.DataFrame, dict]:
	"""
	Query MAST for observations from selected missions.

	Parameters
	----------
	ra, dec : float
		Coordinates in degrees.

	radius_arcsec : float, optional
		Search radius in arcsec.

	missions : sequence of str, optional
		MAST obs_collection names, e.g. HST, JWST.

	Returns
	-------
	summary : pandas.DataFrame
		One row per mission.

	raw_results : dict
		Raw MAST observation tables.
	"""
	coord = _to_coord(ra, dec)

	rows = []
	raw_results = {}

	for mission in missions:
		try:
			obs = Observations.query_criteria(
				coordinates=coord,
				radius=radius_arcsec * u.arcsec,
				obs_collection=mission,
			)

			raw_results[mission] = obs
			n_obs = len(obs)

			if n_obs > 0:
				instruments = sorted(set(map(str, obs["instrument_name"])))
				filters = sorted(set(map(str, obs["filters"])))
				data_products = sorted(set(map(str, obs["dataproduct_type"])))

				total_exptime = 0.0
				if "t_exptime" in obs.colnames:
					total_exptime = float(pd.Series(obs["t_exptime"]).fillna(0).sum())
			else:
				instruments = []
				filters = []
				data_products = []
				total_exptime = 0.0

			rows.append({
				"service": "MAST",
				"survey": mission,
				"catalog_id": "",
				"has_data": n_obs > 0,
				"n_matches": n_obs,
				"instruments": ", ".join(instruments),
				"filters": ", ".join(filters),
				"data_products": ", ".join(data_products),
				"total_exptime_s": total_exptime,
				"error": "",
			})

		except Exception as exc:
			raw_results[mission] = None
			rows.append({
				"service": "MAST",
				"survey": mission,
				"catalog_id": "",
				"has_data": None,
				"n_matches": None,
				"instruments": "",
				"filters": "",
				"data_products": "",
				"total_exptime_s": None,
				"error": str(exc),
			})

	return pd.DataFrame(rows), raw_results

def download_mast_results(
	raw_results: dict,
	download_dir: str = "",
	product_types: Sequence[str] = ("SCIENCE",),
	extension: Sequence[str] | None = None,
	mrp_only: bool = False,
) -> Tuple[pd.DataFrame, dict]:
	"""
	Download MAST products from raw query results.

	Parameters
	----------
	raw_results : dict
		Dictionary returned by query_mast_missions.

	download_dir : str, optional
		Directory where files will be downloaded.

	product_types : sequence of str, optional
		Product types to download. Usually ("SCIENCE",).

	extension : sequence of str or None, optional
		File extensions to keep, e.g. ("fits", "fits.gz").
		If None, no extension filtering is applied.

	mrp_only : bool, optional
		If True, download only minimum recommended products.

	Returns
	-------
	downloaded_table : pandas.DataFrame
		Table with downloaded file information.

	product_tables : dict
		Product tables for each mission.
	"""

	download_dir = Path(download_dir)
	download_dir.mkdir(parents=True, exist_ok=True)

	all_downloads = []
	product_tables = {}

	for mission, obs in raw_results.items():

		if obs is None or len(obs) == 0:
			print(f"{mission}: no observations to download.")
			product_tables[mission] = None
			continue

		print(f"{mission}: getting product list...")

		products = Observations.get_product_list(obs)

		if len(products) == 0:
			print(f"{mission}: no products found.")
			product_tables[mission] = products
			continue

		# Filter by product type, usually SCIENCE
		mask = pd.Series(products["productType"]).isin(product_types).to_numpy()

		# Optional: only minimum recommended products
		if mrp_only and "productGroupDescription" in products.colnames:
			mask &= pd.Series(products["productGroupDescription"]).astype(str).str.contains(
				"Minimum Recommended Products",
				case=False,
				na=False,
			).to_numpy()

		# Optional: filter by extension
		if extension is not None and "productFilename" in products.colnames:
			allowed_ext = tuple(extension)
			filenames = pd.Series(products["productFilename"]).astype(str)
			mask &= filenames.str.endswith(allowed_ext).to_numpy()

		selected_products = products[mask]
		product_tables[mission] = selected_products

		print(f"{mission}: {len(selected_products)} products selected.")

		if len(selected_products) == 0:
			continue

		manifest = Observations.download_products(
			selected_products,
			download_dir=str(download_dir),
		)

		all_downloads.append(manifest.to_pandas())

	if len(all_downloads) > 0:
		downloaded_table = pd.concat(all_downloads, ignore_index=True)
	else:
		downloaded_table = pd.DataFrame()

	return downloaded_table, product_tables