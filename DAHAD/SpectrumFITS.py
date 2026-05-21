import re
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

@dataclass
class SpectrumData:
    wave: Optional[np.ndarray]
    flux: Optional[np.ndarray]
    ivar: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    wave_sigma: Optional[np.ndarray] = None
    flux_unit: str = ""
    snr_regions: dict = field(default_factory=dict)
    snr: Optional[np.ndarray] = None

@dataclass
class SpectrumFITS:
    spectrum: SpectrumData
    header0: fits.Header
    header1: fits.Header
    instrument: str = "UNKNOWN"
    MJD: Optional[float] = None
    EXPTIME: Optional[float] = None
    OBJECT: str = "UNKNOWN"
    @classmethod
    def from_file(
        cls,
        filename: str,
        ext: int = 1,
        instrument: Optional[str] = None,
    ):
        with fits.open(filename) as hdul:
            header0 = hdul[0].header.copy()
            header1 = hdul[ext].header.copy()
            object_name = header0.get("OBJECT", "UNKNOWN")
            exptime = header0.get("EXPTIME", np.nan)
            #cls.OBJECT = header0.get("OBJECT", "UNKNOWN")
            #cls.EXPTIME = header0["EXPTIME"]
            if instrument is None:
                instrument = cls._detect_instrument(header0, header1)
            else:
                instrument = instrument.upper()

            if instrument == "SDSS":
                spectrum,MJD = cls._read_sdss(hdul, ext)
            elif instrument == "DESI":
                spectrum,MJD  = cls._read_desi(hdul, ext)
            elif instrument == "4MOST":
                spectrum,MJD = cls._read_4most(hdul, ext)
            else:
                spectrum = cls._read_default(hdul, ext)
                MJD = 0
            
            spectrum.snr_regions = cls._compute_snr_regions(
                wave=spectrum.wave,
                flux=spectrum.flux,
                sigma=spectrum.sigma,
                mask=spectrum.mask,
            )
            spectrum.snr = cls._compute_snr_main(
                flux=spectrum.flux,
                sigma=spectrum.sigma,
            )
            return cls(
                spectrum=spectrum,
                header0=header0,
                header1=header1,
                instrument=instrument,
                MJD=MJD,
                OBJECT=object_name,
                EXPTIME=exptime,
            )

    @staticmethod
    def _get_header_value(header: fits.Header, key: str) -> str:
        value = header.get(key)
        return "" if value is None else str(value).strip().upper()

    @classmethod
    def _detect_instrument(cls, header0: fits.Header, header1: fits.Header) -> str:
        keys = ["TELESCOP", "INSTRUME", "HIERARCH INSTRUMENT", "ORIGIN", "EXTNAME"]
        values = []

        for header in (header0, header1):
            for key in keys:
                values.append(cls._get_header_value(header, key))

        for value in values:
            if "SDSS" in value:
                return "SDSS"
            if "4MOST" in value or "ESO" in value:
                return "4MOST"
            if "DESI" in value:
                return "DESI"

        return "UNKNOWN"

    @staticmethod
    def _bunit_scale(unit_string: str) -> float:
        if not unit_string:
            return 1.0

        match = re.search(r'([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)', str(unit_string))
        if match:
            return float(match.group(1))
        return 1.0

    @staticmethod
    def _ivar_to_sigma(ivar: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if ivar is None:
            return None

        ivar = np.asarray(ivar, dtype=float)
        sigma = np.full_like(ivar, np.nan, dtype=float)

        good = ivar > 0
        sigma[good] = 1.0 / np.sqrt(ivar[good])

        return sigma

    @staticmethod
    def _compute_snr(
        wave: np.ndarray,
        flux: np.ndarray,
        sigma: np.ndarray,
        wmin: float,
        wmax: float,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        region = (wave >= wmin) & (wave <= wmax)

        valid = region.copy()
        valid &= np.isfinite(wave)
        valid &= np.isfinite(flux)
        valid &= np.isfinite(sigma)
        valid &= sigma > 0

        if mask is not None:
            # assume mask == 0 means good pixel
            valid &= (mask == 0)

        if np.sum(valid) == 0:
            return np.nan

        sn = flux[valid] / sigma[valid]
        return np.nanmedian(sn)
    @staticmethod
    
    def _compute_snr_main(
        flux: np.ndarray,
        sigma: np.ndarray,
    ) -> float:
        
        valid = np.ones_like(flux).astype(bool)
        valid &= sigma > 0
        
        sn = flux[valid] / sigma[valid]
        return np.nanmedian(sn)
    
    @classmethod
    def _compute_snr_regions(
        cls,
        wave: Optional[np.ndarray],
        flux: Optional[np.ndarray],
        sigma: Optional[np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> dict:
        if wave is None or flux is None or sigma is None:
            return {}

        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        regions = {

            "snr_3800_4200": (3800.0, 4200.0),
            "snr_5000_6000": (5000.0, 6000.0),
            "snr_8000_9000": (8000.0, 9000.0),
            "telluric_B_6860_6930": (6860.0, 6930.0),
            "telluric_A_7590_7700": (7590.0, 7700.0),
            "telluric_H2O_7150_7400": (7150.0, 7400.0),
            "telluric_H2O_8100_8400": (8100.0, 8400.0),
        }

        return {
            name: cls._compute_snr(wave, flux, sigma, wmin, wmax, mask=None)
            for name, (wmin, wmax) in regions.items()
        }

    @classmethod
    def _read_sdss(cls, hdul, ext: int) -> SpectrumData:
        data = hdul[ext].data
        names = list(data.names) if hasattr(data, "names") else []
        cls.MJD = hdul[0].header["MJD"]
        wave = 10 ** np.asarray(data["loglam"], dtype=float) if "loglam" in names else None
        flux = np.asarray(data["flux"], dtype=float) if "flux" in names else None
        ivar = np.asarray(data["ivar"], dtype=float) if "ivar" in names else None
        mask = np.asarray(data["and_mask"]) if "and_mask" in names else None

        flux_unit = hdul[0].header.get("BUNIT", "")
        scale = cls._bunit_scale(flux_unit)

        if flux is not None:
            flux = flux * scale

        sigma = cls._ivar_to_sigma(ivar)
        if sigma is not None:
            sigma = sigma * scale

        if ivar is not None and scale != 0:
            ivar = ivar / (scale ** 2)

        return SpectrumData(
            wave=wave,
            flux=flux,
            ivar=ivar,
            sigma=sigma,
            mask=mask,
            flux_unit=flux_unit,),MJD

    @classmethod
    def _read_desi(cls, hdul, ext: int) -> SpectrumData:
        data = hdul[ext].data
        header = hdul[ext].header
        names = list(data.names) if hasattr(data, "names") else []
        MJD = Time(hdul[0].header["HIERARCH dateobs_center"].replace("+00", ""), format="iso", scale="utc").mjd
        
        wave = np.asarray(data["WAVELENGTH"], dtype=float) if "WAVELENGTH" in names else None
        flux = np.asarray(data["FLUX"], dtype=float) if "FLUX" in names else None
        ivar = np.asarray(data["IVAR"], dtype=float) if "IVAR" in names else None
        wave_sigma = np.asarray(data["WAVE_SIGMA"], dtype=float) if "WAVE_SIGMA" in names else None

        flux_unit = header.get("TUNIT2", "")
        scale = cls._bunit_scale(flux_unit)

        if flux is not None:
            flux = flux * scale

        sigma = cls._ivar_to_sigma(ivar)
        if sigma is not None:
            sigma = sigma * scale

        if ivar is not None and scale != 0:
            ivar = ivar / (scale ** 2)

        return SpectrumData(
            wave=wave,
            flux=flux,
            ivar=ivar,
            sigma=sigma,
            wave_sigma=wave_sigma,
            flux_unit='erg/cm^2/s/Ang',),MJD

    @classmethod
    def _read_4most(cls, hdul, ext: int) -> SpectrumData:
        data = hdul[ext].data
        row = data[0]
        MJD = hdul[0].header["MJD-OBS"]
        #print(hdul[0].header["MJD-OBS"])
        wave = np.asarray(row["WAVE"], dtype=float)
        flux = np.asarray(row["FLUX"], dtype=float)
        sigma = np.asarray(row["ERR"], dtype=float)
        mask = np.asarray(row["QUAL"])
        flux_unit = "erg/s/cm^2/Ang"

        return SpectrumData(
            wave=wave,
            flux=flux,
            sigma=sigma,
            mask=mask,
            flux_unit=flux_unit,),MJD

    @staticmethod
    def _read_default(hdul, ext: int) -> SpectrumData:
        data = hdul[ext].data
        return SpectrumData(
            wave=None,
            flux=np.asarray(data, dtype=float),
        )

    @property
    def wcs(self):
        return WCS(self.header1)