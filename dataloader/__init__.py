from .wsi_utils import use_local_vips_linux, tune_vips_cache

VIPS_PREFIX = "libvips/bin"
OPENSLIDE_PREFIX = "openslide/bin"
DICOM_PREFIX = "libdicom/bin"
use_local_vips_linux(VIPS_PREFIX, OPENSLIDE_PREFIX, DICOM_PREFIX)
