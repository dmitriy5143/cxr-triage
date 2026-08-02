from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fluoro_mvp_backend.image_scoring import (
    image_to_chexfound_tensor,
    load_image_pixels,
    resize_pad_array,
)


def test_chexfound_tensor_uses_research_two_stage_resize():
    arr = np.linspace(0, 1, 317 * 701, dtype=np.float32).reshape(317, 701)
    tensor = image_to_chexfound_tensor(arr, 512, letterbox_size=1024).numpy()

    padded = resize_pad_array(arr, 1024)
    expected_image = Image.fromarray((padded * 255).astype(np.uint8), mode="L").convert("RGB")
    expected_image = expected_image.resize((512, 512), Image.Resampling.BICUBIC)
    expected = np.asarray(expected_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    lo = expected.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    hi = expected.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    expected = (expected - lo) / np.maximum(hi - lo, 1e-6)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    expected = (expected - mean) / std

    np.testing.assert_array_equal(tensor, expected)
    direct = resize_pad_array(arr, 512)
    assert not np.array_equal(padded[::2, ::2], direct)


def _write_dicom(path: Path, pixels: np.ndarray, *, photometric: str = "MONOCHROME2") -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.Modality = "DX"
    ds.ViewPosition = "PA"
    ds.RescaleSlope = 2.0
    ds.RescaleIntercept = -100.0
    ds.WindowCenter = 1000.0
    ds.WindowWidth = 1000.0
    ds.PixelData = pixels.astype("<u2").tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def test_load_12bit_dicom_applies_rescale_window_and_metadata(tmp_path):
    pixels = np.asarray([[0, 250, 500], [750, 1000, 1500]], dtype=np.uint16)
    path = tmp_path / "twelve_bit.dcm"
    _write_dicom(path, pixels)
    arr, metadata, source_type = load_image_pixels(path)

    expected = np.clip(pixels.astype(np.float32) * 2.0 - 100.0, 500.0, 1500.0)
    np.testing.assert_array_equal(arr, expected)
    assert source_type == "dicom"
    assert metadata["bits_allocated"] == 16
    assert metadata["bits_stored"] == 12
    assert metadata["pixel_representation"] == 0
    assert metadata["window_center"] == 1000.0
    assert metadata["window_width"] == 1000.0


def test_load_monochrome1_dicom_inverts_pixels(tmp_path):
    pixels = np.asarray([[100, 200], [300, 400]], dtype=np.uint16)
    path = tmp_path / "mono1.dcm"
    _write_dicom(path, pixels, photometric="MONOCHROME1")
    arr, metadata, _ = load_image_pixels(path)
    rescaled = pixels.astype(np.float32) * 2.0 - 100.0
    expected = np.clip(rescaled.max() - rescaled, 500.0, 1500.0)
    np.testing.assert_array_equal(arr, expected)
    assert metadata["photometric_interpretation"] == "MONOCHROME1"
