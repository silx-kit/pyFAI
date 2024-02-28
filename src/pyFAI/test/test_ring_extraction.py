"""Test suite for RingExtraction class."""

# from unittest.mock import MagicMock

import numpy as np
import pytest

from .. import ring_extraction
from ..ring_extraction import RingExtraction


@pytest.fixture
def ring_extraction_instance(mocker):
    single_geometry = mocker.MagicMock()
    massif = mocker.MagicMock()
    ring_extraction_inst = RingExtraction(single_geometry, massif)
    two_theta_values = np.array([1, 2, 3, 4, 5])
    ring_extraction_inst.two_theta_values = two_theta_values
    return ring_extraction_inst


@pytest.fixture
def mock_numpy(mocker):
    return mocker.patch(ring_extraction.__name__ + ".np")


ring_extraction_attributes = [
    "single_geometry",
    "image",
    "detector",
    "calibrant",
    "massif",
    "two_theta_array",
    "two_theta_values",
]


@pytest.mark.parametrize("attribute_name", ring_extraction_attributes)
def test_calib(attribute_name, ring_extraction_instance):
    assert hasattr(ring_extraction_instance, attribute_name)


def test_extract_control_points(mocker, ring_extraction_instance):
    # arrange
    mock_control_points = mocker.patch(ring_extraction.__name__ + ".ControlPoints")
    mock_extract_peaks_in_one_ring = mocker.patch.object(
        ring_extraction_instance, "extract_list_of_peaks_in_one_ring"
    )

    # act
    ring_extraction_instance.extract_control_points(max_number_of_rings=3)

    # assert
    mock_control_points.assert_called_once_with(
        calibrant=ring_extraction_instance.calibrant
    )

    assert mock_extract_peaks_in_one_ring.call_count == 3
    assert mock_extract_peaks_in_one_ring.call_args_list[0][0] == (
        0,
        1,
    )
    assert mock_extract_peaks_in_one_ring.call_args_list[1][0] == (
        1,
        1,
    )
    assert mock_extract_peaks_in_one_ring.call_args_list[2][0] == (
        2,
        1,
    )


def test_extract_list_of_peaks_in_one_ring(mocker, ring_extraction_instance):
    # arrange
    ones_array = np.ones((5, 5))

    ring_extraction_instance.two_theta_array = ones_array
    mock_marching_squares = mocker.patch(
        ring_extraction.__name__ + ".marchingsquares.MarchingSquaresMergeImpl"
    )
    mock_create_mask_around_ring = mocker.patch.object(
        ring_extraction_instance,
        "_create_mask_around_ring",
        return_value=ones_array,
    )
    mock_remove_low_intensity_pixels_from_mask = mocker.patch.object(
        ring_extraction_instance,
        "_remove_low_intensity_pixels_from_mask",
        return_value=(ones_array, 0),
    )
    mock_calculate_num_of_points_to_keep = mocker.patch.object(
        ring_extraction_instance,
        "_calculate_num_of_points_to_keep",
        return_value=5,
    )

    ring_extraction_instance.image = ones_array
    ring_extraction_instance.detector.mask = "detector_mask_string"

    # act
    ring_extraction_instance.extract_list_of_peaks_in_one_ring(ring_index=1)

    # assert
    mock_marching_squares.assert_called_once()
    assert np.array_equal(mock_marching_squares.call_args_list[0][0][0], ones_array)
    assert mock_marching_squares.call_args_list[0][1]["mask"] == "detector_mask_string"
    assert mock_marching_squares.call_args_list[0][1]["use_minmax_cache"]
    mock_create_mask_around_ring.assert_called_once()
    assert mock_create_mask_around_ring.call_args_list[0][0][0] == 1

    mock_remove_low_intensity_pixels_from_mask.assert_called_once()
    assert np.array_equal(
        mock_remove_low_intensity_pixels_from_mask.call_args_list[0][0][0], ones_array
    )

    mock_calculate_num_of_points_to_keep.assert_called_once()

    ring_extraction_instance.massif.peaks_from_area.assert_called_once()
    assert np.array_equal(
        ring_extraction_instance.massif.peaks_from_area.call_args_list[0][0][0],
        ones_array,
    )
    assert (
        ring_extraction_instance.massif.peaks_from_area.call_args_list[0][1]["keep"]
        == 5
    )
    assert (
        ring_extraction_instance.massif.peaks_from_area.call_args_list[0][1]["dmin"]
        == 0
    )
    assert (
        ring_extraction_instance.massif.peaks_from_area.call_args_list[0][1]["seed"]
        == set()
    )
    assert (
        ring_extraction_instance.massif.peaks_from_area.call_args_list[0][1]["ring"]
        == 1
    )
