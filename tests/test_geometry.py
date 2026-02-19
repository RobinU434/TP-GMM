import numpy as np

from tpgmm.utils.geometry import (
    obtain_pick_and_place_translation,
    transform_into_frames,
)


class TestObtainPickAndPlaceTranslation:
    def test_extracts_first_and_last_frames(self):
        demos = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        frame_idx = [0, -1]
        result = obtain_pick_and_place_translation(demos, frame_idx)
        expected = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        np.testing.assert_array_equal(result, expected)


class TestTransformIntoFrames:
    def test_identity_rotation_with_translation(self):
        trajectories = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        )
        translations = np.array(
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
        )
        rotations = np.array(
            [
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            ]
        )
        result = transform_into_frames(trajectories, translations, rotations)
        expected = np.array(
            [
                [[[2, 3, 4], [5, 6, 7]], [[3, 4, 5], [6, 7, 8]]],
                [[[10, 11, 12], [13, 14, 15]], [[11, 12, 13], [14, 15, 16]]],
            ]
        )
        np.testing.assert_array_equal(result, expected)
