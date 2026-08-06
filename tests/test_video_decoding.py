from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.convert_selected_lerobot_to_rlds import _decode_rgb_frames, _read_video

cv2 = pytest.importorskip("cv2")
av = pytest.importorskip("av")


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A small deterministic clip. Solid colours survive lossy encoding, so the
    two decoders can be compared exactly."""
    path = tmp_path / "clip.mp4"
    size = 32
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (size, size))
    for index in range(8):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :, index % 3] = 255  # BGR, written as-is
        writer.write(frame)
    writer.release()
    assert path.is_file()
    return path


def _decode_with_opencv_only(path: Path, monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    # `import av` raises ImportError when sys.modules holds None for the name.
    monkeypatch.setitem(sys.modules, "av", None)
    return _decode_rgb_frames(path)


def test_both_decoders_agree_frame_for_frame(
    sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyAV replaced OpenCV as the primary decoder because OpenCV cannot handle
    the AV1 files. The swap must not change the pixels on files OpenCV can read,
    or converted RLDS would stop matching the source."""
    with monkeypatch.context() as patched:
        opencv_frames = _decode_with_opencv_only(sample_video, patched)
    pyav_frames = _decode_rgb_frames(sample_video)

    assert len(opencv_frames) == len(pyav_frames) == 8
    for index, (from_opencv, from_pyav) in enumerate(zip(opencv_frames, pyav_frames)):
        np.testing.assert_array_equal(from_opencv, from_pyav, err_msg=f"frame {index}")


def test_frames_are_rgb_uint8(sample_video: Path) -> None:
    frames = _decode_rgb_frames(sample_video)

    assert frames[0].dtype == np.uint8
    assert frames[0].shape == (32, 32, 3)
    # Frame 0 was written with the blue channel set in BGR, so RGB output puts
    # the peak in the last channel.
    assert frames[0][0, 0].argmax() == 2


def test_a_corrupt_video_fails_loudly(tmp_path: Path) -> None:
    """PyAV raises instead of returning nothing, unlike OpenCV on a missing codec."""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(Exception, match="broken.mp4"):
        _read_video(broken, image_size=32, rotate_180=False)


def test_zero_frames_names_the_missing_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenCV fallback returns zero frames when it lacks the codec, which is
    how the AV1 problem first appeared. Say what to install rather than only that
    the file looked empty."""
    monkeypatch.setattr(
        "src.data.convert_selected_lerobot_to_rlds._decode_rgb_frames", lambda path: []
    )

    with pytest.raises(ValueError, match="Install `av`"):
        _read_video(tmp_path / "silent.mp4", image_size=32, rotate_180=False)


def test_read_video_resizes_and_rotates(sample_video: Path) -> None:
    frames = _read_video(sample_video, image_size=16, rotate_180=False)
    rotated = _read_video(sample_video, image_size=16, rotate_180=True)

    assert frames.shape == (8, 16, 16, 3)
    assert rotated.shape == (8, 16, 16, 3)
    np.testing.assert_array_equal(rotated[0], np.rot90(frames[0], 2))
