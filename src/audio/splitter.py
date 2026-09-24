"""ステレオ→モノラル チャンネル分離。"""

from __future__ import annotations

import numpy as np


def split_channels(stereo_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ステレオ配列 [frames, 2] を L/R モノラルに分離して返す。"""
    return stereo_data[:, 0].copy(), stereo_data[:, 1].copy()
