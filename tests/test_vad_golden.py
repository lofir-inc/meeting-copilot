import numpy as np

from src.audio.vad import ChunkBuilder, VadConfig


def test_rms_chunk_boundaries_are_fixed_for_interview_mode():
    sample_rate = 10_000
    config = VadConfig(
        silence_threshold_db=-72,
        silence_duration_sec=1.0,
        min_chunk_sec=0.1,
        max_chunk_sec=10.0,
    )
    time = np.arange(2 * sample_rate) / sample_rate
    first_voice = (0.1 * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)
    time = np.arange(3 * sample_rate) / sample_rate
    second_voice = (0.1 * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)
    waveform = np.concatenate((first_voice, np.zeros(sample_rate), second_voice, np.zeros(2 * sample_rate))).astype(np.float32)

    builder = ChunkBuilder("guest", config, sample_rate)
    chunks = []
    for offset in range(0, waveform.size, sample_rate // 10):
        chunks.extend(builder.feed(waveform[offset:offset + sample_rate // 10]))

    assert [(chunk.start_time, chunk.end_time) for chunk in chunks] == [(0.0, 3.0), (3.0, 7.0)]
