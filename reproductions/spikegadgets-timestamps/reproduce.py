"""Reproduce Neo #1856 with tiny generated files, without downloading recordings.

With Neo, NumPy and quantities installed, run:
    python reproduce.py

Checked against Neo commit d50b9b9553680fdbc99e8441eda0ae01d7c292c8.
The script writes five tiny generated .rec files under ./synthetic beside itself
and prints JSON containing the observed RawIO and NeoIO timing results.

The elapsed tick oracle is specified by these synthetic inputs. It does not infer
that arbitrary backward acquisition-clock jumps represent genuine uint32 wraps.
"""

import json
from pathlib import Path
import struct
import warnings

import neo
import numpy as np
from neo.io.spikegadgetsio import SpikeGadgetsIO
from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO


SAMPLE_RATE = 30_000
MODULUS = 2**32
HEADER = b'''<Configuration>
<GlobalConfiguration systemTimeAtCreation="0" />
<HardwareConfiguration samplingRate="30000" numChannels="1" />
<SpikeConfiguration device="intan" chanPerChip="1">
  <SpikeNTrode id="1" spikeScalingToUv="1">
    <SpikeChannel hwChan="0" />
  </SpikeNTrode>
</SpikeConfiguration>
</Configuration>
'''


def write_rec(path, ticks):
    """Write one int16 channel and the format's uint32 tick in each packet."""
    with path.open("wb") as output:
        output.write(HEADER)
        for index, tick in enumerate(ticks):
            output.write(struct.pack("<BIh", 0x55, tick, 10 + index))


def inspect_case(path, ticks, elapsed_ticks):
    """Compare actual RawIO output with the synthetic acquisition-time oracle."""
    write_rec(path, ticks)
    reader = SpikeGadgetsRawIO(filename=str(path))
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        reader.parse_header()
    samples = reader.get_analogsignal_chunk(block_index=0, seg_index=0, stream_index=0)
    assert samples[:, 0].tolist() == [10, 11, 12, 13]
    assert reader.segment_count(0) == 1
    assert reader.get_signal_size(0, 0, 0) == 4
    nominal_start = reader.get_signal_t_start(0, 0, 0)
    nominal_stop = reader.segment_t_stop(0, 0)
    nominal_times = nominal_start + np.arange(4) / SAMPLE_RATE
    assert nominal_start == 0
    assert nominal_stop == 4 / SAMPLE_RATE
    signal = SpikeGadgetsIO(filename=str(path)).read_block().segments[0].analogsignals[0]
    np.testing.assert_array_equal(signal.times.rescale("s").magnitude, nominal_times)
    assert signal.t_stop.rescale("s").item() == nominal_stop
    raw_ticks = np.ndarray(
        shape=(4,),
        dtype="<u4",
        buffer=reader._raw_memmap,
        offset=reader._timestamp_byte,
        strides=(reader._raw_memmap.shape[1],),
    )
    assert raw_ticks.tolist() == ticks
    steps = [(int(b) - int(a)) % MODULUS for a, b in zip(ticks[:-1], ticks[1:])]
    result = {
        "case": path.stem,
        "file_bytes": path.stat().st_size,
        "ticks_written_and_read": raw_ticks.tolist(),
        "tick_steps_modulo_uint32": steps,
        "parse_succeeded": True,
        "warnings": [str(item.message) for item in observed],
        "segments": reader.segment_count(0),
        "samples": samples[:, 0].tolist(),
        "nominal_relative_times_us": (nominal_times * 1e6).tolist(),
        "nominal_exclusive_t_stop_us": nominal_stop * 1e6,
        "neoio_analogsignal_times_match_nominal": True,
    }
    if elapsed_ticks is not None:
        reference_times = np.array(elapsed_ticks, dtype=float) / SAMPLE_RATE
        result["known_acquisition_relative_times_us"] = (reference_times * 1e6).tolist()
        result["sample_time_error_us"] = ((nominal_times - reference_times) * 1e6).tolist()
        result["known_exclusive_t_stop_us"] = (elapsed_ticks[-1] + 1) / SAMPLE_RATE * 1e6
    else:
        result["note"] = "Counter reset: elapsed time is unknown; do not interpret it as a 2**32-tick data gap."
    return result


def main():
    fixture_dir = Path(__file__).resolve().parent / "synthetic"
    fixture_dir.mkdir(exist_ok=True)
    cases = [
        ("normal", [100, 101, 102, 103], [0, 1, 2, 3]),
        ("one_missing_tick", [100, 101, 103, 104], [0, 1, 3, 4]),
        ("uint32_rollover", [MODULUS - 2, MODULUS - 1, 0, 1], [0, 1, 2, 3]),
        ("missing_tick_at_rollover", [MODULUS - 2, 0, 1, 2], [0, 2, 3, 4]),
        ("clock_reset", [100, 101, 0, 1], None),
    ]
    results = [inspect_case(fixture_dir / f"{name}.rec", ticks, elapsed) for name, ticks, elapsed in cases]
    print(json.dumps({"neo": neo.__version__, "numpy": np.__version__, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
