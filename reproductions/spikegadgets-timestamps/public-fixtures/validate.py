"""Bounded, unmodified public-fixture validation for Neo issue #1856.

No data are generated or edited. Raw ticks are decoded independently with
struct.unpack_from from original packet bytes, before comparing the PR getter.
Run with this directory's .venv/bin/python and PYTHONDONTWRITEBYTECODE=1.
"""

import argparse
import hashlib
import json
import platform
import struct
import sys
import tomllib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
BASE = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/"
LIMIT = 100_000_000
SOURCE_SHA = "92d169cb11fcf5759a148881b7dfd233ac2bedd2733aba90bc7cece2bbffa824"
FILES = [
    ("20210225_em8_minirec2_ac.rec", 8707791,
     "Eric Denovellis recording, reduced in size by Samuel Garcia"),
    ("W122_06_09_2019_1_fromSD.rec", 19271,
     "Jess Breda recording, reduced in size by Cody Baker"),
    ("SpikeGadgets_test_data_2xNpix1.0_20240318_173658.rec", 11012556,
     "Test recording provided by Gregory Knoll; biological/simulator origin unspecified"),
    ("neuropixels2_4shank/20260122_134412_merged_cropped_1min_NP2.rec", 509398,
     "Krishna Kasuba Neuropixels 2.0 recording; Heberto Mayorquin retained header + 300 packets"),
    ("msh_1024ch/msh_1024ch.rec", 719992,
     "Anonymous MSH 1024-channel recording; Heberto Mayorquin retained header + 30 packets and anonymised metadata"),
]


def download():
    """No retry loops; reject changed/unknown sizes before body download."""
    budget = 0
    metadata = []
    for name, expected, provenance in FILES:
        path = HERE / "data" / name
        url = BASE + "raw/master/spikegadgets/" + name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size != expected:
                raise ValueError(f"Unexpected existing size: {name}")
            metadata.append({"file": name, "cached": True, "bytes": expected})
            continue
        if budget + expected > LIMIT:
            raise ValueError("Total public-data download cap would be exceeded")
        with urllib.request.urlopen(url, timeout=30) as response:
            declared = int(response.headers.get("Content-Length", "-1"))
            if declared != expected:
                raise ValueError(f"Changed/unknown download size for {name}: {declared}")
            chunks = []
            count = 0
            while True:
                chunk = response.read(min(1_048_576, expected - count + 1))
                if not chunk:
                    break
                count += len(chunk)
                budget += len(chunk)
                if count > expected or budget > LIMIT:
                    raise ValueError("Download budget exceeded")
                chunks.append(chunk)
            if count != expected:
                raise ValueError(f"Truncated download for {name}: {count}")
            path.write_bytes(b"".join(chunks))
            metadata.append({"file": name, "cached": False, "bytes": count,
                             "final_url": response.url,
                             "last_modified": response.headers.get("Last-Modified"),
                             "etag": response.headers.get("ETag")})
        print(f"Downloaded {name}: {count} bytes", flush=True)
    (HERE / "download.json").write_text(json.dumps({
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "body_bytes_downloaded_this_run": budget,
        "cap_bytes": LIMIT, "files": metadata,
    }, indent=2) + "\n")


def inspect_file(name, expected, provenance, RawIO, np):
    path = HERE / "data" / name
    data = path.read_bytes()
    assert len(data) == expected
    # Match a complete XML closing line, without using reader-derived offsets.
    header_end = 0
    for line in data.splitlines(keepends=True):
        header_end += len(line)
        if b"</Configuration>" in line:
            break
    else:
        raise ValueError("Missing XML closing tag")
    root = ElementTree.fromstring(data[:header_end])
    hconf = root.find("HardwareConfiguration")
    sconf = root.find("SpikeConfiguration")
    n_declared = int(hconf.attrib["numChannels"])
    n_saved = sum(len(group) for group in sconf)
    assert n_saved <= n_declared
    device_bytes = sum(int(device.attrib.get("numBytes", 0)) for device in hconf)
    timestamp_offset = 1 + device_bytes
    packet_bytes = timestamp_offset + 4 + 2 * n_saved
    packets, leftover = divmod(len(data) - header_end, packet_bytes)
    ticks = [struct.unpack_from("<I", data, header_end + i * packet_bytes + timestamp_offset)[0]
             for i in range(packets)]
    markers = [data[header_end + i * packet_bytes] for i in range(packets)]
    counts = {"contiguous_increment_one": 0, "normal_uint32_rollovers": 0,
              "duplicate_tick": 0, "forward_jump": 0,
              "backward_jump_not_normal_rollover": 0}
    anomalies = []
    diff_histogram = {}
    for i, (left, right) in enumerate(zip(ticks, ticks[1:])):
        modulo_delta = (right - left) % 2**32
        diff_histogram[str(modulo_delta)] = diff_histogram.get(str(modulo_delta), 0) + 1
        if modulo_delta == 1:
            counts["contiguous_increment_one"] += 1
            counts["normal_uint32_rollovers"] += int(right < left)
            continue
        kind = ("duplicate_tick" if right == left else
                "forward_jump" if right > left else "backward_jump_not_normal_rollover")
        counts[kind] += 1
        if len(anomalies) < 20:
            anomalies.append({"left_packet_index": i, "left_tick": left,
                              "right_tick": right, "modulo_delta": modulo_delta,
                              "kind": kind})
    reader = RawIO(filename=str(path))
    reader.parse_header()
    got = reader._get_spikegadgets_timestamps()
    expected_ticks = np.asarray(ticks, dtype="<u4")
    equal = bool(np.array_equal(got, expected_ticks))
    assert equal
    assert got.ndim == 1 and got.dtype == np.dtype("<u4")
    assert not got.flags.writeable
    assert np.shares_memory(got, reader._raw_memmap)
    slices = [(None, None), (0, 0), (0, 1), (1, 4), (-3, None), (packets, packets + 3)]
    for start, stop in slices:
        part = reader._get_spikegadgets_timestamps(start, stop)
        assert np.array_equal(part, expected_ticks[start:stop])
        assert not part.flags.writeable
    assert reader._timestamp_byte == timestamp_offset
    assert reader._raw_memmap.shape == (packets, packet_bytes)
    streams = []
    for i, stream in enumerate(reader.header["signal_streams"]):
        size = int(reader.get_signal_size(block_index=0, seg_index=0, stream_index=i))
        assert size == packets
        streams.append({"id": str(stream["id"]), "packets": size,
                        "nominal_t_start_seconds": float(reader.get_signal_t_start(0, 0, i))})
    bad_markers = sum(marker != 0x55 for marker in markers)
    format_ok = leftover == 0 and bad_markers == 0 and packets > 1
    interpretation = (
        "Packet framing checks pass; transition counts describe stored acquisition ticks."
        if format_ok else
        "Incomplete trailing packet; complete prefix observations only. Excluded from complete-file timing evidence."
        if bad_markers == 0 else
        "Packet framing checks fail; decoded words must NOT be interpreted as acquisition gaps, resets or rollovers. Excluded from timing evidence."
    )
    return {
        "file": name, "source_url": BASE + "raw/master/spikegadgets/" + name,
        "provenance_url": BASE + "src/master/spikegadgets/README.md",
        "provenance": provenance,
        "data_origin": "Unmodified publicly supplied recording fixture, not generated by this validation. Acquisition conditions are only known to the extent documented by contributors.",
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "header_bytes": header_end, "device_bytes": device_bytes,
        "timestamp_offset_in_packet": timestamp_offset, "packet_bytes": packet_bytes,
        "packets": packets, "trailing_nonpacket_bytes": leftover,
        "non_0x55_start_markers": bad_markers,
        "payload_utf8_replacement_sequence_count": data[header_end:].count(bytes.fromhex("efbfbd")),
        "usable_as_complete_file_acquisition_timing_evidence": format_ok,
        "interpretation": interpretation,
        "sample_rate_hz": float(hconf.attrib["samplingRate"]),
        "retained_packet_duration_seconds": packets / float(hconf.attrib["samplingRate"]),
        "first_tick": ticks[0] if ticks else None, "last_tick": ticks[-1] if ticks else None,
        "transition_counts": counts, "modulo_delta_histogram": diff_histogram,
        "anomaly_examples_first_twenty": anomalies,
        "getter_matches_every_raw_tick": equal, "getter_readonly_shared_memory": True,
        "slice_checks_passed": len(slices), "streams": streams,
        "nominal_segment_t_stop_seconds": float(reader.segment_t_stop(0, 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--neo-source", type=Path, required=True)
    args = parser.parse_args()
    REPO = args.neo_source.resolve()
    if args.download:
        download()
    sys.path.insert(0, str(REPO))
    import neo
    import numpy as np
    import quantities
    from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO
    source = REPO / "neo/rawio/spikegadgetsrawio.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    result = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "quantities": quantities.__version__,
        "installed_neo_distribution_version_for_import_metadata": neo.__version__,
        "local_source_pyproject_version": tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"],
        "neo_import": "neo/__init__.py (checkout selected by --neo-source)",
        "getter_pr": "https://github.com/NeuralEnsemble/python-neo/pull/1949",
        "getter_pr_head": "cfe51171b0e24883e5eacc8f379567f52f00a410",
        "local_base_head": "d50b9b9553680fdbc99e8441eda0ae01d7c292c8",
        "getter_module_sha256": SOURCE_SHA,
        "getter_module_matches_published_head": True,
        "fixture_selection": "All five packet-bearing files in test_spikegadgetsrawio.py; excluded documented header-only missing-channel stub.",
        "data_cap_bytes": LIMIT,
        "fixture_total_bytes": sum(size for _, size, _ in FILES),
        "download_transfer_method": "urllib HTTP response.read() bytes -> Path.write_bytes(); no text decode/encode or connector content conversion",
        "synthesis_or_injection": False,
        "limitations": [
            "Fixtures are small retained excerpts, not the user's recordings; a negative result does not exclude gaps in other files.",
            "README provenance cannot independently establish biological rather than bench/simulator acquisition for every fixture.",
            "A backward jump other than a normal uint32 rollover is only a reset/backward-jump candidate; elapsed time cannot be inferred.",
            "Module bytes were compared with the published PR head before running; other Neo code is local base d50b9b9. No repository was edited.",
            "A PyPI Neo distribution supplies importlib package metadata only; sys.path loads executable Neo source from the stated local worktree.",
            "Results from payloads with incomplete packets or invalid start markers are explicitly excluded from complete-file acquisition timing evidence; their cause is not investigated here.",
            "Downloaded byte counts match HTTP Content-Length and GIN's rounded file-page sizes; no independent git-annex SHA256 was obtained, so no claim is made about the origin of a malformed HTTP payload.",
        ],
        "files": [inspect_file(*entry, SpikeGadgetsRawIO, np) for entry in FILES],
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for file in result["files"]:
        print(file["file"], "packets=", file["packets"], file["transition_counts"], flush=True)


if __name__ == "__main__":
    main()
