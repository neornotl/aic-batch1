"""A review-only range ZIP reader with correct local-header offsets."""

from __future__ import annotations

import struct
from pathlib import Path

from aic_pipeline.range_zip import RangeZip as BaseRangeZip
from aic_pipeline.range_zip import RemoteEntry


class RangeZip(BaseRangeZip):
    """Keep evidence-download fixes separate from the retrieval pipeline."""

    def _data_start(self, entry: RemoteEntry) -> int:
        header = self._request(entry.local_offset, entry.local_offset + 29)
        signature, _, _, compression, _, _, _, _, _, filename_size, extra_size = struct.unpack(
            "<4s5H3I2H", header
        )
        if signature != b"PK\x03\x04":
            raise RuntimeError(f"invalid local header for {entry.name}")
        if compression != entry.compression:
            raise RuntimeError(f"compression mismatch for {entry.name}")
        return entry.local_offset + 30 + filename_size + extra_size

    def download(self, name: str, output: Path) -> RemoteEntry:
        entry = self.find(name)
        if entry.compression != 0:
            raise RuntimeError(f"entry is compressed; range extraction cannot stream {entry.name}")
        data_start = self._data_start(entry)
        data_end = data_start + entry.compressed_size - 1
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            position = data_start
            while position <= data_end:
                end = min(position + self.block_size - 1, data_end)
                handle.write(self._request(position, end))
                position = end + 1
        if output.stat().st_size != entry.uncompressed_size:
            output.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch for {entry.name}")
        return entry
