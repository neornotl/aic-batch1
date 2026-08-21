"""A review-only range ZIP reader with correct local-header offsets."""

from __future__ import annotations

import struct
from pathlib import Path

from aic_pipeline.range_zip import RangeZip as BaseRangeZip
from aic_pipeline.range_zip import RemoteEntry


class RangeZip(BaseRangeZip):
    """Keep evidence-download fixes separate from the retrieval pipeline."""

    def _read_entries(self) -> dict[str, RemoteEntry]:
        """Read ZIP64 central-directory offsets when video archives exceed 4 GiB.

        The shared reader is sufficient for the keyframe archives.  Video
        archives can use ``0xffffffff`` in the ordinary central-directory
        offset, however, with the actual 64-bit value stored in extra field 1.
        Keeping this override here prevents changing the existing pipeline.
        """
        tail_start = max(0, self.size - 65536)
        tail = self._request(tail_start, self.size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        zip64 = tail.rfind(b"PK\x06\x06")
        if zip64 >= 0:
            _, _, _, _, _, _, count_disk, count, directory_size, directory_offset = struct.unpack_from(
                "<4sQ2H2I4Q", tail, zip64
            )
        elif eocd >= 0:
            _, _, _, _, count, directory_size, directory_offset, _ = struct.unpack_from(
                "<4s4H2LH", tail, eocd
            )
        else:
            raise RuntimeError("ZIP end record not found")
        directory = self._request(directory_offset, directory_offset + directory_size - 1)
        entries: dict[str, RemoteEntry] = {}
        cursor = 0
        while cursor + 46 <= len(directory) and directory[cursor:cursor + 4] == b"PK\x01\x02":
            header = struct.unpack_from("<4s6H3I5H2I", directory, cursor)
            compression = header[4]
            compressed_size, uncompressed_size = header[8], header[9]
            filename_size, extra_size, comment_size = header[10], header[11], header[12]
            local_offset = header[16]
            name_start = cursor + 46
            name = directory[name_start:name_start + filename_size].decode("utf-8")
            extra_start = name_start + filename_size
            extra = directory[extra_start:extra_start + extra_size]
            # ZIP64 field 0x0001 stores only values whose 32-bit counterpart
            # was saturated, in the documented uncompressed/compressed/offset
            # order.
            extra_cursor = 0
            while extra_cursor + 4 <= len(extra):
                field_id, field_size = struct.unpack_from("<HH", extra, extra_cursor)
                value = extra[extra_cursor + 4:extra_cursor + 4 + field_size]
                extra_cursor += 4 + field_size
                if field_id != 1:
                    continue
                value_cursor = 0
                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = struct.unpack_from("<Q", value, value_cursor)[0]
                    value_cursor += 8
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = struct.unpack_from("<Q", value, value_cursor)[0]
                    value_cursor += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack_from("<Q", value, value_cursor)[0]
                break
            entries[name] = RemoteEntry(
                name, compressed_size, uncompressed_size, local_offset,
                filename_size, extra_size, compression,
            )
            cursor += 46 + filename_size + extra_size + comment_size
        if len(entries) != count:
            raise RuntimeError(f"central directory truncated: expected {count}, got {len(entries)}")
        return entries

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
