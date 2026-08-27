"""Read and download stored ZIP entries through HTTP byte ranges."""

from __future__ import annotations

import io
import struct
import urllib.request
import zipfile
import concurrent.futures
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    local_offset: int
    filename_size: int
    extra_size: int
    compression: int


class RangeZip:
    def __init__(self, url: str, block_size: int = 65536, timeout: int = 120):
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self.size = self._content_length()
        self.entries = self._read_entries()

    def _request(self, start: int, end: int, retries: int = 4) -> bytes:
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Mozilla/5.0 AIC26-range-client",
            "Accept": "*/*",
        })
        expected = end - start + 1
        last_error = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read()
                    content_range = response.headers.get("Content-Range", "")
                if len(data) != expected or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"invalid range response: {content_range!r}, {len(data)} bytes")
                return data
            except Exception as error:
                last_error = error
                if attempt + 1 < retries:
                    import time
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"range {start}-{end} failed") from last_error

    def _content_length(self, retries: int = 4) -> int:
        """Return archive size, tolerating transient BTC 5xx responses."""
        last_error = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(self.url, method="HEAD", headers={
                    "User-Agent": "Mozilla/5.0 AIC26-range-client",
                    "Accept": "*/*",
                })
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    length = response.headers.get("Content-Length")
                    ranges = response.headers.get("Accept-Ranges", "")
                if not length or "bytes" not in ranges.lower():
                    raise RuntimeError("remote archive does not advertise byte ranges")
                return int(length)
            except Exception as error:
                last_error = error
                if attempt + 1 < retries:
                    import time
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"could not read remote archive size: {self.url}") from last_error

    def _read_entries(self) -> dict[str, RemoteEntry]:
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
            offset = header[16]
            name_start = cursor + 46
            name = directory[name_start:name_start + filename_size].decode("utf-8")
            extra_start = name_start + filename_size
            extra = directory[extra_start:extra_start + extra_size]
            # ZIP64 archives saturate 32-bit size/offset fields at 0xffffffff
            # and store the real values in extra field 0x0001.  Several BTC
            # video archives cross the 4 GiB boundary after the early videos.
            extra_cursor = 0
            while extra_cursor + 4 <= len(extra):
                field_id, field_size = struct.unpack_from("<HH", extra, extra_cursor)
                value = extra[extra_cursor + 4:extra_cursor + 4 + field_size]
                extra_cursor += 4 + field_size
                if field_id != 1:
                    continue
                value_cursor = 0
                if uncompressed_size == 0xFFFFFFFF:
                    if value_cursor + 8 > len(value):
                        raise RuntimeError(f"truncated ZIP64 extra field for {name}")
                    uncompressed_size = struct.unpack_from("<Q", value, value_cursor)[0]
                    value_cursor += 8
                if compressed_size == 0xFFFFFFFF:
                    if value_cursor + 8 > len(value):
                        raise RuntimeError(f"truncated ZIP64 extra field for {name}")
                    compressed_size = struct.unpack_from("<Q", value, value_cursor)[0]
                    value_cursor += 8
                if offset == 0xFFFFFFFF:
                    if value_cursor + 8 > len(value):
                        raise RuntimeError(f"truncated ZIP64 extra field for {name}")
                    offset = struct.unpack_from("<Q", value, value_cursor)[0]
                break
            entries[name] = RemoteEntry(
                name, compressed_size, uncompressed_size, offset,
                filename_size, extra_size, compression,
            )
            cursor += 46 + filename_size + extra_size + comment_size
        if len(entries) != count:
            raise RuntimeError(f"central directory truncated: expected {count}, got {len(entries)}")
        return entries

    def find(self, name: str) -> RemoteEntry:
        candidates = [name, f"video/{name}", f"videos/{name}", f"video/{name}.mp4"]
        for candidate in candidates:
            if candidate in self.entries:
                return self.entries[candidate]
        stem = Path(name).stem
        for entry in self.entries.values():
            if Path(entry.name).stem == stem:
                return entry
        raise KeyError(name)

    def _data_start(self, entry: RemoteEntry) -> int:
        """Return the payload offset from the entry's *local* ZIP header.

        A ZIP central-directory extra field is allowed to differ from the
        local-header extra field.  The BTC archives use that legal variation
        (the central header is four bytes shorter), so using the central
        ``extra_size`` shifts every payload and produces corrupt MP4 files.
        Read the local header before extracting instead of assuming both
        headers have identical lengths.
        """
        header = self._request(entry.local_offset, entry.local_offset + 29)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"local ZIP header not found for {entry.name}")
        _, _, _, _, _, _, _, _, _, filename_size, extra_size = struct.unpack(
            "<4s5H3I2H", header
        )
        if filename_size != entry.filename_size:
            raise RuntimeError(
                f"local filename size mismatch for {entry.name}: "
                f"{filename_size} != {entry.filename_size}"
            )
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

    def download_parallel(self, name: str, output: Path, workers: int = 4,
                          chunk_size: int = 4 * 1024 * 1024,
                          sha256: str | None = None) -> RemoteEntry:
        """Download independent stored ZIP ranges concurrently and verify them."""
        entry = self.find(name)
        if entry.compression != 0:
            raise RuntimeError(f"entry is compressed; range extraction cannot stream {entry.name}")
        data_start = self._data_start(entry)
        data_end = data_start + entry.compressed_size - 1
        ranges = [(start, min(start + chunk_size - 1, data_end))
                  for start in range(data_start, data_end + 1, chunk_size)]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            handle.truncate(entry.uncompressed_size)

        def fetch(item: tuple[int, int]) -> tuple[int, bytes]:
            return item[0], self._request(*item)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, item) for item in ranges]
            with output.open("r+b") as handle:
                for future in concurrent.futures.as_completed(futures):
                    start, data = future.result()
                    handle.seek(start - data_start)
                    handle.write(data)
        if output.stat().st_size != entry.uncompressed_size:
            output.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch for {entry.name}")
        if sha256:
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            if digest.lower() != sha256.lower():
                output.unlink(missing_ok=True)
                raise RuntimeError(f"sha256 mismatch for {entry.name}")
        return entry
