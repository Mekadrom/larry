import datetime
import email.utils
from pathlib import Path

import pytest
import requests

from larry.common.config.data.downloader_configs import MultiFileDownloaderConfig, SingleFileDownloaderConfig
from larry.common.data.downloaders import MultiFileDownloader, SingleFileDownloader
from larry.common.data.preprocessing.url_mappers import UrlMapper


def _downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes) -> SingleFileDownloader:
    config = SingleFileDownloaderConfig(
        download_base_url="https://example.invalid/",
        output_file_name="manifest.html",
        validate_content_contains="2024",
        max_retries=2,
    )
    downloader = SingleFileDownloader(config, tmp_path)

    def fake_download(url: str, file_path: Path, return_contents: bool = False) -> bytes | None:
        file_path.write_bytes(body)
        return body if return_contents else None

    monkeypatch.setattr(downloader, "_download_file", fake_download)
    return downloader


def test_content_that_never_validates_raises_instead_of_reporting_success(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = _downloader(tmp_path, monkeypatch, b"<html>2019</html>")

    with pytest.raises(ValueError):
        downloader.ensure_downloaded()

    assert not downloader.output_file_path.exists()


def test_validated_content_is_moved_into_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = _downloader(tmp_path, monkeypatch, b"<html>2024</html>")

    assert downloader.ensure_downloaded()

    assert downloader.output_file_path.read_bytes() == b"<html>2024</html>"


def test_retry_after_accepts_delay_seconds_and_http_dates() -> None:
    def response(retry_after: str | None) -> requests.Response:
        r = requests.Response()
        if retry_after is not None:
            r.headers["Retry-After"] = retry_after
        return r

    in_thirty = email.utils.format_datetime(
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=30), usegmt=True
    )
    long_ago = "Wed, 21 Oct 2015 07:28:00 GMT"

    assert UrlMapper._retry_after_seconds(response("120"), default=60.0) == 120.0
    assert 25.0 < UrlMapper._retry_after_seconds(response(in_thirty), default=60.0) <= 30.0
    assert UrlMapper._retry_after_seconds(response(long_ago), default=60.0) == 0.0
    assert UrlMapper._retry_after_seconds(response("soon"), default=60.0) == 60.0
    assert UrlMapper._retry_after_seconds(response(None), default=7.0) == 7.0


def _multi_downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MultiFileDownloader:
    def fake_download(self: SingleFileDownloader, url: str, file_path: Path, return_contents: bool = False) -> bytes:
        file_path.write_bytes(b"ok")
        return b"ok"

    monkeypatch.setattr(SingleFileDownloader, "_download_file", fake_download)
    config = MultiFileDownloaderConfig(configs=[
        ("SingleFileDownloader", SingleFileDownloaderConfig(
            download_base_url="https://example.invalid/", output_file_name=name
        ))
        for name in ("a.html", "b.html")
    ])
    return MultiFileDownloader(config, tmp_path)


def test_multi_file_downloader_reports_a_download_when_any_delegate_downloads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.html").write_bytes(b"cached")

    assert _multi_downloader(tmp_path, monkeypatch).ensure_downloaded()


def test_multi_file_downloader_reports_no_download_when_everything_is_cached(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.html").write_bytes(b"cached")
    (tmp_path / "b.html").write_bytes(b"cached")

    assert not _multi_downloader(tmp_path, monkeypatch).ensure_downloaded()


class _Raw:
    def __init__(self, encoded: bytes) -> None:
        self.encoded = encoded

    def stream(self, size: int, decode_content: bool = True) -> list[bytes]:
        assert decode_content is False
        return [self.encoded]


class _Response:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self.headers = headers
        self.raw = _Raw(body)
        self.body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        import gzip
        return [gzip.decompress(self.body) if "Content-Encoding" in self.headers else self.body]


@pytest.mark.parametrize(("name", "kept_compressed"), [("photos.csv.gz", True), ("page.html", False)])
def test_gzip_encoded_downloads_keep_their_bytes_only_for_gz_files(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, kept_compressed: bool,
) -> None:
    import gzip
    body = gzip.compress(b"photo_id\tlicense\n1\tCC0\n")
    response = _Response(body, {"Content-Encoding": "gzip", "Content-Length": str(len(body))})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    downloader = SingleFileDownloader(
        SingleFileDownloaderConfig(download_base_url="https://example.test", output_file_name=name), tmp_path
    )

    downloader.ensure_downloaded()

    written = (tmp_path / name).read_bytes()
    assert written == (body if kept_compressed else gzip.decompress(body))


def test_a_truncated_download_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    response = _Response(b"partial", {"Content-Length": "100"})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    downloader = SingleFileDownloader(
        SingleFileDownloaderConfig(download_base_url="https://example.test", output_file_name="f.bin"), tmp_path
    )

    with pytest.raises(ValueError, match="Could not download"):
        downloader.ensure_downloaded()
    assert not (tmp_path / "f.bin").exists()
