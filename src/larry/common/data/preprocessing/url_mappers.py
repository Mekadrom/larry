import datetime
import email.utils
import hashlib
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import requests
from datasets import Value
from datasets.features.features import FeatureType

from larry.common.config.data.preprocessing.mapper_configs import UrlBytesMapperConfig, UrlMapperConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import SingleColumnMapper

_SESSIONS: dict[str, requests.Session] = {}


class UrlMapper[O, C: UrlMapperConfig = UrlMapperConfig](SingleColumnMapper[str | None, O | None, C], ABC):
    """Superclass for all ``SingleColumnMapper`` implementations that download data from the web."""

    def __init__(self, provenance: Provenance, config: C) -> None:
        """Sets up a cache for downloads, separate from the huggingface cache and without versioning or TTL."""
        super().__init__(provenance, config)
        if self.config.download_cache_dir is not None:
            self.download_cache_dir = Path(self.config.download_cache_dir).expanduser().resolve()
            os.makedirs(self.download_cache_dir, exist_ok=True)
        else:
            self.download_cache_dir = None

    def preprocess_example(self, example: str | None) -> O | None:
        """For a given example (URL string), download the data at the given URL and store it using overridden logic.

        Has error handling to protect the overall pipeline from crapping out if a download fails, and supports both
        connection pooling/sharing via a ``Session`` and a configurable busy wait sleep timer for throttling scraping.
        """
        if example is None:
            return None

        url_hash = str(hashlib.sha256(example.encode('utf-8')).hexdigest())
        cache_file_path = self.download_cache_dir / url_hash if self.download_cache_dir is not None else None

        content = None
        if cache_file_path is not None and cache_file_path.exists() and os.path.getsize(cache_file_path) > 0:
            self.log.info(f"cache hit for url: {example} at {str(cache_file_path)}")
            content = cache_file_path.read_bytes()
            try:
                self.validate_content(example, content)
            except Exception:
                self.log.error(
                    f"Cached content at {cache_file_path} did not pass validation; redownloading from: {example}"
                )
                content = None

        if content is None:
            # lazy init session
            session = self._lazy_get_session()

            if session is None:
                raise ValueError("session was not initialized")

            for attempt in range(self.config.max_retries + 1):
                # throttle ourselves
                sleep = self.config.busy_wait + (self.config.retry_backoff * 2 ** (attempt - 1) if attempt else 0)
                time.sleep(sleep)

                try:
                    response = session.get(example, timeout=self.config.timeout)
                except (requests.ConnectionError, requests.Timeout):
                    self.log.warning(f"network error (attempt {attempt + 1}) for {example}", exc_info=True)
                    continue

                # unretryable; stop immediately so reconfiguration and waiting can occur
                if response.status_code == 403:
                    # we have been kicked off the website in the worst way possible: ip ban. abort the mission
                    raise ValueError(
                        f"You have been IP banned from {urlparse(example).hostname} - preprocessing has been cancelled."
                    )

                # retryable technically, can be transient
                if response.status_code == 404:
                    # not found - typical, can actually be retried because sometimes this is transient in my experience
                    continue
                if response.status_code == 429:
                    time.sleep(self._retry_after_seconds(response, default=60.0))
                    continue
                if 500 <= response.status_code < 600:
                    time.sleep(self._retry_after_seconds(response, default=0.0))
                    continue

                if response.ok:
                    try:
                        self._validate_response(example, response)
                    except Exception:
                        self.log.warning(f"validation error (attempt {attempt + 1}) for {example}", exc_info=True)
                        continue

                    if self.config.respect_opt_out_headers and not self.opt_out_headers_allow(response):
                        self.log.info(f"{example} opts out of AI training in its response headers; dropping it")
                        return None

                    content = response.content

                    if cache_file_path is not None:
                        tmp = cache_file_path.with_suffix(".part")
                        tmp.write_bytes(content)
                        os.replace(tmp, cache_file_path)

                    break
                else:
                    self.log.error(f"{response.status_code} for {example}: {response.text[:200]!r}")

        if content is not None:
            return self.extract_content(example, content)

        self.log.warning(
            f"Failed to obtain content for url={example} - "
            f"this row will contain a null for one of output_columns={self.config.output_columns}, "
            f"and nulls {'are' if self.config.filter_null_outputs else 'are not'} filtered out after mapping"
        )
        return None

    @staticmethod
    def _retry_after_seconds(response: requests.Response, default: float) -> float:
        """Seconds to wait from a ``Retry-After`` header given as delay-seconds or an HTTP-date, else ``default``."""
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return default

        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

        try:
            retry_at = email.utils.parsedate_to_datetime(retry_after)
        except (TypeError, ValueError):
            return default

        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=datetime.UTC)
        return max(0.0, (retry_at - datetime.datetime.now(datetime.UTC)).total_seconds())

    def _lazy_get_session(self) -> requests.Session:
        """Helper for caching to basically take the ``Session`` object out of the mapper's fingerprint state."""
        key = repr(self.config)
        if key not in _SESSIONS:
            session = requests.Session()
            session.headers["User-Agent"] = self.user_agent()
            _SESSIONS[key] = session
        return _SESSIONS[key]

    def opt_out_headers_allow(self, response: requests.Response) -> bool:
        from datadiligence.rules import TDMRepHeader, XRobotsTagHeader

        rules = (XRobotsTagHeader(user_agent=self.user_agent()), TDMRepHeader())
        return all(rule.is_allowed(response=response) for rule in rules)

    def _validate_response(self, example: str, response: requests.Response) -> None:
        """Initial validation for just the content-length for all download responses, but also delegates to subclass."""
        if "Content-Length" in response.headers:
            expected = int(response.headers["Content-Length"])
            actual = response.raw.tell()
            if expected != actual:
                raise ValueError(
                    f"Content-Length header value ({expected}) does not match real content length {actual}"
                )
        self.validate_content(example, response.content)

    @abstractmethod
    def user_agent(self) -> str:
        """Allows each subclass to configure a different user-agent if necessary; there is no default."""
        ...

    @abstractmethod
    def validate_content(self, example: str, content: bytes) -> None:
        """Additional content validation to be performed if necessary."""
        ...

    @abstractmethod
    def extract_content(self, example: str, content: bytes) -> O:
        """For converting raw bytes into the actual output column feature type."""
        ...


class UrlBytesMapper(UrlMapper[bytes, UrlBytesMapperConfig]):
    """Implementation of ``UrlMapper`` that just the bytes"""

    def user_agent(self) -> str:
        """Generic bytes-downloader user-agent tell, traces uniquely back to this project but not to the individual."""
        return "larry-bytes-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def validate_content(self, example: str, content: bytes) -> None:
        """Validates conditionally that the bytes in the response start with some expected content prefix."""
        if self.config.validate_starts_with and not content.startswith(self.config.validate_starts_with.encode()):
            raise ValueError(
                f"did not start with {self.config.validate_starts_with}: {example} - {str(content)[:200]!r}"
            )

    def extract_content(self, example: str, content: bytes) -> bytes:
        """Identity to extract bytes from a URL."""
        return content

    def output_feature(self) -> FeatureType | None:
        """Retains the single output feature's type as byte/binary data."""
        return Value("binary")
