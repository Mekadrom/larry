import csv
import glob
import html
import os.path
import re
import urllib.parse
from datetime import datetime
from pathlib import Path

from larry.common.config.data.downloader_configs import SingleFileBackupDownloaderConfig
from larry.common.data.downloaders import MultiFileDownloader, RangedIndexMultiFileDownloader, \
    SingleFileBackupDownloader
from larry.voice.config.preprocessing.voice_downloader_configs import SCOTUSDocketManifestDownloaderConfig, \
    SCOTUSTermManifestDownloaderConfig

# ex: "href='../audio/1234/az_0-9.ext'"
#                     term/docket
_CASE_LINK = re.compile(r"href='\.\./audio/(\d{4})/([0-9A-Za-z._-]+)'")
_TRANSCRIPT_LINK = re.compile(r"argument_transcripts/(\d{4})/([0-9A-Za-z._-]+?)(?:_[0-9a-z]{4})?\.pdf")
_MP3 = re.compile(r"""["'](https?://[^"']*?/mp3files/[^"']+\.mp3)["']""", re.IGNORECASE)
_PDF = re.compile(r"""href=['"]([^'"]*argument_transcripts/[^'"]+\.pdf)['"]""", re.IGNORECASE)
_SPAN = re.compile(r"""<span id=["'][^"']*%s["'][^>]*>(.*?)</span>""", re.DOTALL)


class SCOTUSTermManifestDownloader(RangedIndexMultiFileDownloader[SCOTUSTermManifestDownloaderConfig]):
    def __init__(self, config: SCOTUSTermManifestDownloaderConfig, download_cache_abs_dir: Path) -> None:
        super().__init__(config, download_cache_abs_dir)
        self.output_file_path = self.download_cache_abs_dir / self.config.term_manifest_file_name

    def ensure_downloaded(self) -> bool:
        if os.path.exists(self.output_file_path):
            self.log.info(f"Already downloaded and produced {self.output_file_path}")
            return False

        self.log.info(f"Downloading term manifest files and aggregating to {self.output_file_path}")

        super().ensure_downloaded()

        # need to process the index files into a single file
        term_htmls = glob.glob(str(self.download_cache_abs_dir / "term_manifest_audio/*.html"))

        self.log.info(f"globbed: {term_htmls}")

        docket_indices = {}

        for term_html in term_htmls:
            term_html_path = Path(term_html)
            term = term_html_path.stem
            term_html_content = term_html_path.read_text("utf-8")

            self.log.info(f"processing term={term}")

            seen = set()
            out = []
            for t, docket in _CASE_LINK.findall(term_html_content):
                # check term contents because the wrong one is sometimes served
                # (the index url for 2023 frequently serves 2025 for some reason)
                if t == term and docket not in seen:
                    self.log.info(f"found docket={docket} for term={term} in first pass")
                    seen.add(docket)
                    out.append(docket)

            if out:
                docket_indices[term] = out
                continue

            for t, docket in _TRANSCRIPT_LINK.findall(term_html_content):
                if t == term and docket not in seen:
                    self.log.info(f"found docket={docket} for term={term} in second pass")
                    seen.add(docket)
                    out.append(docket)
            if out:
                self.log.info(
                    f"[fetch] term {term}: recovered {len(out)} dockets via the transcript index "
                    f"(the audio index served the wrong term on all {self.config.max_retries} tries -- "
                )
                docket_indices[term] = out

        with open(self.output_file_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(("term", "docket"))
            w.writerows((k, v) for k, values in docket_indices.items() for v in values)

        return True


class SCOTUSCaseDetailSingleFileBackupDownloader(SingleFileBackupDownloader):
    def __init__(self, config: SingleFileBackupDownloaderConfig, download_cache_abs_dir: Path, docket: str) -> None:
        self.docket = docket

        super().__init__(config, download_cache_abs_dir)

    def make_backup_download_paths(self, default_url: str) -> list[str]:
        out = []

        docket_upper_url = re.sub(self.docket, self.docket.upper(), default_url, flags=re.I)
        docker_suffix_url = re.sub(r"orig", "Orig", default_url, flags=re.I)

        for v in (docket_upper_url, docker_suffix_url):
            if v not in out:
                out.append(v)
        return out


class SCOTUSDocketManifestDownloader(MultiFileDownloader[SCOTUSDocketManifestDownloaderConfig]):
    def __init__(self, config: SCOTUSDocketManifestDownloaderConfig, download_cache_abs_dir: Path) -> None:
        super().__init__(config, download_cache_abs_dir)
        self.input_file_path = self.download_cache_abs_dir / self.config.term_manifest_file_name
        self.output_file_path = self.download_cache_abs_dir / self.config.docket_manifest_file_name

    def ensure_downloaded(self) -> bool:
        if os.path.exists(self.output_file_path):
            return False

        if not os.path.exists(self.input_file_path):
            raise ValueError(
                f'expected {self.download_cache_abs_dir / self.config.term_manifest_file_name} to exist but it did not'
            )

        self.delegates = [
            SCOTUSCaseDetailSingleFileBackupDownloader(SingleFileBackupDownloaderConfig(
                download_base_url=self.config.base_url,
                download_path=f"oral_arguments/audio/{term}/{docket}",
                output_file_name=f"case_detail/{term}/{docket}.html",
            ), self.download_cache_abs_dir, docket)
            for term, docket in self._get_term_indices()
        ]

        super().ensure_downloaded()

        # post-process the downloaded folder of html files to another tsv that maps term, docket
        term_dirs = glob.glob(str(self.download_cache_abs_dir / "case_detail/*"))

        records = []
        for term_dir in term_dirs:
            term = Path(term_dir).stem
            docket_htmls = glob.glob(str(self.download_cache_abs_dir / "case_detail" / f"{term}/*.html"))
            for docket_html in docket_htmls:
                docket = Path(docket_html).stem

                url = urllib.parse.urljoin(self.config.base_url, f"oral_arguments/audio/{term}/{docket}")

                content = Path(docket_html).read_text()
                mp3 = _MP3.search(content)
                pdf = _PDF.search(content)
                pdf_url = urllib.parse.urljoin(url, html.unescape(pdf.group(1))) if pdf else None

                records.append({
                    "docket": docket,
                    "term": int(term),
                    "case_name": self._get_span(content, "lblCaseName"),
                    "date_argued": self._normalize_date(self._get_span(content, "lblDate")),
                    "mp3_url": html.unescape(mp3.group(1)) if mp3 else None,
                    "pdf_url": pdf_url,
                    "page_url": url,
                })

        fieldnames = list(dict.fromkeys(k for r in records for k in r))  # union of keys, first-seen order
        with open(self.output_file_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", restval="")
            w.writeheader()
            w.writerows(records)

        return True

    @staticmethod
    def _normalize_date(s: str) -> str:
        return datetime.strptime(s, "%m/%d/%y").date().isoformat()  # "10/29/18" -> "2018-10-29"

    @staticmethod
    def _get_text(raw):
        return html.unescape(re.sub(r"<[^>]+>", " ", raw or "")).strip()

    def _get_span(self, page, suffix):
        m = re.search(_SPAN.pattern % re.escape(suffix), page, re.S)
        return self._get_text(m.group(1)) if m else ""

    def _get_term_indices(self) -> list[tuple[str, str]]:
        with open(self.input_file_path, newline="", encoding="utf-8") as f:
            return [(row["term"], row["docket"]) for row in csv.DictReader(f, delimiter="\t")]
