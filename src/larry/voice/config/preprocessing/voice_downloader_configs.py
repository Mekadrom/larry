import dataclasses
from collections.abc import Sequence

from larry.common.config.data.downloader_configs import MultiFileDownloaderConfig, RangedIndexMultiFileDownloaderConfig, \
    DownloaderConfig, SingleFileDownloaderConfig


@dataclasses.dataclass(kw_only=True)
class SCOTUSDownloaderConfig:
    base_url: str
    term_manifest_file_name: str
    docket_manifest_file_name: str


@dataclasses.dataclass(kw_only=True)
class SCOTUSTermManifestDownloaderConfig(RangedIndexMultiFileDownloaderConfig, SCOTUSDownloaderConfig):
    def configs_for(self, range_value: int) -> Sequence[tuple[str, DownloaderConfig]]:
        return [
            ("SingleFileDownloader", SingleFileDownloaderConfig(
                download_base_url=self.base_url,
                download_path=f"/oral_arguments/argument_audio/{range_value}",
                output_file_name=f"term_manifest_audio/{range_value}.html",
                validate_content_contains=str(range_value),
                busy_wait=self.busy_wait,
            )),
            ("SingleFileDownloader", SingleFileDownloaderConfig(
                download_base_url=self.base_url,
                download_path=f"/oral_arguments/argument_transcript/{range_value}",
                output_file_name=f"term_manifest_transcript/{range_value}.html",
                validate_content_contains=str(range_value),
                busy_wait=self.busy_wait,
            )),
        ]


@dataclasses.dataclass(kw_only=True)
class SCOTUSDocketManifestDownloaderConfig(MultiFileDownloaderConfig, SCOTUSDownloaderConfig):
    ...
