import hashlib
from abc import ABC, abstractmethod
from urllib.parse import quote


class ResizedImageUrls(ABC):
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size

    def fits(self, width: int | None, height: int | None) -> bool:
        return width is not None and height is not None and max(width, height) <= self.max_size

    @abstractmethod
    def original(self, identifier: str) -> str:
        ...

    @abstractmethod
    def resized(self, identifier: str) -> str:
        ...

    def url(self, identifier: str, width: int | None, height: int | None) -> str:
        if self.fits(width, height):
            return self.original(identifier)

        return self.resized(identifier)


class WikimediaThumbnailUrls(ResizedImageUrls):
    STANDARD_THUMBNAIL_WIDTHS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)
    URL_SAFE = ";@$!*(),~:"

    def __init__(self, upload_url: str, max_size: int) -> None:
        super().__init__(max_size)
        if max_size not in self.STANDARD_THUMBNAIL_WIDTHS:
            raise ValueError(
                f"thumbnail_width={max_size} is not a standard Wikimedia thumbnail width "
                f"{self.STANDARD_THUMBNAIL_WIDTHS}; upload.wikimedia.org rejects other sizes"
            )

        self.upload_url = upload_url

    def fits(self, width: int | None, height: int | None) -> bool:
        return width is None or width <= self.max_size

    def path(self, name: str) -> tuple[str, str]:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()
        quoted = quote(name, safe=self.URL_SAFE)
        return f"{digest[0]}/{digest[:2]}/{quoted}", quoted

    def original(self, identifier: str) -> str:
        path, _ = self.path(identifier)
        return f"{self.upload_url}/{path}"

    def resized(self, identifier: str) -> str:
        path, quoted = self.path(identifier)
        return f"{self.upload_url}/thumb/{path}/{self.max_size}px-{quoted}"


class SmithsonianIdsUrls(ResizedImageUrls):
    def __init__(self, ids_url: str, max_size: int) -> None:
        super().__init__(max_size)
        self.ids_url = ids_url

    def original(self, identifier: str) -> str:
        return f"{self.ids_url}?id={quote(identifier, safe=':/')}"

    def resized(self, identifier: str) -> str:
        return f"{self.original(identifier)}&max={self.max_size}"


class IIIFImageUrls(ResizedImageUrls):
    def __init__(
            self,
            max_size: int,
            base_url: str | None = None,
            full_size: str = "max",
            size_template: str = "!{size},{size}",
            quality: str = "default",
            image_format: str = "jpg",
    ) -> None:
        super().__init__(max_size)
        self.base_url = base_url
        self.full_size = full_size
        self.size_template = size_template
        self.quality = quality
        self.image_format = image_format

    def service(self, identifier: str) -> str:
        if self.base_url is None:
            return identifier.rstrip("/")

        return f"{self.base_url.rstrip('/')}/{quote(identifier, safe='')}"

    def image(self, identifier: str, size: str) -> str:
        return f"{self.service(identifier)}/full/{size}/0/{self.quality}.{self.image_format}"

    def original(self, identifier: str) -> str:
        return self.image(identifier, self.full_size)

    def fits(self, width: int | None, height: int | None) -> bool:
        if self.size_template.startswith("{size},"):
            return width is not None and width <= self.max_size

        return super().fits(width, height)

    def resized(self, identifier: str) -> str:
        return self.image(identifier, self.size_template.format(size=self.max_size))
