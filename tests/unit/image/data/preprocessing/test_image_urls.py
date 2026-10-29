import hashlib

import pytest

from larry.image.data.preprocessing.image_urls import IIIFImageUrls, SmithsonianIdsUrls, WikimediaThumbnailUrls

IDS = "https://ids.si.edu/ids/deliveryService"


def test_images_that_fit_are_requested_whole_and_others_resized() -> None:
    urls = SmithsonianIdsUrls(IDS, 960)

    assert urls.url("NMAH-1", 960, 640) == f"{IDS}?id=NMAH-1"
    assert urls.url("NMAH-1", 961, 640) == f"{IDS}?id=NMAH-1&max=960"
    assert urls.url("NMAH-1", None, None) == f"{IDS}?id=NMAH-1&max=960"


def test_smithsonian_ark_identifiers_keep_their_slashes() -> None:
    assert SmithsonianIdsUrls(IDS, 960).resized("ark:/65665/m3abc") == f"{IDS}?id=ark:/65665/m3abc&max=960"


def test_wikimedia_uses_a_width_step_and_originals_when_width_is_unknown() -> None:
    urls = WikimediaThumbnailUrls("https://upload.wikimedia.org/wikipedia/commons", 960)
    digest = hashlib.md5(b"Tall.jpg").hexdigest()
    base = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{digest[0]}/{digest[:2]}/Tall.jpg"

    assert urls.url("Tall.jpg", 900, 4000) == f"https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[:2]}/Tall.jpg"
    assert urls.url("Tall.jpg", 2000, 1000) == f"{base}/960px-Tall.jpg"
    assert "/thumb/" not in urls.url("Tall.jpg", None, None)


def test_wikimedia_rejects_non_standard_widths() -> None:
    with pytest.raises(ValueError, match="standard Wikimedia thumbnail width"):
        WikimediaThumbnailUrls("https://upload.wikimedia.org/wikipedia/commons", 1000)


def test_iiif_identifiers_are_escaped_under_a_base_url() -> None:
    urls = IIIFImageUrls(960, base_url="https://iiif.example.org/iiif/2/")

    assert urls.url("a/b c", 4000, 3000) == "https://iiif.example.org/iiif/2/a%2Fb%20c/full/!960,960/0/default.jpg"
    assert urls.url("a/b c", 800, 600) == "https://iiif.example.org/iiif/2/a%2Fb%20c/full/max/0/default.jpg"


def test_iiif_service_urls_can_be_used_as_identifiers() -> None:
    urls = IIIFImageUrls(960, full_size="full")

    assert urls.url("https://iiif.example.org/iiif/2/abc/", 100, 100) == (
        "https://iiif.example.org/iiif/2/abc/full/full/0/default.jpg"
    )
    assert urls.resized("https://iiif.example.org/iiif/2/abc") == (
        "https://iiif.example.org/iiif/2/abc/full/!960,960/0/default.jpg"
    )


def test_iiif_can_size_by_width_alone() -> None:
    urls = IIIFImageUrls(843, base_url="https://www.artic.edu/iiif/2", full_size="full", size_template="{size},")

    assert urls.url("abc", 2203, 3128) == "https://www.artic.edu/iiif/2/abc/full/843,/0/default.jpg"
    assert urls.url("abc", 800, 3000) == "https://www.artic.edu/iiif/2/abc/full/full/0/default.jpg"
