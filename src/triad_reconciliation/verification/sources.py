"""Data endpoints of this release.

The studied cohort is private and is not distributed. The four auxiliary corpora the
manuscript names for representation pretraining are public, and each entry below records what
the probe returned. A live probe is opt-in; the default pass reads this list and opens no
socket, so a verification run is reproducible offline.

Ref: Sec. 5.2, p. 14 (the four auxiliary resources, their access route and their total image
count), Data availability, p. 17.
"""

from __future__ import annotations

from dataclasses import dataclass

PROBE_SCHEME = "https"


@dataclass(frozen=True)
class Source:
    name: str
    version: str
    licence: str
    access: str
    url: str
    probe: str
    images: int | None = None
    cited_by: str = ""


SOURCES: tuple[Source, ...] = (
    Source(
        name="analysed multicentre cohort (four sites, two regions)",
        version="not stated in the manuscript",
        licence="not stated; held under the sites' data-sharing agreements",
        access="not distributable at record level; the Data availability statement, p. 17, "
        "withholds both institution and site identities",
        url="none. No URL describes this cohort and none is invented.",
        probe="not applicable; the release ships a reader for a table the caller supplies",
        images=None,
        cited_by="Sec. 5.2, p. 13-14",
    ),
    Source(
        name="PAPILA data descriptor (the article the deposition asks to be cited)",
        version="Scientific Data 9, 291 (2022)",
        licence="article text CC BY 4.0; the images follow the deposition's own GPL 3.0+",
        access="open full text",
        url="https://doi.org/10.1038/s41597-022-01388-1",
        probe="resolved; the full text states that there are 488 files in JPEG format "
        "corresponding to the right and left eye of 244 patients",
        images=None,
        cited_by="reference 38",
    ),
    Source(
        name="PAPILA fundus images with clinical data of both eyes",
        version="figshare deposition version 2, 2022-05-06",
        licence="GPL 3.0+, as stated on the deposition page",
        access="open download from the deposition page",
        url="https://figshare.com/articles/dataset/PAPILA/14798004",
        probe="resolved in the rendering client; the page text carries the dataset name, the "
        "glaucoma keyword and the fundus-image description, and the descriptor it links states "
        "488 files from 244 patients",
        images=488,
        cited_by="reference 38; the descriptor reports 488 images from 244 patients",
    ),
    Source(
        name="ACRIMA labelled fundus images",
        version="the database released with the 2019 validation study",
        licence="not stated on the article page; the images are distributed for research use",
        access="open download from the deposition page linked by the study",
        url="https://doi.org/10.1186/s12938-019-0649-y",
        probe="the full text was read directly; it states 705 labelled images, 396 glaucomatous "
        "and 309 normal",
        images=705,
        cited_by="reference 39",
    ),
    Source(
        name="RIM-ONE DL unified retinal image database",
        version="the release as hosted by the authors, last updated 2020-11-29",
        licence="research and educational use, free of charge; redistribution and commercial "
        "use prohibited, as stated in the repository README",
        access="open download of the image archive from the repository",
        url="https://github.com/miag-ull/rim-one-dl",
        probe="resolved; the repository text states 313 normal and 172 glaucomatous "
        "retinographies, 485 in total",
        images=485,
        cited_by="reference 40",
    ),
    Source(
        name="ODIR-5K ocular disease recognition set",
        version="the 2019 Ocular Disease Intelligent Recognition release",
        licence="challenge terms, as specified on the challenge page",
        access="registration on the challenge platform for the scored leaderboard; the image "
        "set is also published publicly",
        url="https://odir2019.grand-challenge.org/",
        probe="resolved; the challenge page states 5,000 structured patient records with "
        "binocular colour fundus photographs, and the published count is 8,000 images split "
        "7,000 training and 1,000 testing",
        images=8000,
        cited_by="reference 41",
    ),
)


def auxiliary_total() -> int:
    """The image total over the auxiliary corpora that publish a count."""

    return sum(source.images or 0 for source in SOURCES)


def render_dataset_urls(probed: bool, probe_notes: tuple[str, ...] = ()) -> str:
    """Render the endpoint list as plain text, with the probe design recorded."""

    lines: list[str] = [
        "# Data endpoints of this release.",
        "#",
        "# The studied cohort is private and is not distributed, so no link describes it. The four",
        "# auxiliary corpora the manuscript names for representation pretraining are public; each",
        "# entry below records its access route, its conditions and, where the source publishes one,",
        "# its image count.",
        "#",
        f"# Live probing is {'enabled' if probed else 'disabled for this pass'}; the default pass reads",
        "# this list and opens no socket, so the file is a fixed point of the repository rather than a",
        "# record of one moment's network.",
        f"# Probe design: scheme {PROBE_SCHEME}, one attempt per endpoint, redirects followed,",
        "# content matched against the resource named above before a line is written as live.",
        "",
    ]
    for source in SOURCES:
        lines.append(f"name     : {source.name}")
        lines.append(f"version  : {source.version}")
        lines.append(f"licence  : {source.licence}")
        lines.append(f"access   : {source.access}")
        lines.append(f"url      : {source.url}")
        lines.append(f"probe    : {source.probe}")
        if source.images is not None:
            lines.append(f"images   : {source.images}")
        lines.append(f"cited in : {source.cited_by}")
        lines.append("")
    lines.append("# --- probed this pass ---")
    lines.append("")
    if probe_notes:
        for note in probe_notes:
            lines.append(note)
    else:
        lines.append("count    : 0")
        lines.append(
            "detail   : no endpoint was probed in this pass; the list above is the shipped record"
        )
    lines.append("")
    lines.append("# --- what was not added ---")
    lines.append("")
    lines.append("count    : 0")
    lines.append(
        "detail   : the manuscript names no other public resource, and no substitute link has been"
    )
    lines.append(
        "           added for the private cohort. A link that did not resolve is not written here."
    )
    return "\n".join(lines) + "\n"
