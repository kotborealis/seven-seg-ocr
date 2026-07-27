"""Command-line interface for seven-seg-ocr."""

from __future__ import annotations

import json
import sys

import click

from seven_seg_ocr import __version__, read_display


@click.command()
@click.argument("image", type=click.Path(exists=True))
@click.option(
    "--n-digits", "-n", type=int, default=2, help="Expected number of digits (default: 2)."
)
@click.option("--json", "-j", "as_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--positions",
    "-p",
    type=str,
    default=None,
    help="Comma-separated x-positions of digit left edges (e.g. '32,48').",
)
@click.option(
    "--y",
    type=int,
    default=None,
    help="Top of digit band in pixels (auto-detected if omitted).",
)
@click.option(
    "--dh",
    type=int,
    default=35,
    help="Digit height in pixels (default: 35).",
)
@click.version_option(version=__version__)
def main(
    image: str,
    n_digits: int,
    as_json: bool,
    positions: str | None,
    y: int | None,
    dh: int,
) -> None:
    """Read a 7-segment display from IMAGE.

    Recognizes numbers (2.2, 3.5) and hex codes (A1, b3).

    \b
    Examples:
      seven-seg-ocr photo.jpg
      seven-seg-ocr photo.jpg -n 3 --json
      seven-seg-ocr photo.jpg -p 32,48
    """
    pos_list = None
    if positions:
        pos_list = [int(p.strip()) for p in positions.split(",")]

    try:
        result = read_display(
            image,
            n_digits=n_digits,
            positions=pos_list,
            y=y,
            dh=dh,
        )
    except Exception as exc:
        if as_json:
            json.dump({"error": str(exc)}, sys.stdout)
        else:
            click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        # Convert numpy types for JSON serialization
        clean = {
            "reading": result["reading"],
            "confidence": result["confidence"],
            "format": result["format"],
            "details": [
                {
                    "char": char,
                    "confidence": float(conf),
                    "top3": [{"char": c, "confidence": float(s)} for c, s in top3],
                }
                for char, conf, top3 in result["details"]
            ],
        }
        json.dump(clean, sys.stdout, indent=2)
    else:
        click.echo(f"Reading:  {result['reading']}")
        click.echo(f"Format:  {result['format']}")
        click.echo(f"Confidence: {result['confidence']:.3f}")
        click.echo()
        for i, (char, conf, top3) in enumerate(result["details"]):
            if char == ".":
                click.echo(f"  [{i}] '.' (decimal point)")
            else:
                top_str = " ".join(f"{c}={s:.3f}" for c, s in top3)
                click.echo(f"  [{i}] '{char}'  conf={conf:.3f}  [{top_str}]")


if __name__ == "__main__":
    main()
