"""Save the original real-robot tracking plot with revised labels."""

import argparse
import re
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject


FIGURE_MEMBER = "figures/real_test_position_errors_compare.pdf"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "RLC1.zip"
DEFAULT_OUTPUT = Path("real_test_position_errors_compare_revised.pdf")

# Matplotlib 3.1.1 writes the TeX label as separate glyph runs. The next
# relative text position places the subscript in the parenthetical condition.
OLD_LABEL = (
    b"[ (h) 27.1987915039 (ybrid) -326.3854980469 (con) 27.1987915039 (troller)\n"
    b"-326.3854980469 (\\(\x01) ]\n"
    b"TJ\n/F6 9.7633447326 Tf\n121.0159928831 -2.0921452999 Td"
)
Y_LABEL_START = (
    b"BT\n/F3 13.9476961395 Tf\n0 1 -1 0 16.3435224597 114.0173519346 Tm\n"
)
Y_LABEL_TAIL = (
    b"/F5 9.7633447326 Tf\n0 1 -1 0 18.4356677596 153.3177446271 Tm\n0 0 Td\n(2) Tj\n"
    b"/F1 13.9476961395 Tf\n0 1 -1 0 16.3435224597 163.3854290429 Tm\n0 0 Td\n([mm]) Tj\nET"
)


def open_original(source):
    if source.suffix.lower() != ".zip":
        return PdfReader(source)
    with ZipFile(source) as archive:
        if FIGURE_MEMBER in archive.namelist():
            return PdfReader(BytesIO(archive.read(FIGURE_MEMBER)))
        # The downloaded archive wraps the Overleaf project in a second ZIP.
        with ZipFile(BytesIO(archive.read("RLC1.zip"))) as project:
            return PdfReader(BytesIO(project.read(FIGURE_MEMBER)))


def remove_norm_subscript(content):
    start = content.find(Y_LABEL_START)
    tail = content.find(Y_LABEL_TAIL, start)
    if start < 0 or tail < 0 or b"\nET" in content[start:tail]:
        raise ValueError("Expected the original rotated y-axis label.")

    # Keep the original word space between the norm and the units. Recenter
    # the shortened label around the old midpoint along the vertical axis.
    norm_end = Decimal("153.3177446271")
    word_space = Decimal("13.9476961395") * Decimal("0.3263854980469")
    units_y = norm_end + word_space
    shift = (Decimal("163.3854290429") - units_y) / 2

    def move_glyph(match):
        y = Decimal(match[2].decode("ascii")) + shift
        return match[1] + f"{y:.10f}".encode("ascii") + b" Tm"

    norm = re.sub(
        rb"(0 1 -1 0 16\.3435224597 )(\d+\.\d+) Tm",
        move_glyph,
        content[start:tail],
    )
    units = (
        f"/F1 13.9476961395 Tf\n0 1 -1 0 16.3435224597 {units_y + shift:.10f} Tm\n"
        "0 0 Td\n([mm]) Tj\nET"
    ).encode("ascii")
    return content[:start] + norm + units + content[tail + len(Y_LABEL_TAIL):]


def revise_figure(reader):
    if len(reader.pages) != 1:
        raise ValueError("Expected the standalone, single-page figure PDF.")

    page = reader.pages[0]
    content = page.get_contents().get_data()
    if content.count(OLD_LABEL) != 1:
        raise ValueError("Expected one copy of the original Matplotlib legend label.")

    font = page["/Resources"]["/Font"]["/F1"]
    if font["/BaseFont"] != "/CMR12":
        raise ValueError("Expected the original Computer Modern Roman legend font.")
    widths = font["/Widths"]
    first_char = font["/FirstChar"]

    def width(text):
        return sum(Decimal(str(widths[ord(char) - first_char])) for char in text)

    # Keep the original font, word spacing and controller kerning. Shift the
    # condition by the width difference between the old and new text prefixes.
    old_width = width("hybrid") - Decimal("27.1987915039")
    new_width = width("residualSAC-PI") + Decimal("326.3854980469")
    shift = (new_width - old_width) * Decimal("13.9476961395") / 1000
    subscript_offset = Decimal("121.0159928831") + shift
    new_label = (
        b"[ (residual) -326.3854980469 (SAC-PI) -326.3854980469 (con) 27.1987915039 (troller)\n"
        b"-326.3854980469 (\\(\x01) ]\n"
        b"TJ\n/F6 9.7633447326 Tf\n"
        + f"{subscript_offset:.10f} -2.0921452999 Td".encode("ascii")
    )

    writer = PdfWriter(clone_from=reader)
    stream = DecodedStreamObject()
    stream.set_data(remove_norm_subscript(content.replace(OLD_LABEL, new_label, 1)))
    writer.pages[0].replace_contents(stream.flate_encode())
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path, default=DEFAULT_ARCHIVE,
        help="Original figure PDF or Overleaf project ZIP (default: ~/Downloads/RLC1.zip).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.source.resolve() == args.output.resolve():
        parser.error("Choose a different path for the revised PDF.")
    writer = revise_figure(open_original(args.source))
    writer.write(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
