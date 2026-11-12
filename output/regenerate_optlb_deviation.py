"""Save the simulation and real-robot PSD deviation PDFs with a revised title."""

import argparse
from decimal import Decimal
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject


LOGS_DIR = (
    Path(__file__).resolve().parents[1] / "spinup/examples/pytorch/logs"
)
SOURCE_DIR = LOGS_DIR / "Fep_HW_314/kinematics_error_bounds"
FIGURES = ("dev_OptLB_dev.pdf", "dev_OptLB_dev_real.pdf")

# Both original PDFs use the same Matplotlib 3.1.1 TeX title operators.
FONT_SIZE = Decimal("13.9476961395")
TITLE_X = Decimal("149.9531809325")
TITLE_Y = "437.9136435504"
WORD_SPACE = Decimal("326.3854980469")
CONTROLLER_KERN = b"(Con) 27.1987915039 (troller)"
OLD_TITLE = (
    f"/F1 {FONT_SIZE} Tf\n{TITLE_X} {TITLE_Y} Td\n"
    f"[ (Hybrid) -{WORD_SPACE} ".encode("ascii")
    + CONTROLLER_KERN
    + b" ] TJ"
)


def revise_title(source):
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected a single-page figure in {source}.")

    page = reader.pages[0]
    contents = page.get_contents().get_data()
    if contents.count(OLD_TITLE) != 1:
        raise ValueError(f"Could not find the original title exactly once in {source}.")

    font = page["/Resources"]["/Font"]["/F1"]
    if font["/BaseFont"] != "/CMR12":
        raise ValueError(f"Expected the original Computer Modern Roman font in {source}.")
    widths = font["/Widths"]
    first_char = font["/FirstChar"]

    def width(text):
        return sum(Decimal(str(widths[ord(char) - first_char])) for char in text)

    # The Controller run and its preceding space are identical in both titles.
    # Move the starting x coordinate by half the added width to keep it centered.
    prefix = "Residual"
    controller = "SAC-PI"
    added_width = width(prefix + controller) + WORD_SPACE - width("Hybrid")
    x = TITLE_X - added_width * FONT_SIZE / 2000
    new_title = (
        f"/F1 {FONT_SIZE} Tf\n{x:.10f} {TITLE_Y} Td\n"
        f"[ ({prefix}) -{WORD_SPACE} ({controller}) -{WORD_SPACE} ".encode("ascii")
        + CONTROLLER_KERN
        + b" ] TJ"
    )

    writer = PdfWriter(clone_from=reader)
    stream = DecodedStreamObject()
    stream.set_data(contents.replace(OLD_TITLE, new_title, 1))
    writer.pages[0].replace_contents(stream.flate_encode())
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=LOGS_DIR)
    args = parser.parse_args()
    if args.output_dir.resolve() == SOURCE_DIR.resolve():
        parser.error("Choose an output directory separate from the original PDFs.")

    figures = [(name, revise_title(SOURCE_DIR / name)) for name in FIGURES]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, writer in figures:
        output = args.output_dir / name
        writer.write(output)
        print(output.resolve())


if __name__ == "__main__":
    main()
