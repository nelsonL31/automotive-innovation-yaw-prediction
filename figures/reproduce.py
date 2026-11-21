"""Rebuild the English figures from the bundled derived CSV files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory to receive Fig1–Fig7 as PNG, SVG and PDF")
    parser.add_argument("--figures", nargs="+", type=int, choices=range(1, 8), default=range(1, 8),
                        help="Optional subset of figure numbers")
    parser.add_argument("--verify", action="store_true",
                        help="Compare the rebuilt PDFs with the archived reference pixel hashes")
    return parser.parse_args()


def verify_pdf(path, expected):
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(path) as doc:
        if len(doc) != 1:
            raise ValueError(f"{path.name} has {len(doc)} pages")
        image = doc[0].render(scale=2).to_pil().convert("RGB")
        if list(image.size) != expected["size_px"] or hashlib.sha256(image.tobytes()).hexdigest() != expected["rgb_sha256"]:
            raise ValueError(f"{path.name}: artwork differs from the reference rendering. "
                             "Check the font, plotting environment and PDF renderer listed in README.md.")


def main():
    args = arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    reference = json.loads((ROOT / "reference.json").read_text(encoding="utf-8"))
    for number in dict.fromkeys(args.figures):
        name = f"Fig{number}"
        for filename, expected in reference[name]["inputs"].items():
            actual = hashlib.sha256((ROOT / name / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"{name}/{filename}: the derived data differ from the archived inputs")
        subprocess.run([sys.executable, str(ROOT / name / "plot.py")], cwd=output, env=environment, check=True)
        for extension in ("png", "svg", "pdf"):
            if not (output / f"{name}.{extension}").is_file():
                raise FileNotFoundError(f"{name}.{extension}")
        if args.verify:
            verify_pdf(output / f"{name}.pdf", reference[name])
        print(f"{name} complete", flush=True)


if __name__ == "__main__":
    main()

