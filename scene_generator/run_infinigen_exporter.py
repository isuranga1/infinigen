import argparse
import subprocess
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(
    description="Export generated Infinigen scenes to FBX."
)
parser.add_argument(
    "--input_root",
    type=str,
    required=False,
    help="Root folder containing generated scenes",
)
parser.add_argument(
    "--input_folder", type=str, required=False, help="Specific scene folder to export"
)
parser.add_argument(
    "--output_root", type=str, required=False, help="Root folder for exports"
)
parser.add_argument(
    "--output_folder", type=str, required=False, help="Specific output folder"
)
parser.add_argument(
    "--name", type=str, default=None, help="Export name (defaults to folder name)"
)
parser.add_argument(
    "--cleanup", action="store_true", help="Delete input scene after export"
)
args = parser.parse_args()


def export_scene(input_folder: Path, output_folder: Path, name: str):
    output_folder.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "-m",
        "infinigen.tools.export",
        "--input_folder",
        str(input_folder),
        "--output_folder",
        str(output_folder),
        "--name",
        name,
        "-f",
        "fbx",
        "-r",
        "1024",
    ]
    subprocess.run(command, check=True)
    if args.cleanup:
        shutil.rmtree(input_folder, ignore_errors=True)
        print(f"🧹 Cleaned up {input_folder}")


if args.input_folder and args.output_folder:
    name = args.name or Path(args.input_folder).name
    export_scene(Path(args.input_folder), Path(args.output_folder), name)
elif args.input_root and args.output_root:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    for folder in sorted(input_root.rglob("*")):
        if (folder / "scene.blend").exists():  # assume valid scene
            rel = folder.relative_to(input_root)
            export_scene(folder, output_root / rel, folder.name)
else:
    raise ValueError(
        "Must provide either --input_folder + --output_folder OR --input_root + --output_root"
    )
