import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from queue import Queue
import logging
import threading

# --- 1. Setup Logging and Thread-Safe State ---
log_state = {
    "generation_active": 0,
    "export_active": 0,
    "generated_success": [],
    "generated_fail": [],
    "exported_success": [],
    "exported_fail": [],
}
log_lock = threading.Lock()


def setup_logging():
    """Configures logging to file and console."""
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler("batch_run.log", mode="w")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)


def log_status():
    """Logs the current count of active and completed tasks."""
    with log_lock:
        logging.info(
            f"PROGRESS: Generating: {log_state['generation_active']} | "
            f"Exporting: {log_state['export_active']} | "
            f"Generated OK: {len(log_state['generated_success'])} | "
            f"Exported OK: {len(log_state['exported_success'])}"
        )


# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Batch run Infinigen with saved floor plans using separate parallel pools for generation and exporting."
)
parser.add_argument(
    "--floorplans_root",
    type=str,
    required=True,
    help="Root folder containing floor plan JSONs",
)
parser.add_argument(
    "--outputs_root",
    type=str,
    required=True,
    help="Root folder where generated scenes will be saved",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument(
    "--task", type=str, default="coarse", help="Task type (e.g., coarse, fine)"
)
parser.add_argument(
    "--retries", type=int, default=3, help="Retries for failed generation attempts"
)
parser.add_argument(
    "--generation_workers",
    type=int,
    default=None,
    help="Optional: number of parallel workers for scene generation.",
)
parser.add_argument(
    "--export_workers",
    type=int,
    default=None,
    help="Optional: number of parallel workers for exporting scenes.",
)
parser.add_argument(
    "--export", action="store_true", help="Enable the export process after generation"
)
parser.add_argument(
    "--exports_root",
    type=str,
    default=None,
    help="Root folder for exports (required if --export)",
)
parser.add_argument(
    "--cleanup",
    action="store_true",
    help="Delete generated scenes after a successful export",
)
args = parser.parse_args()

# --- Setup Paths and Workers ---
floorplans_root = Path(args.floorplans_root)
outputs_root = Path(args.outputs_root)
if args.export and not args.exports_root:
    raise ValueError("Must provide --exports_root if --export flag is set")
json_files = sorted(floorplans_root.rglob("*.json"))
if not json_files:
    raise FileNotFoundError(f"No JSON files found in {floorplans_root}")
total_cpus = os.cpu_count() or 4
if args.export:
    gen_workers = args.generation_workers or max(1, int(total_cpus * 0.75))
    exp_workers = args.export_workers or max(1, int(total_cpus * 0.25))
else:
    gen_workers = args.generation_workers or min(total_cpus, 8)
    exp_workers = 0

# --- Modified Task Functions with Logging ---


def run_generation(json_file: Path, export_queue: Queue = None):
    """Generates a scene, with detailed, thread-safe logging."""
    with log_lock:
        log_state["generation_active"] += 1
    log_status()

    rel_path = json_file.relative_to(floorplans_root)

    # ▼▼▼ THIS IS THE MODIFIED LINE ▼▼▼
    output_folder = outputs_root / rel_path.parent / json_file.stem / f"seed{args.seed}"
    # ▲▲▲ THIS IS THE MODIFIED LINE ▲▲▲

    output_folder.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "-m",
        "infinigen_examples.generate_indoors",
        "--seed",
        str(args.seed),
        "--task",
        args.task,
        "--output_folder",
        str(output_folder),
        "-g",
        "singleroom.gin",
        "fast_solve.gin",
        "-p",
        "compose_indoors.terrain_enabled=False",
        f"Solver.floor_plan='{str(json_file)}'",
    ]

    attempt = 0
    success = False
    while attempt < args.retries:
        try:
            logging.info(
                f"Starting generation for {json_file.name} (attempt {attempt + 1})"
            )
            subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"SUCCESS: Generation finished for {json_file.name}")
            if args.export and export_queue:
                export_task_info = {
                    "input_folder": output_folder,
                    "rel_path": rel_path,
                    "name": json_file.stem,
                }
                export_queue.put(export_task_info)
            success = True
            break
        except subprocess.CalledProcessError as e:
            attempt += 1
            logging.error(
                f"FAIL: Generation attempt {attempt} for {json_file.name} failed.\n--- ERROR ---\n{e.stderr.strip()}\n--- END ERROR ---"
            )
            time.sleep(2)

    with log_lock:
        log_state["generation_active"] -= 1
        if success:
            log_state["generated_success"].append(json_file.name)
        else:
            log_state["generated_fail"].append(json_file.name)
            logging.warning(
                f"GIVING UP on generation for {json_file.name} after {args.retries} attempts."
            )
    log_status()
    return str(json_file), success


def run_export(export_queue: Queue):
    """Pulls a scene from the queue and runs the exporter, with logging."""
    while True:
        task_info = export_queue.get()
        if task_info is None:
            break

        name = task_info["name"]
        with log_lock:
            log_state["export_active"] += 1
        log_status()

        try:
            logging.info(f"Starting export for {name}")
            export_folder = (
                Path(args.exports_root) / task_info["rel_path"].parent / name
            )
            export_cmd = [
                "python",
                "scene_generator/run_infinigen_exporter.py",
                "--input_folder",
                str(task_info["input_folder"]),
                "--output_folder",
                str(export_folder),
                "--name",
                name,
            ]
            if args.cleanup:
                export_cmd.append("--cleanup")
            subprocess.run(export_cmd, check=True, capture_output=True, text=True)
            logging.info(f"SUCCESS: Export finished for {name}")
            with log_lock:
                log_state["exported_success"].append(name)
        except subprocess.CalledProcessError as e:
            logging.error(
                f"FAIL: Export for {name} failed.\n--- ERROR ---\n{e.stderr.strip()}\n--- END ERROR ---"
            )
            with log_lock:
                log_state["exported_fail"].append(name)
        finally:
            with log_lock:
                log_state["export_active"] -= 1
            log_status()


# --- Main Execution ---

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting batch process...")
    logging.info(f"Found {len(json_files)} floor plans to process.")
    if args.export:
        logging.info(
            f"Using {gen_workers} generation workers and {exp_workers} export workers."
        )
    else:
        logging.info(f"Using {gen_workers} generation workers. Exporting is disabled.")

    if args.export:
        export_job_queue = Queue()
        with (
            ThreadPoolExecutor(
                max_workers=gen_workers, thread_name_prefix="Gen"
            ) as gen_executor,
            ThreadPoolExecutor(
                max_workers=exp_workers, thread_name_prefix="Exp"
            ) as exp_executor,
        ):
            for _ in range(exp_workers):
                exp_executor.submit(run_export, export_job_queue)

            gen_futures = {
                gen_executor.submit(run_generation, jf, export_job_queue): jf
                for jf in json_files
            }
            for future in as_completed(gen_futures):
                future.result()

            for _ in range(exp_workers):
                export_job_queue.put(None)
    else:
        with ThreadPoolExecutor(
            max_workers=gen_workers, thread_name_prefix="Gen"
        ) as executor:
            futures = {executor.submit(run_generation, jf): jf for jf in json_files}
            for future in as_completed(futures):
                future.result()

    # Final Summary
    logging.info("=" * 50)
    logging.info("BATCH PROCESS COMPLETE: FINAL SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total Generations Successful: {len(log_state['generated_success'])}")
    logging.info(f"Total Generations Failed:    {len(log_state['generated_fail'])}")
    if log_state["generated_fail"]:
        logging.info(
            f"Failed Generation Files: {', '.join(log_state['generated_fail'])}"
        )

    if args.export:
        logging.info("-" * 25)
        logging.info(
            f"Total Exports Successful:    {len(log_state['exported_success'])}"
        )
        logging.info(f"Total Exports Failed:        {len(log_state['exported_fail'])}")
        if log_state["exported_fail"]:
            logging.info(
                f"Failed Export Files: {', '.join(log_state['exported_fail'])}"
            )
    logging.info("=" * 50)
