# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "rich",
#     "bentoml>=1.4.12",
#     "huggingface-hub",
# ]
# ///
from __future__ import annotations

import yaml, subprocess, os, argparse, multiprocessing, hashlib, pathlib, dataclasses, typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if typing.TYPE_CHECKING:
  from rich.progress import TaskID


@dataclasses.dataclass
class BuildResult:
  model_name: str
  bento_tag: str
  success: bool
  error: str = ""


def hash_file(file_path):
  hasher = hashlib.sha256()
  with file_path.open("rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def ensure_venv(req_txt, venv_dir, cfg):
  build = cfg.get("build", {})
  build_args = build.get("args", [])

  if not venv_dir.exists():
    subprocess.run(["uv", "venv", venv_dir, "-p", "3.11"], check=True)
    subprocess.run(
      [
        "uv",
        "pip",
        "install",
        "--compile-bytecode",
        "--prerelease=allow",
        "bentoml>=1.4.7",
        "-p",
        venv_dir / "bin" / "python",
      ],
      check=True,
      capture_output=True,
    )
    subprocess.run(
      ["uv", "pip", "install", "-r", req_txt, *build_args, "-p", venv_dir / "bin" / "python"],
      check=True,
      capture_output=True,
    )
  return venv_dir


def build_model(
  model_name: str, cfg: dict[str, typing.Any], git_dir: pathlib.Path, progress: Progress, task_id: TaskID
) -> BuildResult:
  """Build a single model's bento."""
  model_dir = git_dir / model_name
  if not model_dir.exists():
    return BuildResult(model_name, "", False, f"Directory {model_dir} does not exist")

  req_txt_file = model_dir / "requirements.txt"
  venv_dir = git_dir / "venv" / f"{model_name}-{hash_file(req_txt_file)[:7]}"
  version_path = ensure_venv(req_txt_file, venv_dir, cfg)

  try:
    # Run bentoml build with output capture
    result = subprocess.run(
      [version_path / "bin" / "python", "-m", "bentoml", "build", "service:VLLM", "--output", "tag"],
      capture_output=True,
      text=True,
      check=True,
      cwd=model_dir,
      env=os.environ,
    )

    # Extract bento tag from output - format: __tag__:bentovllm-model-name-service:hash
    output = result.stdout.strip()
    if output.startswith("__tag__:"):
      bento_tag = output[8:]  # Remove "__tag__:" prefix
      return BuildResult(model_name, bento_tag, True)

    return BuildResult(model_name, "", False, f"Unexpected output format: {output}")

  except subprocess.CalledProcessError as e:
    return BuildResult(model_name, "", False, f"Build failed: {e.stderr}")
  except Exception as e:
    return BuildResult(model_name, "", False, str(e))


def build_bentos(config: dict[str, typing.Any], git_dir: pathlib.Path, workers: int) -> list[BuildResult]:
  """Build all models in parallel using a thread pool."""
  console = Console()
  results = []
  num_models = len(config)

  with Progress(
    SpinnerColumn(spinner_name="bouncingBar"),
    TextColumn("[progress.description]{task.description}"),
    console=console,
  ) as progress:
    overall_task = progress.add_task(f"[yellow]Building {num_models} bentos...[/]", total=num_models)

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_model = {
        executor.submit(build_model, model, cfg, git_dir, progress, overall_task): model
        for model, cfg in config.items()
      }

      built_count = 0
      for future in as_completed(future_to_model):
        result = future.result()
        results.append(result)
        built_count += 1
        model_name = result.model_name

        # Update progress bar description to show general progress
        progress.update(
          overall_task, description=f"[blue]({built_count}/{num_models}) Processing models...[/]", advance=1
        )

        if result.success:
          console.print(f"  [green]✓ Built {result.bento_tag}[/]")
        else:
          console.print(f"  [red]✗ Failed {model_name}: {result.error}[/]")

    progress.update(overall_task, description=f"[bold green]Finished processing {num_models} models.[/]")

  console.print()
  return results


def main() -> int:
  parser = argparse.ArgumentParser(description="Build all model bentos in parallel")
  parser.add_argument(
    "model_names", nargs="*", help="Specific model names to build. If not provided, builds all models."
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=multiprocessing.cpu_count(),
    help=f"Number of parallel workers (default: {multiprocessing.cpu_count()})",
  )
  args = parser.parse_args()

  git_dir = pathlib.Path(__file__).parent.parent.parent
  tools_dir = git_dir / ".github" / "tools"
  with (tools_dir / "config.yaml").open("r") as f:
    config = yaml.safe_load(f)

  console = Console()

  if args.model_names:
    invalid_models = [model for model in args.model_names if model not in config]
    if invalid_models:
      console.print(f"[red]Error: Models not found in config.yaml: {', '.join(invalid_models)}[/]")
      return 1
    filtered_config = {model: config[model] for model in args.model_names}
  else:
    filtered_config = config

  console.print()
  console.print(f"[bold]Building {len(filtered_config)} bentos with {args.workers} workers[/]")

  results = build_bentos(filtered_config, git_dir, args.workers)
  successful_builds = [r for r in results if r.success]

  # Print summary
  console.print("\n[bold]Build Summary:[/]")
  console.print(f"Total bentos: {len(filtered_config)}")
  console.print(f"Successful builds: {len(successful_builds)}")
  failed_builds = len(results) - len(successful_builds)
  console.print(f"Failed builds: {failed_builds}")

  # Print details for failed builds
  if failed_builds > 0:
    console.print("\n[bold red]Failed Build Details:[/]")
    for r in results:
      if not r.success:
        console.print(f"  - {r.model_name}: {r.error}")

  # Save successful bento tags to file for later use
  if successful_builds:
    bento_tags = [r.bento_tag for r in successful_builds]
    with open(git_dir / "successful_builds.txt", "w") as f:
      f.write("\n".join(bento_tags))
    console.print("\n[green]Saved successful build tags to successful_builds.txt[/]")

  return 0 if len(successful_builds) == len(filtered_config) else 1


if __name__ == "__main__":
  raise SystemExit(main())
