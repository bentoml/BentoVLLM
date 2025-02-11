# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "rich",
#     "bentoml",
#     "uv",
# ]
# ///
import yaml, subprocess, os, argparse, multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


@dataclass
class BuildResult:
  model_name: str
  bento_tag: str
  success: bool
  error: str = ""


def load_config() -> Dict:
  with open("config.yaml", "r") as f:
    return yaml.safe_load(f)


def build_model(model_name: str, template_dir: Path, console: Console) -> BuildResult:
  """Build a single model's bento."""
  model_dir = template_dir / model_name
  if not model_dir.exists():
    return BuildResult(model_name, "", False, f"Directory {model_dir} does not exist")

  try:
    # Change to model directory
    os.chdir(model_dir)
    console.print(f"[yellow]Building {model_name}...[/]")

    # Run bentoml build with output capture
    result = subprocess.run(
      [
        "uv",
        "run",
        "--with-requirements",
        str(model_dir / "requirements.txt"),
        "bentoml",
        "build",
        "service:VLLM",
        "--output",
        "tag",
      ],
      capture_output=True,
      text=True,
      check=True,
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
  finally:
    # Return to original directory
    os.chdir(template_dir)


def build_bentos(models: List[str], template_dir: Path, workers: int) -> List[BuildResult]:
  """Build all models in parallel using a thread pool."""
  console = Console()
  results = []

  with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
  ) as progress:
    task = progress.add_task("Building bentos...", total=len(models))

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_model = {executor.submit(build_model, model, template_dir, console): model for model in models}

      for future in as_completed(future_to_model):
        result = future.result()
        results.append(result)
        progress.advance(task)

        if result.success:
          console.print(f"[green]✓ {result.model_name}: {result.bento_tag}[/]")
        else:
          console.print(f"[red]✗ {result.model_name}: {result.error}[/]")

  return results


def main() -> int:
  parser = argparse.ArgumentParser(description="Build all model bentos in parallel")
  parser.add_argument(
    "--workers",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of parallel workers (default: number of CPU cores)",
  )
  args = parser.parse_args()

  template_dir = Path(__file__).parent
  config = load_config()
  models = list(config.keys())

  console = Console()
  console.print(f"[bold]Building {len(models)} bentos with {args.workers} workers[/]")

  results = build_bentos(models, template_dir, args.workers)
  successful_builds = [r for r in results if r.success]

  # Print summary
  console.print("\n[bold]Build Summary:[/]")
  console.print(f"Total bentos: {len(models)}")
  console.print(f"Successful builds: {len(successful_builds)}")
  console.print(f"Failed builds: {len(results) - len(successful_builds)}")

  if successful_builds:
    console.print("\n[bold]Successfully built bentos:[/]")
    for result in successful_builds:
      console.print(f"- {result.bento_tag}")

  # Save successful bento tags to file for later use
  if successful_builds:
    bento_tags = [r.bento_tag for r in successful_builds]
    with open(template_dir / "successful_builds.txt", "w") as f:
      f.write("\n".join(bento_tags))
    console.print("\n[green]Saved successful build tags to successful_builds.txt[/]")

  return 0 if len(successful_builds) == len(models) else 1


if __name__ == "__main__":
  raise SystemExit(main())
