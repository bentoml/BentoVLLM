# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "rich",
# ]
# ///
import subprocess, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


@dataclass
class PushResult:
  bento_tag: str
  success: bool
  error: str = ""


def push_bento(bento_tag: str, context: str, console: Console) -> PushResult:
  """Push a single bento to the registry."""
  try:
    console.print(f"[yellow]Pushing {bento_tag} to {context}...[/]")

    # Run bentoml push with output capture
    result = subprocess.run(
      ["uvx", "--with", "huggingface-hub", "bentoml", "push", bento_tag, "--context", context],
      capture_output=True,
      text=True,
      check=True,
    )
    return PushResult(bento_tag, True)

  except subprocess.CalledProcessError as e:
    return PushResult(bento_tag, False, f"Push failed: {e.stderr}")
  except Exception as e:
    return PushResult(bento_tag, False, str(e))


def push_all_bentos(bento_tags: List[str], context: str, workers: int) -> List[PushResult]:
  """Push all bentos in parallel using a thread pool."""
  console = Console()
  results = []

  with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
  ) as progress:
    task = progress.add_task("Pushing bentos...", total=len(bento_tags))

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_bento = {executor.submit(push_bento, tag, context, console): tag for tag in bento_tags}

      for future in as_completed(future_to_bento):
        result = future.result()
        results.append(result)
        progress.advance(task)

        if result.success:
          console.print(f"[green]✓ {result.bento_tag}[/]")
        else:
          console.print(f"[red]✗ {result.bento_tag}: {result.error}[/]")

  return results


def main() -> int:
  parser = argparse.ArgumentParser(description="Push all built bentos to registry")
  parser.add_argument(
    "--context",
    type=str,
    required=True,
    help="Context to push bentos",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Number of parallel workers (default: 4)",
  )
  args = parser.parse_args()

  template_dir = Path(__file__).parent
  builds_file = template_dir / "successful_builds.txt"

  if not builds_file.exists():
    print("Error: successful_builds.txt not found. Run build.py first.")
    return 1

  with open(builds_file) as f:
    bento_tags = [line.strip() for line in f if line.strip()]

  if not bento_tags:
    print("Error: No bento tags found in successful_builds.txt")
    return 1

  console = Console()
  console.print(f"[bold]Pushing {len(bento_tags)} bentos to {args.context} with {args.workers} workers[/]")

  results = push_all_bentos(bento_tags, args.context, args.workers)
  successful_pushes = [r for r in results if r.success]

  # Print summary
  console.print("\n[bold]Push Summary:[/]")
  console.print(f"Total bentos: {len(bento_tags)}")
  console.print(f"Successful pushes: {len(successful_pushes)}")
  console.print(f"Failed pushes: {len(results) - len(successful_pushes)}")

  if successful_pushes:
    console.print("\n[bold]Successfully pushed bentos:[/]")
    for result in successful_pushes:
      console.print(f"- {result.bento_tag}")

  return 0 if len(successful_pushes) == len(bento_tags) else 1


if __name__ == "__main__": raise SystemExit(main())
