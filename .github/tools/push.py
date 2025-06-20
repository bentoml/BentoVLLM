# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bentoml>=1.4.12",
#     "huggingface-hub",
#     "rich",
# ]
# ///
from __future__ import annotations
import multiprocessing, subprocess, argparse, typing, pathlib, dataclasses

from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if typing.TYPE_CHECKING:
  from rich.progress import TaskID


@dataclasses.dataclass
class PushResult:
  bento_tag: str
  success: bool
  error: str = ''


def push_bento(bento_tag: str, context: str, progress: Progress, task_id: TaskID) -> PushResult:
  """Push a single bento to the registry."""
  try:
    progress.update(task_id, description=f'[blue]Pushing {bento_tag} to {context}...[/]')

    # Run bentoml push with output capture
    subprocess.run(['bentoml', 'push', bento_tag, '--context', context], capture_output=True, text=True, check=True)
    return PushResult(bento_tag, True)

  except subprocess.CalledProcessError as e:
    return PushResult(bento_tag, False, f'[red]Push failed: {e.stderr}[/]')
  except Exception as e:
    return PushResult(bento_tag, False, str(e))


def push_all_bentos(bento_tags: list[str], context: str, workers: int) -> list[PushResult]:
  """Push all bentos in parallel using a thread pool."""
  console = Console()
  results = []
  num_bentos = len(bento_tags)

  console.print()
  console.print(f'[bold]Pushing {num_bentos} bentos to {context} with {workers} workers[/]')

  with Progress(
    SpinnerColumn(spinner_name='bouncingBar'), TextColumn('[progress.description]{task.description}'), console=console
  ) as progress:
    overall_task = progress.add_task(f'[yellow]Pushing {num_bentos} bentos...[/]', total=num_bentos)

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_bento = {executor.submit(push_bento, tag, context, progress, overall_task): tag for tag in bento_tags}

      pushed_count = 0
      for future in as_completed(future_to_bento):
        result = future.result()
        results.append(result)
        pushed_count += 1
        bento_tag = result.bento_tag

        progress.update(
          overall_task, description=f'[blue]({pushed_count}/{num_bentos}) Processing pushes...[/]', advance=1
        )

        if result.success:
          console.print(f'  [green]✓ Pushed {bento_tag}[/]')
        else:
          console.print(f'  [red]✗ Failed {bento_tag}: {result.error}[/]')

    progress.update(overall_task, description=f'[bold green]Finished processing {num_bentos} pushes.[/]')

  console.print()

  successful_pushes = [r for r in results if r.success]
  failed_pushes = [r for r in results if not r.success]

  console.print('[bold]Push Summary:[/]')
  console.print(f'Total bentos: {num_bentos}')
  console.print(f'Successful pushes: {len(successful_pushes)}')
  console.print(f'Failed pushes: {len(failed_pushes)}')

  if failed_pushes:
    console.print('\n[bold red]Failed Push Details:[/]')
    for r in failed_pushes:
      console.print(f'  - {r.bento_tag}: {r.error}')

  return results


def main() -> int:
  parser = argparse.ArgumentParser(description='Push all built bentos to registry')
  parser.add_argument('--context', type=str, required=True, help='Context to push bentos')
  parser.add_argument(
    '--workers', type=int, default=multiprocessing.cpu_count(), help='Number of parallel workers (default: 4)'
  )
  args = parser.parse_args()

  git_dir = pathlib.Path(__file__).parent.parent.parent
  builds_file = git_dir / 'successful_builds.txt'

  if not builds_file.exists():
    print('Error: successful_builds.txt not found. Run build.py first.')
    return 1

  with open(builds_file) as f:
    bento_tags = [line.strip() for line in f if line.strip()]

  if not bento_tags:
    print('Error: No bento tags found in successful_builds.txt')
    return 1

  results = push_all_bentos(bento_tags, args.context, args.workers)
  successful_pushes = [r for r in results if r.success]

  return 0 if len(successful_pushes) == len(bento_tags) else 1


if __name__ == '__main__':
  raise SystemExit(main())
