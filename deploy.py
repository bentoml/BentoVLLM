# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bentoml==1.4.0a1",
#     "huggingface-hub",
#     "rich",
# ]
# ///
import multiprocessing
import subprocess, argparse, uuid, os, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


@dataclass
class DeployResult:
  bento_tag: str
  success: bool
  error: str = ""
  is_update: bool = False


def get_deployment_name(bento_tag: str) -> str:
  suffix = str(uuid.uuid4())[:6]
  return f"{bento_tag.split(':')[0].replace('.', '-')}-{suffix}"


def check_and_pull_bento(bento_tag: str, context: str) -> bool:
  """Check if bento exists locally, if not try to pull it from the context."""
  try:
    # Check if bento exists locally
    subprocess.run(
      ["bentoml", "get", bento_tag],
      capture_output=True,
      text=True,
      check=True,
    )
    return True
  except subprocess.CalledProcessError:
    try:
      # Attempt to pull the bento
      subprocess.run(
        ["bentoml", "pull", bento_tag, "--context", context],
        capture_output=True,
        text=True,
        check=True,
      )
      return True
    except subprocess.CalledProcessError:
      return False


def deploy_bento(bento_tag: str, context: str, progress: Progress, task_id: int) -> DeployResult:
  """Deploy a single bento or update existing deployment."""
  try:
    # First check if bento exists and try to pull if needed
    progress.update(task_id, description=f"[blue]Checking/pulling bento {bento_tag}...[/]")
    if not check_and_pull_bento(bento_tag, context):
      return DeployResult(bento_tag, False, f"[red]Failed to find or pull bento {bento_tag}[/]")

    deployment_name = get_deployment_name(bento_tag)

    progress.update(task_id, description=f"[blue]Creating new deployment {deployment_name} with {bento_tag}...[/]")
    subprocess.run(
      [
        "bentoml",
        "deploy",
        bento_tag,
        "--name",
        deployment_name,
        "--env",
        "HF_TOKEN",
        "--no-wait",
        "--timeout",
        "3600",
        "--context",
        context,
      ],
      capture_output=True,
      text=True,
      check=True,
    )

    # Wait for deployment to reach "running" status
    progress.update(task_id, description=f"[blue]Waiting for {deployment_name} to be ready...[/]")
    time.sleep(1)
    while (
      json.loads(
        subprocess.run(
          ["bentoml", "deployment", "get", deployment_name, "-o", "json", "--context", context],
          capture_output=True,
          text=True,
          check=True,
        ).stdout
      )["status"]["status"]
      != "running"
    ):
      time.sleep(2)

    # Terminate the deployment
    progress.update(task_id, description=f"[blue]Terminating {deployment_name}...[/]")
    time.sleep(1)
    subprocess.run(
      ["bentoml", "deployment", "terminate", deployment_name, "--context", context],
      capture_output=True,
      text=True,
      check=True,
    )

    # Wait for deployment to be fully terminated
    while (
      json.loads(
        subprocess.run(
          ["bentoml", "deployment", "get", deployment_name, "-o", "json", "--context", context],
          capture_output=True,
          text=True,
          check=True,
        ).stdout
      )["status"]["status"]
      != "terminated"
    ):
      time.sleep(2)

    # Delete the deployment
    progress.update(task_id, description=f"[blue]Deleting deployment {deployment_name}...[/]")
    subprocess.run(
      ["bentoml", "deployment", "delete", deployment_name, "--context", context],
      capture_output=True,
      text=True,
      check=True,
    )

    return DeployResult(bento_tag, True)
  except subprocess.CalledProcessError as e:
    return DeployResult(bento_tag, False, f"[red]Deployment failed: {e.stderr}[/]")
  except Exception as e:
    return DeployResult(bento_tag, False, str(e))


def deploy_all_bentos(bento_tags: List[str], context: str, workers: int) -> List[DeployResult]:
  """Deploy all bentos in parallel using a thread pool."""
  console = Console()
  results = []

  with Progress(
    SpinnerColumn(spinner_name="bouncingBar"),
    TextColumn("[progress.description]{task.description}"),
    console=console,
  ) as progress:
    overall_task = progress.add_task("[yellow]Deploying bentos...[/]", total=len(bento_tags))
    deploy_tasks = {tag: progress.add_task(f"[cyan]Waiting to deploy {tag}...[/]", total=1) for tag in bento_tags}

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_bento = {
        executor.submit(deploy_bento, tag, context, progress, deploy_tasks[tag]): tag for tag in bento_tags
      }

      for future in as_completed(future_to_bento):
        result = future.result()
        results.append(result)
        progress.advance(overall_task)
        tag_task = deploy_tasks[result.bento_tag]

        if result.success:
          action = "updated" if result.is_update else "deployed"
          progress.update(tag_task, description=f"[green]✓ {result.bento_tag} {action}[/]", completed=1)
        else:
          progress.update(tag_task, description=f"[red]✗ {result.bento_tag}: {result.error}[/]", completed=1)

  return results


def main() -> int:
  parser = argparse.ArgumentParser(description="Deploy all built bentos")
  parser.add_argument(
    "--context",
    type=str,
    required=True,
    help="Context to push bentos",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=multiprocessing.cpu_count(),
    help=f"Number of parallel workers (default: {multiprocessing.cpu_count()})",
  )
  args = parser.parse_args()

  template_dir = Path(__file__).parent
  builds_file = template_dir / "successful_builds.txt"

  if not builds_file.exists():
    print("Error: successful_builds.txt not found. Run build.py first.")
    return 1

  with open(builds_file) as f:
    bento_tags = [line.strip() for line in f if line.strip()]
    bento_tags.reverse()

  if not bento_tags:
    print("Error: No bento tags found in successful_builds.txt")
    return 1

  if not os.environ.get("HF_TOKEN"):
    print("Error: Must set HF_TOKEN")
    return 1

  console = Console()
  console.print(f"[bold]Deploying {len(bento_tags)} bentos to {args.context} with {args.workers} workers[/]")

  results = deploy_all_bentos(bento_tags, args.context, args.workers)
  successful = [r for r in results if r.success]
  updates = [r for r in successful if r.is_update]
  new_deploys = [r for r in successful if not r.is_update]

  # Print summary
  console.print("\n[bold]Deployment Summary:[/]")
  console.print(f"Total bentos: {len(bento_tags)}")
  console.print(f"Successful deployments: {len(successful)}")
  console.print(f"  - New deployments: {len(new_deploys)}")
  console.print(f"  - Updated deployments: {len(updates)}")
  console.print(f"Failed deployments: {len(results) - len(successful)}")

  # Write successful deployments to file
  successful_deploys_file = template_dir / "successful_deploys.txt"
  with open(successful_deploys_file, "w") as f:
    for result in successful:
      f.write(f"{result.bento_tag}\n")

  console.print(f"\nSuccessful deployments written to {successful_deploys_file}")

  return 0 if len(successful) == len(bento_tags) else 1


if __name__ == "__main__":
  raise SystemExit(main())
