# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2",
#     "pyyaml",
#     "ruff",
#     "rich",
#     "pathspec",
# ]
# ///
from __future__ import annotations

import yaml, shutil, copy, subprocess, json, argparse, multiprocessing, pathlib, typing, dataclasses

from jinja2 import Template
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from concurrent.futures import ThreadPoolExecutor, as_completed

if typing.TYPE_CHECKING:
  from rich.progress import TaskID


def is_nightly_branch():
  """Check if we are on the nightly branch."""
  try:
    result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip() == "nightly"
  except subprocess.SubprocessError:
    return False


def update_model_descriptions(config, git_dir):
  certified_repos_path = git_dir / "bentocloud-homepage-news" / "certified-bento-repositories.json"
  if not certified_repos_path.exists():
    print("Warning: certified-bento-repositories.json not found, skipping description updates")
    return

  with certified_repos_path.open("r") as f:
    certified_repos_struct = json.load(f)
    certified_bentos = certified_repos_struct["certified_bentos"]

  # Create a mapping of model names to their certified repo data
  certified_list = [repo["repo_name"] for repo in certified_bentos]

  # Update descriptions for each model
  for model_name, model_config in config.items():
    repo_name = model_name.lower()

    if repo_name in certified_list:
      continue

    labels = ["✍️ Text Generation"]
    if model_config.get("vision", False):
      labels.append("👁️ Image-to-Text")

    metadata = model_config["metadata"]

    image_url = f"https://raw.githubusercontent.com/bentoml/bentocloud-homepage-news/main/imgs/{metadata['provider'].lower().replace(' ', '-')}.png"
    bentos = dict(
      org_name="bentoml",
      repo_name=repo_name,
      description=dict(
        name=metadata["description"],
        text=f"{metadata['description']} developed by {metadata['provider']} and served using vLLM and BentoML. It offers capabilities for streaming and compatibility with OpenAI's API",
        link=f"github.com/bentoml/BentoVLLM/tree/main/{model_name}",
        image=image_url,
        label=labels,
      ),
    )
    certified_bentos.append(bentos)

  with certified_repos_path.open("w", encoding="utf-8") as f:
    f.write(json.dumps(dict(certified_bentos=certified_bentos), indent=2, ensure_ascii=False))


def generate_jinja_context(model_name: str, config: dict[str, dict[str, typing.Any]]) -> dict[str, typing.Any]:
  model_config = config[model_name]
  use_mla = model_config.get("use_mla", False)
  use_nightly = model_config.get("nightly", False)
  use_vision = model_config.get("vision", False)
  engine_config_struct = model_config.get("engine_config", {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"})
  model = engine_config_struct.pop("model")

  service_config = model_config.get("service_config", {})

  requires_hf_tokens = "envs" in service_config and any(it["name"] == "HF_TOKEN" for it in service_config["envs"])
  if "envs" not in service_config:
    service_config["envs"] = []

  service_config["envs"].extend([{"name": "UV_NO_PROGRESS", "value": "1"}])
  if use_mla:
    service_config["envs"].append({"name": "VLLM_ATTENTION_BACKEND", "value": "FLASHMLA"})

  if "max_num_seqs" not in engine_config_struct:
    engine_config_struct["max_num_seqs"] = 256  # Aligned with v0

  if "tensor_parallel_size" not in engine_config_struct:
    engine_config_struct["tensor_parallel_size"] = service_config.get("resources", {}).get("gpu", 1)

  build_config = model_config.get("build", {})
  if "exclude" not in build_config:
    build_config["exclude"] = []
  build_config["exclude"] = [*build_config["exclude"], "*.pth", "*.pt", "original/**/*"]

  if "post" not in build_config:
    build_config["post"] = []
  if "pre" not in build_config:
    build_config["pre"] = []
  if "system_packages" not in build_config:
    build_config["system_packages"] = []

  if use_nightly:
    build_config["system_packages"].extend(["pkg-config", "libssl-dev", "curl", "git"])
    build_config["pre"].extend([
      "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -v -y --profile complete --default-toolchain nightly"
    ])
    build_config["post"].extend([
      "uv pip install --compile-bytecode --no-progress torch --index-url https://download.pytorch.org/whl/cu126",
      "uv pip install --compile-bytecode --no-progress xformers --index-url https://download.pytorch.org/whl/cu126",
      "uv pip install --compile-bytecode --no-progress vllm --extra-index-url https://wheels.vllm.ai/3d13ca0e242a99ef1ca53de1828689130924b3f5",
    ])
  build_config["post"].append(
    "uv pip install --compile-bytecode --no-progress https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl"
  )
  build_config["system_packages"] = set(build_config["system_packages"])

  context = {
    "model_name": model_name,
    "model_id": model,
    "vision": use_vision,
    "c2a": model_config.get("c2a", True),
    "task": model_config.get("task", "generate"),
    "deployment_config": model_config.get("deployment_config", {}),
    "generate_config": model_config.get("generate_config", {}),
    "nightly": use_nightly,
    "service_config": service_config,
    "engine_config": engine_config_struct,
    "labels": dict(owner="bentoml-team", type="prebuilt", project="bentovllm", openai_endpoint="/v1"),
    "metadata": model_config["metadata"],
    "requires_hf_tokens": requires_hf_tokens,
    "build": build_config,
    "exclude": build_config["exclude"],
    "reasoning": model_config.get("reasoning", False),
    "embeddings": model_config.get("embeddings", False),
    "system_prompt": model_config.get("system_prompt", None),
    "prompt": model_config.get("prompt", None),
  }

  requirements = model_config.get("requirements", [])
  if len(requirements) > 0:
    context["requirements"] = requirements

  return context


def generate_readme(config, git_dir: pathlib.Path, skip_nightly):
  models = [{"name": name, "engine_config": cfg.get("engine_config", {})} for name, cfg in config.items()]
  is_nightly = not skip_nightly and is_nightly_branch()
  with open((git_dir / ".github" / "README.md"), "w") as f:
    f.write(Template((git_dir / ".github" / "README.md.j2").read_text()).render(models=models, nightly=is_nightly))


@dataclasses.dataclass
class GenerateResult:
  model_name: str
  success: bool
  error: str = ""
  no_changes: bool = False


def generate_model(
  model_name: str, config: dict, git_dir: pathlib.Path, progress: Progress, task_id: TaskID
) -> GenerateResult:
  """Generate a single model's project."""
  output_dir = git_dir / model_name
  try:
    progress.update(task_id, description=f"[blue]Generating {model_name}...[/]")

    context = generate_jinja_context(model_name, config)

    # Create a temporary directory for new generation
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_output_dir = pathlib.Path(temp_dir) / model_name
      temp_output_dir.mkdir(parents=True)

      # Walk through template directory and render each file
      template_source = git_dir / ".github" / "src"

      shutil.copytree(
        template_source,
        temp_output_dir,
        dirs_exist_ok=True,
        ignore=lambda _, names: [i for i in names if i.endswith(".j2")],
      )

      if "requirements" in context:
        with (temp_output_dir / "requirements.txt").open("a") as f:
          for req in context["requirements"]:
            f.write(f"{req}\n")

      for template_path in template_source.rglob("*.j2"):
        # Get relative path from template root
        rel_path = template_path.relative_to(template_source)
        target_path = temp_output_dir / str(rel_path).replace("_src", model_name).replace(".j2", "")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Read and render template
        with open(template_path, "r") as f:
          template_content = f.read()
        rendered = Template(template_content).render(**context)

        # Write rendered content
        with open(target_path, "w") as f:
          f.write(rendered)

      deployment_config = context["deployment_config"]
      if deployment_config:
        for k, v in deployment_config.items():
          with (temp_output_dir / f"{k}.yaml").open("w") as f:
            yaml.safe_dump(dict(args=v), f)

      if output_dir.exists():
        shutil.rmtree(output_dir)
        shutil.copytree(temp_output_dir, output_dir)
      else:
        # If directory doesn't exist, just move the generated one
        shutil.copytree(temp_output_dir, output_dir)

    progress.update(task_id, description=f"[green]✓ {model_name}[/]", completed=1)
    return GenerateResult(model_name, True)

  except Exception as e:
    progress.update(task_id, description=f"[red]✗ {model_name}: {e!s}[/]", completed=1)
    return GenerateResult(model_name, False, str(e))


def generate_all_models(config: dict, git_dir: pathlib.Path, force: bool = False, workers: int = 1) -> bool:
  """Generate all model projects in parallel."""
  console = Console()
  models = list(config.keys())
  results = []
  num_models = len(models)

  console.print()
  console.print(f"[bold]Generating {num_models} models with {workers} workers[/]")

  with Progress(
    SpinnerColumn(spinner_name="bouncingBar"),
    TextColumn("[progress.description]{task.description}"),
    console=console,
  ) as progress:
    overall_task = progress.add_task(f"[yellow]Generating {num_models} models...[/]", total=num_models)
    # Removed individual task creation

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_model = {
        # Pass overall_task ID instead of individual task ID
        executor.submit(generate_model, model_name, config, git_dir, progress, overall_task): model_name
        for model_name in models
      }

      generated_count = 0
      for future in as_completed(future_to_model):
        result = future.result()
        results.append(result)
        generated_count += 1
        model_name = result.model_name

        # Update progress bar description to show general progress
        progress.update(
          overall_task,
          description=f"[blue]({generated_count}/{num_models}) Processing models...[/]",
          advance=1,
        )

        # Print status on a new line
        if result.success:
          if result.no_changes:
            console.print(f"  [cyan]✓ No changes for {model_name}[/]")
          else:
            console.print(f"  [green]✓ Generated {model_name}[/]")
        else:
          console.print(f"  [red]✗ Failed {model_name}: {result.error}[/]")

    # Final update after loop completes - just show completion message
    progress.update(overall_task, description=f"[bold green]Finished processing {num_models} models.[/]")

  console.print()  # Add newline after progress bar

  successful = [r for r in results if r.success]
  no_changes = [r for r in successful if r.no_changes]
  updated = [r for r in successful if not r.no_changes]
  failed = [r for r in results if not r.success]

  # Print summary
  console.print("[bold]Generation Summary:[/]")
  console.print(f"Total models: {num_models}")
  console.print(f"Successful generations: {len(successful)}")
  console.print(f"  - Updated: {len(updated)}")
  console.print(f"  - No changes needed: {len(no_changes)}")
  console.print(f"Failed generations: {len(failed)}")

  if failed:
    console.print("\n[bold red]Failed Generation Details:[/]")
    for r in failed:
      console.print(f"  - {r.model_name}: {r.error}")

  return len(successful) == len(models)


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate model service from config.yaml")
  parser.add_argument(
    "model_names", nargs="*", help="Specific model names to generate. If not provided, generates all models."
  )
  parser.add_argument("--force", action="store_true", help="Force regeneration even if directory exists")
  parser.add_argument(
    "--workers",
    type=int,
    default=multiprocessing.cpu_count(),
    help=f"Number of parallel workers (default: {multiprocessing.cpu_count()})",
  )
  parser.add_argument(
    "--skip-readme-nightly",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="whether to skip generating nightly readme.",
  )
  args = parser.parse_args()

  git_dir = pathlib.Path(__file__).parent.parent.parent
  tools_dir = git_dir / ".github" / "tools"
  with (tools_dir / "config.yaml").open("r") as f:
    config = yaml.safe_load(f)
  readme_config = copy.deepcopy(config)

  console = Console()

  expected_dirs = set(config.keys())
  known_dirs = {".git", ".github", "venv", ".venv", "bentocloud-homepage-news", ".mypy_cache", ".ruff_cache"}
  found_dirs = {d.name for d in git_dir.iterdir() if d.is_dir()}
  unexpected_dirs = found_dirs - expected_dirs - known_dirs

  if unexpected_dirs:
    console.print(
      "[bold yellow]Warning:[/bold yellow] Found directories in the repository root not listed in config.yaml:"
    )
    for dir_name in unexpected_dirs:
      console.print(f"  - {dir_name}")
    # Remove the unexpected directories
    console.print("[bold red]Removing unexpected directories...[/]")
    for dir_name in unexpected_dirs:
      try:
        target_dir = git_dir / dir_name
        if target_dir.is_dir():  # Double check it's a directory before removing
          shutil.rmtree(target_dir)
          console.print(f"[green]Removed {dir_name}[/]")
        else:
          console.print(f"[yellow]Skipped {dir_name} as it is not a directory.[/]")
      except OSError as e:  # noqa: PERF203
        console.print(f"[red]Error removing {dir_name}: {e}[/]")

  if args.model_names:
    invalid_models = [model for model in args.model_names if model not in config]
    if invalid_models:
      console.print(f"[red]Error: Models not found in config.yaml: {', '.join(invalid_models)}[/]")
      return 1
    filtered_config = {model: config[model] for model in args.model_names}
  else:
    filtered_config = config

  success = generate_all_models(filtered_config, git_dir, args.force, args.workers)

  # Generate README.md after all models are processed
  console.print("\n[yellow]Generating README.md...[/]")
  generate_readme(readme_config, git_dir, args.skip_readme_nightly)
  console.print("[green]✓ Generated README.md[/]")

  # Format all python files
  console.print("\n[yellow]Formatting Python files...[/]")
  subprocess.run(["ruff", "format", git_dir.__fspath__()], check=True, capture_output=True)
  subprocess.run(
    [
      "ruff",
      "format",
      "--isolated",
      "--config",
      "indent-width=2",
      "--config",
      "line-length=119",
      "--config",
      "preview=true",
      tools_dir.__fspath__(),
    ],
    check=True,
    capture_output=True,
  )
  console.print("[green]✓ Formatted Python files[/]")

  # Update model descriptions
  console.print("\n[yellow]Updating model descriptions...[/]")
  update_model_descriptions(config, git_dir)
  console.print("[green]✓ Updated model descriptions[/]")

  return 0 if success else 1


if __name__ == "__main__":
  raise SystemExit(main())
