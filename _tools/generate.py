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
import yaml, shutil, subprocess, json, argparse, multiprocessing
from pathlib import Path
from jinja2 import Template
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


def is_nightly_branch():
  """Check if we are on the nightly branch."""
  try:
    result = subprocess.run(
      ["git", "rev-parse", "--abbrev-ref", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
    )
    return result.stdout.strip() == "nightly"
  except subprocess.SubprocessError:
    return False


def update_model_descriptions(config, template_dir):
  certified_repos_path = template_dir / "bentocloud-homepage-news" / "certified-bento-repositories.json"
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
    repo_name = f"bentovllm-{model_name}-service"

    # handle aliases
    if model_name.startswith("deepseek-r1-distill"):
      repo_name = f"bentovllm-r1{model_name.removeprefix('deepseek-r1-distill')}-service"

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


def generate_jinja_context(model_name, config):
  model_config = config[model_name]
  use_mla = model_config.get("use_mla", False)
  use_nightly = model_config.get("use_nightly", False)
  use_vision = model_config.get("vision", False)
  engine_config_struct = model_config.get("engine_config", {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"})

  service_config = model_config.get("service_config", {})

  requires_hf_tokens = "envs" in service_config and any(it["name"] == "HF_TOKEN" for it in service_config["envs"])
  if "envs" not in service_config:
    service_config["envs"] = []

  service_config["envs"].extend([
    {"name": "UV_NO_PROGRESS", "value": "1"},
    {"name": "HF_HUB_DISABLE_PROGRESS_BARS", "value": "1"},
    {
      "name": "VLLM_ATTENTION_BACKEND",
      "value": "FLASHMLA" if use_mla else "FLASH_ATTN",
    },
  ])

  if "enable_prefix_caching" not in engine_config_struct:
    engine_config_struct["enable_prefix_caching"] = True

  build_config = model_config.get("build", {})
  if "exclude" not in build_config:
    build_config["exclude"] = []
  build_config["exclude"] = [*build_config["exclude"], "*.pth", "*.pt", "original/**/*"]

  if "post" not in build_config:
    build_config["post"] = []

  if use_nightly:
    build_config["post"].append(
      "uv pip install --compile-bytecode vllm --pre --extra-index-url https://wheels.vllm.ai/nightly"
    )
  build_config["post"].append(
    "uv pip install --compile-bytecode flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.5"
  )

  context = {
    "model_name": model_name,
    "model_id": engine_config_struct["model"],
    "vision": use_vision,
    "generate_config": model_config.get("generate_config", {}),
    "service_config": service_config,
    "engine_config": engine_config_struct,
    "server_config": model_config.get("server_config", {}),
    "labels": dict(owner="bentoml-team", type="prebuilt"),
    "metadata": model_config["metadata"],
    "requires_hf_tokens": requires_hf_tokens,
    "build": build_config,
    "exclude": build_config["exclude"],
    "reasoning": model_config.get("reasoning", False),
    "embeddings": model_config.get("embeddings", False),
    "nightly": use_nightly,
  }

  requirements = model_config.get("requirements", [])
  if len(requirements) > 0:
    context["requirements"] = requirements

  return context


def generate_readme(config, template_dir):
  models = [{"name": name, "engine_config": cfg.get("engine_config", {})} for name, cfg in config.items()]
  is_nightly = is_nightly_branch()

  with open(template_dir / "README.md.tpl", "r") as f:
    template_content = f.read()

  rendered = Template(template_content).render(models=models, nightly=is_nightly)

  with open(template_dir / "README.md", "w") as f:
    f.write(rendered)


@dataclass
class GenerateResult:
  model_name: str
  success: bool
  error: str = ""
  no_changes: bool = False


def generate_model(
  model_name: str, config: dict, template_dir: Path, progress: Progress, task_id: int
) -> GenerateResult:
  """Generate a single model's project."""
  output_dir = template_dir / model_name
  try:
    progress.update(task_id, description=f"[blue]Generating {model_name}...[/]")

    context = generate_jinja_context(model_name, config)

    # Create a temporary directory for new generation
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_output_dir = Path(temp_dir) / model_name
      temp_output_dir.mkdir(parents=True)

      # Walk through template directory and render each file
      template_source = template_dir / "_src"

      shutil.copytree(
        template_source,
        temp_output_dir,
        dirs_exist_ok=True,
        ignore=lambda src, names: [i for i in names if i.endswith(".j2")],
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

      if output_dir.exists():
        shutil.rmtree(output_dir)
        shutil.copytree(temp_output_dir, output_dir)
      else:
        # If directory doesn't exist, just move the generated one
        shutil.copytree(temp_output_dir, output_dir)

    progress.update(task_id, description=f"[green]✓ {model_name}[/]", completed=1)
    return GenerateResult(model_name, True)

  except Exception as e:
    progress.update(task_id, description=f"[red]✗ {model_name}: {str(e)}[/]", completed=1)
    return GenerateResult(model_name, False, str(e))


def generate_all_models(config: dict, template_dir: Path, force: bool = False, workers: int = 1) -> bool:
  """Generate all model projects in parallel."""
  console = Console()
  models = list(config.keys())
  results = []

  with Progress(
    SpinnerColumn(spinner_name="bouncingBar"),
    TextColumn("[progress.description]{task.description}"),
    console=console,
  ) as progress:
    overall_task = progress.add_task("[yellow]Generating models...[/]", total=len(models))
    gen_tasks = {model: progress.add_task(f"[cyan]Waiting to generate {model}...[/]", total=1) for model in models}

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_model = {
        executor.submit(generate_model, model_name, config, template_dir, progress, gen_tasks[model_name]): model_name
        for model_name in models
      }

      for future in as_completed(future_to_model):
        result = future.result()
        results.append(result)
        progress.advance(overall_task)

    successful = [r for r in results if r.success]
    no_changes = [r for r in successful if r.no_changes]
    updated = [r for r in successful if not r.no_changes]

    # Print summary
    console.print("\n[bold]Generation Summary:[/]")
    console.print(f"Total models: {len(models)}")
    console.print(f"Successful generations: {len(successful)}")
    console.print(f"  - Updated: {len(updated)}")
    console.print(f"  - No changes needed: {len(no_changes)}")
    console.print(f"Failed generations: {len(results) - len(successful)}")

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
  args = parser.parse_args()

  template_dir = Path(__file__).parent.parent
  tools_dir = template_dir / "_tools"
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

  success = generate_all_models(filtered_config, template_dir, args.force, args.workers)

  # Generate README.md after all models are processed
  console.print("\n[yellow]Generating README.md...[/]")
  generate_readme(config, template_dir)
  console.print("[green]✓ Generated README.md[/]")

  # Format all python files
  console.print("\n[yellow]Formatting Python files...[/]")
  subprocess.run(["ruff", "format", template_dir.__fspath__()], check=True, capture_output=True)
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
  update_model_descriptions(config, template_dir)
  console.print("[green]✓ Updated model descriptions[/]")

  return 0 if success else 1


if __name__ == "__main__":
  raise SystemExit(main())
