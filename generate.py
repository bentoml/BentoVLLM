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
import yaml, shutil, subprocess, json, argparse
from pathlib import Path
from jinja2 import Template
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

GIT_DIRECTORY = Path(__file__).parent


def load_config():
  with open("config.yaml", "r") as f:
    return yaml.safe_load(f)


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
  engine_config_struct = model_config.get("engine_config", {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"})

  service_config = model_config.get("service_config", {})
  if not service_config.get("envs"):
    service_config["envs"] = []
  service_config["envs"].append(dict(name="UV_COMPILE_BYTECODE", value=1))

  requires_hf_tokens = any(it["name"] == "HF_TOKEN" for it in service_config["envs"])

  context = {
    "model_name": model_name,
    "model_id": engine_config_struct["model"],
    "vision": model_config.get("vision", False),
    "service_config": service_config,
    "engine_config": engine_config_struct,
    "server_config": model_config.get("server_config", {}),
    "labels": dict(owner="bentoml-team", type="prebuilt"),
    "metadata": model_config["metadata"],
    "requires_hf_tokens": requires_hf_tokens,
    "lock_python_packages": config.get("build", {}).get("lock_python_packages", True),
  }

  requirements = model_config.get("requirements", [])
  if len(requirements) > 0:
    context["requirements"] = requirements

  return context


def generate_readme(config, template_dir):
  models = [{"name": name, "engine_config": cfg.get("engine_config", {})} for name, cfg in config.items()]

  with open(template_dir / "README.md.tpl", "r") as f:
    template_content = f.read()

  rendered = Template(template_content).render(models=models)

  with open(template_dir / "README.md", "w") as f:
    f.write(rendered)


def compare_directories(dir1: Path, dir2: Path) -> bool:
  """Compare two directories recursively to check if they have the same content, respecting .gitignore."""
  if not dir1.exists() or not dir2.exists():
    return False

  # Use pathspec to parse .gitignore rules
  from pathspec import PathSpec
  from pathspec.patterns import GitWildMatchPattern

  # Read .gitignore if it exists
  gitignore_path = GIT_DIRECTORY / ".gitignore"
  ignore_patterns = []
  if gitignore_path.exists():
    with gitignore_path.open("r") as f:
      ignore_patterns = f.readlines()
  spec = PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)

  # Get files while respecting .gitignore
  def get_tracked_files(path: Path) -> list:
    return sorted([
      f.relative_to(path) for f in path.rglob("*") if f.is_file() and not spec.match_file(str(f.relative_to(path)))
    ])

  files1 = get_tracked_files(dir1)
  files2 = get_tracked_files(dir2)

  if files1 != files2:
    return False

  # Use filecmp for faster file comparison
  import filecmp

  for f1, f2 in zip(files1, files2):
    if not filecmp.cmp(dir1 / f1, dir2 / f2, shallow=False):
      return False

  return True


def generate_model(model_name: str, config: dict, template_dir: Path, progress: Progress, task_id: int) -> bool:
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
        # Compare the existing directory with the newly generated one
        if compare_directories(output_dir, temp_output_dir):
          progress.update(task_id, description=f"[yellow]Skipping {model_name} - no changes[/]")
          return True
        else:
          progress.update(task_id, description=f"[blue]Updating {model_name} - changes detected[/]")
          shutil.rmtree(output_dir)
          shutil.copytree(temp_output_dir, output_dir)
      else:
        # If directory doesn't exist, just move the generated one
        shutil.copytree(temp_output_dir, output_dir)

    progress.update(task_id, description=f"[green]✓ {model_name}[/]", completed=1)
    return True

  except Exception as e:
    progress.update(task_id, description=f"[red]✗ {model_name}: {str(e)}[/]", completed=1)
    return False


def generate_all_models(config: dict, template_dir: Path, force: bool = False) -> bool:
  """Generate all model projects in parallel."""
  console = Console()
  models = list(config.keys())
  success = True

  with Progress(
    SpinnerColumn(spinner_name="bouncingBar"),
    TextColumn("[progress.description]{task.description}"),
    console=console,
  ) as progress:
    overall_task = progress.add_task("[yellow]Generating models...[/]", total=len(models))
    gen_tasks = {model: progress.add_task(f"[cyan]Waiting to generate {model}...[/]", total=1) for model in models}

    for model_name in models:
      if force and (template_dir / model_name).exists():
        progress.update(gen_tasks[model_name], description=f"[blue]Removing existing {model_name}...[/]")
        shutil.rmtree(template_dir / model_name)

      result = generate_model(model_name, config, template_dir, progress, gen_tasks[model_name])
      success = success and result
      progress.advance(overall_task)

  return success


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate model service from config.yaml")
  parser.add_argument(
    "model_name", nargs="?", help="Specific model name to generate. If not provided, generates all models."
  )
  parser.add_argument("--force", action="store_true", help="Force regeneration even if directory exists")
  args = parser.parse_args()

  with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
  template_dir = Path(__file__).parent

  console = Console()
  if args.model_name:
    if args.model_name not in config:
      console.print(f"[red]Error: Model {args.model_name} not found in config.yaml[/]")
      return 1
    filtered_config = {args.model_name: config[args.model_name]}
  else:
    filtered_config = config

  success = generate_all_models(filtered_config, template_dir, args.force)

  # Generate README.md after all models are processed
  console.print("\n[yellow]Generating README.md...[/]")
  generate_readme(config, template_dir)
  console.print("[green]✓ Generated README.md[/]")

  # Format all python files
  console.print("\n[yellow]Formatting Python files...[/]")
  subprocess.run(
    [
      "ruff",
      "format",
      "--config",
      "indent-width=4",
      "--config",
      "line-length=119",
      "--config",
      "preview=true",
      "--exclude",
      "_src",
      "--exclude",
      "generate.py",
      "--exclude",
      "build.py",
      "--exclude",
      "push.py",
      "--exclude",
      "deploy.py",
      ".",
    ],
    check=True,
    capture_output=True,
  )
  subprocess.run(
    [
      "ruff",
      "format",
      "--config",
      "indent-width=2",
      "--config",
      "line-length=119",
      "--config",
      "preview=true",
      "generate.py",
      "build.py",
      "push.py",
      "deploy.py",
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
