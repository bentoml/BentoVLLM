# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cookiecutter",
#     "jinja2",
#     "pyyaml",
#     "ruff",
# ]
# ///
import yaml, shutil, subprocess, json, argparse, typing
from pathlib import Path
from jinja2 import Template


def load_config():
  with open("config.yaml", "r") as f:
    return yaml.safe_load(f)


def update_model_descriptions(config: dict[str, dict[str, typing.Any]], template_dir):
  certified_repos_path = template_dir / "bentocloud-homepage-news" / "certified-bento-repositories.json"
  if not certified_repos_path.exists():
    print("Warning: certified-bento-repositories.json not found, skipping description updates")
    return

  with certified_repos_path.open("r") as f:
    certified_repos_struct = json.load(f)
    certified_bentos = certified_repos_struct["certified_bentos"]

  # Create a mapping of model names to their certified repo data
  certified_list = [repo["repo_name"] for repo in certified_bentos]

  image_url = "https://raw.githubusercontent.com/bentoml/bentocloud-homepage-news/main/imgs/llama3-8b.png"

  # Update descriptions for each model
  for model_name, model_config in config.items():
    repo_name = f"bentovllm-{model_name}-service"

    # handle aliases
    if model_name.startswith("deepseek-r1-distill"): repo_name = f"bentovllm-r1{model_name.removeprefix('deepseek-r1-distill')}-service"

    if repo_name in certified_list:
      continue

    labels = ["✍️ Text Generation"]
    if model_config.get("vision", False):
      labels.append("👁️ Image-to-Text")

    metadata = model_config["metadata"]
    bentos = dict(
      org_name="bentoml",
      repo_name=repo_name,
      description=dict(
        name=metadata["description"],
        text=f"{metadata['description']} developed by {metadata['provider']} and served using vLLM and BentoML. It offers capabilities for streaming and compatibility with OpenAI's API",
        link="https://github.com/bentoml/BentoVLLM",
        image=image_url,
        label=labels,
      ),
    )
    certified_bentos.append(bentos)

  with certified_repos_path.open("w", encoding="utf-8") as f:
    json_str = json.dumps(dict(certified_bentos=certified_bentos), indent=2, ensure_ascii=False)
    f.write(json_str)
  print("Updated prebuilt bentos list")


def generate_cookiecutter_context(model_name, config):
  model_config = config[model_name]
  engine_config_struct = model_config.get("engine_config", {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"})

  # Convert configs to YAML strings to preserve types
  return {
    "model_name": model_name,
    "model_id": engine_config_struct["model"],
    "vision": str(model_config.get("vision", False)).lower(),
    "service_config": json.dumps(model_config.get("service_config", {})),
    "engine_config": json.dumps(engine_config_struct),
    "server_config": json.dumps(model_config.get("server_config", {})),
    "requirements": json.dumps(model_config.get("requirements", [])),
    "engine_config_struct": engine_config_struct,
  }


def generate_readme(config, template_dir):
  # Prepare model data for the template
  models = [{"name": name} for name, cfg in config.items()]

  # Read the template
  with open(template_dir / "README.md.tpl", "r") as f:
    template_content = f.read()

  # Render the template
  template = Template(template_content)
  rendered = template.render(models=models)

  # Write the rendered README
  with open(template_dir / "README.md", "w") as f:
    f.write(rendered)
  print("Generated README.md")


def generate_model(model_name, config, template_dir, force=False):
  output_dir = template_dir / model_name
  if output_dir.exists() and not force:
    print(f"Skipping {model_name} - directory already exists (use --force to override)")
    return
  if output_dir.exists() and force:
    print(f"Removing existing directory {output_dir}...")
    shutil.rmtree(output_dir)

  print(f"Generating project for {model_name}...")
  context = generate_cookiecutter_context(model_name, config)

  config_path = template_dir / "cookiecutter.json"
  with open(config_path, "w") as f:
    json.dump(context, f, indent=2)

  # Run cookiecutter with the config
  subprocess.run([
    "cookiecutter",
    str(template_dir),
    "--no-input",
    "--config-file",
    str(config_path),
    "--output-dir",
    str(template_dir),
  ])

  config_path.unlink()
  print(f"Generated project for {model_name}")


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

  if args.model_name:
    if args.model_name not in config:
      print(f"Error: Model {args.model_name} not found in config.yaml")
      return 1
    generate_model(args.model_name, config, template_dir, args.force)
  else:
    for model_name in config:
      generate_model(model_name, config, template_dir, args.force)

  # Generate README.md after all models are processed
  generate_readme(config, template_dir)

  # Format all python files except cookiecutter template
  subprocess.run([
    "ruff",
    "format",
    "--config",
    "indent-width=4",
    "--config",
    "line-length=119",
    "--config",
    "preview=true",
    "--exclude",
    "\\{\\{cookiecutter.*\\}\\}",
    "--exclude",
    "generate.py",
    "--exclude",
    "build.py",
    "--exclude",
    "push.py",
    ".",
  ])

  # Update model descriptions before generation
  update_model_descriptions(config, template_dir)

  return 0


if __name__ == "__main__": raise SystemExit(main())
