# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cookiecutter",
#     "jinja2",
#     "pyyaml",
#     "ruff",
# ]
# ///
import yaml, shutil, subprocess, json, argparse
from pathlib import Path
from jinja2 import Template


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

  with open("config.yaml", "r") as f: config = yaml.safe_load(f)
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
    ".",
  ])
  return 0

if __name__ == "__main__": raise SystemExit(main())
