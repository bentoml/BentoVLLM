from __future__ import annotations

import yaml, json, argparse, multiprocessing, pathlib, typing
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from scaffold import scaffold_model


def update_model_descriptions(config, git_dir):
  certified_repos_path = git_dir / 'bentocloud-homepage-news' / 'certified-bento-repositories.json'
  if not certified_repos_path.exists():
    print('Warning: certified-bento-repositories.json not found, skipping description updates')
    return

  with certified_repos_path.open('r') as f:
    certified_repos_struct = json.load(f)
    certified_bentos = certified_repos_struct['certified_bentos']

  # Create a mapping of model names to their certified repo data
  certified_list = [repo['repo_name'] for repo in certified_bentos]

  # Update descriptions for each model
  for model_name, model_config in config.items():
    if 'metadata' not in model_config:
      continue

    repo_name = model_name.lower()

    if repo_name in certified_list:
      continue

    metadata = model_config['metadata']

    labels = ['✍️ Text Generation']
    if metadata.get('vision', False):
      labels.append('👁️ Image-to-Text')

    image_url = f'https://raw.githubusercontent.com/bentoml/bentocloud-homepage-news/main/imgs/{metadata["provider"].lower().replace(" ", "-")}.png'
    bentos = dict(
      org_name='bentoml',
      repo_name=repo_name,
      description=dict(
        name=metadata['description'],
        text=f"{metadata['description']} developed by {metadata['provider']} and served using vLLM and BentoML. It offers capabilities for streaming and compatibility with OpenAI's API",
        link=f'github.com/bentoml/BentoVLLM/tree/main/{model_name}.yaml',
        image=image_url,
        label=labels,
      ),
    )
    certified_bentos.append(bentos)

  with certified_repos_path.open('w', encoding='utf-8') as f:
    f.write(json.dumps(dict(certified_bentos=certified_bentos), indent=2, ensure_ascii=False))


def load_generated_config(git_dir: pathlib.Path) -> dict[str, dict[str, typing.Any]]:
  config: dict[str, dict[str, typing.Any]] = {}
  for yaml_path in git_dir.glob('*.yaml'):
    if not yaml_path.is_file():
      continue
    try:
      with yaml_path.open('r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f)
      if not raw_data or 'args' not in raw_data:
        continue
      args_struct = raw_data['args']
      model_name = yaml_path.stem
      if not model_name:
        continue
      config[model_name] = args_struct
    except yaml.YAMLError:
      continue
  return config


def main() -> int:
  parser = argparse.ArgumentParser(description='Generate model service from config.yaml')
  parser.add_argument(
    'model_names', nargs='*', help='Specific model names to generate. If not provided, generates all models.'
  )
  parser.add_argument('--force', action='store_true', help='Force regeneration even if directory exists')
  parser.add_argument(
    '--workers',
    type=int,
    default=multiprocessing.cpu_count(),
    help=f'Number of parallel workers (default: {multiprocessing.cpu_count()})',
  )
  parser.add_argument(
    '--skip-readme-nightly',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='whether to skip generating nightly readme.',
  )
  parser.add_argument('--output-dir', type=pathlib.Path, help='Directory to place scaffolded model folders')
  args = parser.parse_args()

  git_dir = pathlib.Path(__file__).parent.parent.parent
  config = load_generated_config(git_dir / ".github")

  console = Console()

  if args.model_names:
    invalid_models = [model for model in args.model_names if model not in config]
    if invalid_models:
      console.print(f'[red]Error: Models not found in config.yaml: {", ".join(invalid_models)}[/]')
      return 1
    filtered_config = {model: config[model] for model in args.model_names}
  else:
    filtered_config = config

  # Update model descriptions
  console.print('\n[yellow]Updating model descriptions...[/]')
  update_model_descriptions(filtered_config, git_dir)
  console.print('[green]✓ Updated model descriptions[/]')

  # Scaffold model directories
  console.print('\n[yellow]Scaffolding model directories...[/]')

  num_models = len(filtered_config)

  with Progress(
    SpinnerColumn(spinner_name='bouncingBar'), TextColumn('[progress.description]{task.description}'), console=console
  ) as progress:
    overall = progress.add_task(f'[yellow]Scaffolding {num_models} models...[/]', total=num_models)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
      futures = {
        executor.submit(scaffold_model, name, cfg, git_dir, args.output_dir if args.output_dir else git_dir, args.force): name
        for name, cfg in filtered_config.items()
      }
      for fut in as_completed(futures):
        model_name = futures[fut]
        try:
          fut.result()
          console.print(f'  [green]✓ Scaffolded {model_name}[/]')
        except Exception as e:
          console.print(f'  [red]✗ Failed scaffold {model_name}: {e}[/]')
        finally:
          progress.update(overall, advance=1)

  console.print('[green]✓ Finished scaffolding[/]')

  return 0


if __name__ == '__main__':
  raise SystemExit(main())
