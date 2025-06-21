from __future__ import annotations

import subprocess, traceback, os, argparse, multiprocessing, pathlib, dataclasses, typing, yaml, tomllib, tempfile, shutil, tomli_w, importlib.metadata
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


@dataclasses.dataclass
class BuildResult:
  model_name: str
  bento_tag: str
  success: bool
  error: str = ""
  stderr: str = ''
  stdout: str = ''


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


def build_model(model_name: str, cfg: dict, git_dir: pathlib.Path) -> BuildResult:
  pyproject_path = git_dir / 'pyproject.toml'
  with pyproject_path.open('rb') as f:
    base_pyproject = tomllib.load(f)

  with tempfile.TemporaryDirectory(suffix='bentovlllm') as tempdir:
    td = pathlib.Path(tempdir)

    files_to_copy = [
      '.bentoignore',
      '.python-version',
      '.gitignore',
      'service.py',
      'pyproject.toml',
      'requirements.txt',
    ]

    for filename in files_to_copy:
      src = git_dir / filename
      if src.exists():
        shutil.copy2(src, td / filename)

    templates_src = git_dir / 'templates'
    if templates_src.exists():
      shutil.copytree(templates_src, td / 'templates', dirs_exist_ok=True)

    with (td / 'pyproject.toml').open('rb') as s:
      data = tomllib.load(s)
      bento_yaml = data.get('tool', {}).get('bentoml', {}).get('build', {})

    with (td / 'pyproject.toml').open('wb') as s:
      if 'envs' in cfg and not cfg['envs']:
        cfg.pop('envs')
      bento_yaml['args'] = cfg
      data['project']['name'] = model_name
      data['project']['version'] = importlib.metadata.version('bentoml')
      data['tool']['bentoml']['build'] = bento_yaml
      tomli_w.dump(data, s, indent=2)

    try:
      result = subprocess.run(
        ['bentoml', 'build', '--output', 'tag'],
        capture_output=True,
        text=True,
        check=True,
        cwd=tempdir,
        env=os.environ,
      )

      output = result.stdout.strip()
      if output.startswith('__tag__:'):
        bento_tag = output[8:]
        return BuildResult(model_name, bento_tag, True)

      return BuildResult(model_name, '', False, f'Unexpected output format: {output}')

    except subprocess.CalledProcessError as e:
      traceback.print_exc()
      return BuildResult(model_name, '', False, stderr=e.stderr, stdout=e.stdout)
    except Exception as e:
      traceback.print_exc()
      return BuildResult(model_name, '', False, str(e))


def build_bentos(config: dict[str, typing.Any], git_dir: pathlib.Path, workers: int) -> list[BuildResult]:
  console = Console()
  results = []
  num_models = len(config)

  with Progress(
    SpinnerColumn(spinner_name='bouncingBar'), TextColumn('[progress.description]{task.description}'), console=console
  ) as progress:
    overall_task = progress.add_task(f'[yellow]Building {num_models} bentos...[/]', total=num_models)

    with ThreadPoolExecutor(max_workers=workers) as executor:
      future_to_model = {executor.submit(build_model, model, cfg, git_dir): model for model, cfg in config.items()}

      built_count = 0
      for future in as_completed(future_to_model):
        result = future.result()
        results.append(result)
        built_count += 1
        model_name = result.model_name

        # Update progress bar description to show general progress
        progress.update(
          overall_task, description=f'[blue]({built_count}/{num_models}) Processing models...[/]', advance=1
        )

        if result.success:
          console.print(f'  [green]✓ Built {result.bento_tag}[/]')
        else:
          console.print(f'  [red]✗ Failed {model_name}: {result.stderr}[/]')

    progress.update(overall_task, description=f'[bold green]Finished processing {num_models} models.[/]')

  console.print()
  return results


def main() -> int:
  parser = argparse.ArgumentParser(description='Build all model bentos in parallel')
  parser.add_argument(
    'model_names', nargs='*', help='Specific model names to build. If not provided, builds all models.'
  )
  parser.add_argument(
    '--workers',
    type=int,
    default=multiprocessing.cpu_count(),
    help=f'Number of parallel workers (default: {multiprocessing.cpu_count()})',
  )
  parser.add_argument(
    '--output-name', type=str, default='successful_builds.txt', help='Filename to store successful bento tags'
  )
  args = parser.parse_args()

  git_dir = pathlib.Path(__file__).parent.parent.parent
  config = load_generated_config(git_dir)

  console = Console()

  if args.model_names:
    invalid_models = [model for model in args.model_names if model not in config]
    if invalid_models:
      console.print(f'[red]Error: Models not found in config.yaml: {", ".join(invalid_models)}[/]')
      return 1
    filtered_config = {model: config[model] for model in args.model_names}
  else:
    filtered_config = config

  console.print()
  console.print(f'[bold]Building {len(filtered_config)} bentos with {args.workers} workers[/]')

  results = build_bentos(filtered_config, git_dir, args.workers)
  successful_builds = [r for r in results if r.success]

  # Print summary
  console.print('\n[bold]Build Summary:[/]')
  console.print(f'Total bentos: {len(filtered_config)}')
  console.print(f'Successful builds: {len(successful_builds)}')
  failed_builds = len(results) - len(successful_builds)
  console.print(f'Failed builds: {failed_builds}')

  # Print details for failed builds
  if failed_builds > 0:
    console.print('\n[bold red]Failed Build Details:[/]')
    for r in results:
      if not r.success:
        console.print(f'- {r.model_name}:\n{r.stderr}\n{r.stdout}')

  # Save successful bento tags to file for later use
  if successful_builds:
    bento_tags = [r.bento_tag for r in successful_builds]
    with open(git_dir / args.output_name, 'w') as f:
      f.write('\n'.join(bento_tags))
    console.print(f'\n[green]Saved successful build tags to {args.output_name}[/]')

  return 0 if len(successful_builds) == len(filtered_config) else 1


if __name__ == '__main__':
  raise SystemExit(main())
