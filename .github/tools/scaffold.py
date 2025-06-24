from __future__ import annotations

import argparse, multiprocessing, pathlib, shutil, typing, yaml, tomllib, tomli_w, importlib.metadata


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


def scaffold_model(model_name: str, cfg: dict[str, typing.Any], git_dir: pathlib.Path, out_dir: pathlib.Path, force: bool) -> None:
  target_dir = out_dir / model_name
  if target_dir.exists():
    if not force:
      raise FileExistsError(f'Target directory {target_dir} already exists. Use --force to overwrite.')
    shutil.rmtree(target_dir)
  target_dir.mkdir(parents=True, exist_ok=True)

  files_to_copy = [
    '.bentoignore',
    '.python-version',
    '.gitignore',
    'service.py',
    'pyproject.toml',
    'requirements.txt',
    'README.md',
  ]

  for filename in files_to_copy:
    src = git_dir / filename
    if src.exists():
      shutil.copy2(src, target_dir / filename)

  templates_src = git_dir / 'templates'
  if templates_src.exists():
    shutil.copytree(templates_src, target_dir / 'templates', dirs_exist_ok=True)

  pyproject_path = target_dir / 'pyproject.toml'
  with pyproject_path.open('rb') as f:
    data = tomllib.load(f)
    bento_yaml = data.get('tool', {}).get('bentoml', {}).get('build', {})

  if 'envs' in cfg and not cfg['envs']:
    cfg.pop('envs')
  bento_yaml['args'] = cfg
  data['project']['name'] = model_name
  data['project']['version'] = importlib.metadata.version('bentoml')
  data['tool']['bentoml']['build'] = bento_yaml

  with pyproject_path.open('wb') as f:
    tomli_w.dump(data, f, indent=2)



def main() -> int:
  parser = argparse.ArgumentParser(description='Scaffold bento directories for models')
  parser.add_argument('model_names', nargs='*', help='Specific model names to scaffold')
  parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path('.'), help='Directory to place scaffolded model folders (default: current directory)')
  parser.add_argument('--force', action='store_true', help='Overwrite existing directories')
  parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count())
  args = parser.parse_args()

  git_dir = pathlib.Path(__file__).parent.parent.parent
  config = load_generated_config(git_dir)

  if args.model_names:
    invalid_models = [m for m in args.model_names if m not in config]
    if invalid_models:
      raise SystemExit(f'Invalid model names: {", ".join(invalid_models)}')
    filtered = {m: config[m] for m in args.model_names}
  else:
    filtered = config

  args.output_dir.mkdir(parents=True, exist_ok=True)

  from concurrent.futures import ThreadPoolExecutor, as_completed

  with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {
      executor.submit(scaffold_model, name, cfg, git_dir, args.output_dir, args.force): name
      for name, cfg in filtered.items()
    }
    for fut in as_completed(futures):
      model = futures[fut]
      try:
        fut.result()
        print(f'✓ Scaffolded {model}')
      except Exception as e:
        print(f'✗ Failed {model}: {e}')

  return 0


if __name__ == '__main__':
  raise SystemExit(main())
