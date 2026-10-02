"""Locate project-owned services without requiring hyphenated package names."""
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=32)
def project_path(project):
    catalog = json.loads((ROOT / 'catalog.json').read_text())
    match = next((item for item in catalog if item['id'] == project), None)
    if match is None:
        raise ValueError('Unknown project')
    return ROOT / match['category'] / project

@lru_cache(maxsize=64)
def load_component(project, component):
    if component not in ('service', 'advanced'):
        raise ValueError('Unknown component')
    path = project_path(project) / 'backend' / (component + '.py')
    spec = importlib.util.spec_from_file_location('portfolio_' + component + '_' + project.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_service(project):
    return load_component(project, 'service')

def metadata(project):
    return json.loads((project_path(project) / 'project.json').read_text())
