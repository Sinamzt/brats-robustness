"""Download an explicit Hugging Face bundle commit, never an implicit latest release."""
from pathlib import Path


def ensure_bundle(bundle_path, name, revision):
    path = Path(bundle_path)
    required = ['models/model.pt', 'configs/inference.json', 'configs/metadata.json']
    if not all((path / relative).exists() for relative in required):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=f'MONAI/{name}', revision=revision,
                          local_dir=str(path),
                          allow_patterns=['configs/*', 'models/model.pt', 'docs/README.md', 'LICENSE'])
    return path
