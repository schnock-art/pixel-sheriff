from pathlib import Path


class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_uri: str) -> Path:
        return self.root / relative_uri

    def ensure_project_dirs(self, project_id: str) -> None:
        (self.root / "assets" / project_id).mkdir(parents=True, exist_ok=True)
        (self.root / "exports" / project_id).mkdir(parents=True, exist_ok=True)
