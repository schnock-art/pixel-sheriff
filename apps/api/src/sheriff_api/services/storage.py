from pathlib import Path


class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_uri: str) -> Path:
        root = self.root.resolve()
        candidate = (root / relative_uri).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Resolved path escapes storage root")
        return candidate

    def ensure_project_dirs(self, project_id: str) -> None:
        (self.root / "assets" / project_id).mkdir(parents=True, exist_ok=True)
        (self.root / "exports" / project_id).mkdir(parents=True, exist_ok=True)

    def write_bytes(self, relative_uri: str, content: bytes) -> Path:
        target = self.resolve(relative_uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target
