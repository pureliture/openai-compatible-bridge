from pathlib import Path
import unittest

class ProductionBuildPortability(unittest.TestCase):
    def test_build_does_not_require_buildkit_cache_mount(self):
        dockerfile = (Path(__file__).resolve().parents[1] / 'Dockerfile').read_text()
        self.assertNotIn('RUN --mount=', dockerfile)
        self.assertIn('uv sync --frozen --no-dev --no-editable', dockerfile)

    def test_startup_uses_already_installed_environment(self):
        dockerfile = (Path(__file__).resolve().parents[1] / 'Dockerfile').read_text()
        self.assertIn('CMD ["/app/.venv/bin/uvicorn",', dockerfile)
        self.assertNotIn('CMD ["uv", "run",', dockerfile)
