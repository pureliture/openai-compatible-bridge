from pathlib import Path
import unittest

class ProductionOwner(unittest.TestCase):
    def test_github_keeps_pr_tests_without_production_authority(self):
        import yaml
        root=Path(__file__).resolve().parents[1]
        workflow=yaml.safe_load((root/'.github/workflows/build-publish.yml').read_text())
        self.assertEqual(set(workflow['jobs']),{'test'})
        self.assertNotIn('packages',workflow['permissions'])
        self.assertIn('pull_request',workflow.get('on',workflow.get(True)))
        self.assertNotIn('push',workflow.get('on',workflow.get(True)))

if __name__=='__main__':unittest.main()
