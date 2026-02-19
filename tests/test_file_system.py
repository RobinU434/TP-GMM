import os

import yaml
import pytest

from tpgmm.utils.file_system import load_txt, load_yaml, write_yaml


@pytest.fixture
def tmp_yaml(tmp_path):
    """Provide a temporary YAML file path."""
    return str(tmp_path / "test.yaml")


@pytest.fixture
def tmp_txt(tmp_path):
    """Provide a temporary text file path."""
    return str(tmp_path / "test.txt")


class TestLoadYaml:
    def test_loads_yaml_file(self, tmp_yaml):
        content = {"key1": "value1", "key2": "value2"}
        with open(tmp_yaml, "w") as f:
            yaml.dump(content, f)
        result = load_yaml(tmp_yaml)
        assert result == content


class TestLoadTxt:
    def test_loads_text_lines(self, tmp_txt):
        content = ["line1", "line2", "line3"]
        with open(tmp_txt, "w") as f:
            for line in content:
                f.write(line + "\n")
        result = load_txt(tmp_txt)
        assert result == content


class TestWriteYaml:
    def test_writes_yaml_file(self, tmp_yaml):
        content = {"key1": "value1", "key2": "value2"}
        write_yaml(tmp_yaml, content)
        with open(tmp_yaml, "r") as f:
            result = yaml.safe_load(f)
        assert result == content