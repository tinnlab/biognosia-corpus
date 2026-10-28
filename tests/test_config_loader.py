from pathlib import Path

from config_loader import PROJECT_ROOT, load_config


def test_expands_defaults_and_references(tmp_path, monkeypatch):
    conf = tmp_path / "test.conf"
    conf.write_text(
        '# comment\nDATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"\nFILE="$DATA_DIR/a.gz"\nSECRET="${TEST_SECRET:-}"\n'
    )
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("TEST_SECRET", raising=False)
    config = load_config(str(conf))
    assert config["DATA_DIR"] == f"{PROJECT_ROOT}/data"
    assert config["FILE"] == f"{PROJECT_ROOT}/data/a.gz"
    assert config["SECRET"] == ""


def test_environment_overrides_default(tmp_path, monkeypatch):
    conf = tmp_path / "test.conf"
    conf.write_text('DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"\nFILE="$DATA_DIR/a.gz"\n')
    monkeypatch.setenv("DATA_DIR", "/mnt/store")
    assert load_config(str(conf))["FILE"] == "/mnt/store/a.gz"


def test_repository_configs_contain_no_absolute_paths_or_secrets():
    for name in ("pipeline.conf", "db_init.conf"):
        text = (Path(PROJECT_ROOT) / "config" / name).read_text()
        assert "/data/pbya" not in text
        assert "hf_" not in text
        assert "gsk_" not in text
