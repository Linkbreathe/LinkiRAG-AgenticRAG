from linki.config import Settings, load_settings


def test_defaults_have_single_default_kb(tmp_path):
    s = load_settings(root=tmp_path)
    assert s.default_kb.name == "default"
    assert s.qdrant_path.is_absolute()
    assert s.max_rounds == 2 and s.max_attempts == 2


def test_kb_resolution_by_name_and_tool_name():
    s = Settings()
    kb = s.kb("Retrieve_default")
    assert kb is not None and kb.name == "default"
    assert s.kb("default").tool_name == "Retrieve_default"
    assert s.kb("nope") is None


def test_yaml_overrides(tmp_path):
    (tmp_path / "linki.yaml").write_text(
        "max_rounds: 3\nknowledge_bases:\n  - name: api\n    title: API docs\n    usage_hint: query API stuff\n",
        encoding="utf-8",
    )
    s = load_settings(root=tmp_path)
    assert s.max_rounds == 3
    assert s.default_kb.name == "api"
    assert s.kb("Retrieve_api").collection == "kb_api"
