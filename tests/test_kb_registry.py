from linki.config import Settings
from linki.core.kb_registry import KnowledgeBaseRegistry


def test_runtime_topic_registry_persists(tmp_path):
    settings = Settings(data_dir=tmp_path)
    registry = KnowledgeBaseRegistry(settings)

    kb = registry.create("Customer Success", "Support and renewal documents")

    assert kb.name == "customer_success"
    assert kb.collection == "kb_customer_success"
    assert registry.get("Retrieve_customer_success").title == "Customer Success"

    reloaded = KnowledgeBaseRegistry(settings)
    assert reloaded.get("customer_success").usage_hint == "Support and renewal documents"


def test_runtime_topic_slug_fallback_for_non_ascii(tmp_path):
    registry = KnowledgeBaseRegistry(Settings(data_dir=tmp_path))

    kb = registry.create("财务资料")

    assert kb.name.startswith("topic_")
    assert kb.collection.startswith("kb_topic_")
