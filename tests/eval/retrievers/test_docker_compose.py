from pathlib import Path

import yaml


def test_docker_compose_defines_qdrant_service():
    path = Path("docker-compose.yml")
    assert path.exists(), "docker-compose.yml must exist at repo root"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    services = data.get("services", {})
    assert "qdrant" in services, "compose must define a 'qdrant' service"
    qdrant = services["qdrant"]
    image = qdrant["image"]
    assert image.startswith("qdrant/qdrant:"), "must use the qdrant image"
    assert not image.endswith(":latest"), "must pin a qdrant image version"
    ports = qdrant.get("ports", [])
    assert any(str(p).startswith("6333:6333") for p in ports), "must expose port 6333"
    volumes = qdrant.get("volumes", [])
    assert any(str(v).endswith(":/qdrant/storage") for v in volumes), "must persist /qdrant/storage"
