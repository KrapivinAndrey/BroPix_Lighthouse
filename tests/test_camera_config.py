import json
from pathlib import Path

import pytest

from src.camera import load_config


def test_load_config_returns_camera_section(tmp_path: Path) -> None:
    """Успешная загрузка конфига с секцией camera и полем index."""
    config_path = tmp_path / "config.json"
    data = {
        "camera": {
            "index": 1,
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
        "other": {"foo": "bar"},
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")

    config = load_config(config_path=config_path)

    assert "camera" in config
    camera_cfg = config["camera"]
    assert camera_cfg["index"] == 1
    assert camera_cfg["width"] == 1280
    assert camera_cfg["height"] == 720
    assert camera_cfg["fps"] == 30


def test_load_config_raises_file_not_found(tmp_path: Path) -> None:
    """Если файла нет, должен подниматься FileNotFoundError."""
    missing_path = tmp_path / "missing_config.json"

    with pytest.raises(FileNotFoundError):
        load_config(config_path=missing_path)


def test_load_config_invalid_json(tmp_path: Path) -> None:
    """Некорректный JSON приводит к ValueError с описанием проблемы."""
    config_path = tmp_path / "config.json"
    # Нарочно записываем битый JSON
    config_path.write_text("{invalid_json: true", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_config(config_path=config_path)

    # Сообщение должно указывать на проблему с JSON
    assert "Invalid JSON in configuration file" in str(exc_info.value)


def test_load_config_missing_camera_section(tmp_path: Path) -> None:
    """Если нет секции camera, должна быть ошибка ValueError."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"not_camera": {}}), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_config(config_path=config_path)

    assert "must contain 'camera' section" in str(exc_info.value)


def test_load_config_missing_camera_index(tmp_path: Path) -> None:
    """Если нет camera.index, должна быть ошибка ValueError."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"camera": {}}), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_config(config_path=config_path)

    assert "must contain 'index' field" in str(exc_info.value)

