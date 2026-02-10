import pytest

import src.main as main_module


def test_main_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() при успешной загрузке конфига и запуске видеопотока завершается без SystemExit."""

    def fake_load_config():
        return {"camera": {"index": 0}}

    def fake_display_video_stream(*args, **kwargs):  # noqa: ARG001
        # Ничего не делаем, просто имитируем успешную отработку
        return None

    # Подменяем функции в модуле main
    monkeypatch.setattr(main_module, "load_config", fake_load_config)
    monkeypatch.setattr(main_module, "display_video_stream", fake_display_video_stream)

    # При успешном сценарии main не должен вызывать sys.exit
    main_module.main()


def test_main_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если load_config поднимает FileNotFoundError, main должен завершаться с кодом 1."""

    def fake_load_config():
        raise FileNotFoundError("no config")

    monkeypatch.setattr(main_module, "load_config", fake_load_config)

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1


def test_main_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если load_config поднимает ValueError, main должен завершаться с кодом 1."""

    def fake_load_config():
        raise ValueError("bad config")

    monkeypatch.setattr(main_module, "load_config", fake_load_config)

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1


def test_main_camera_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если display_video_stream поднимает CameraError, main должен завершаться с кодом 1."""

    from src.camera import CameraError

    def fake_load_config():
        return {"camera": {"index": 0}}

    def fake_display_video_stream(*args, **kwargs):  # noqa: ARG001
        raise CameraError("camera failed")

    monkeypatch.setattr(main_module, "load_config", fake_load_config)
    monkeypatch.setattr(main_module, "display_video_stream", fake_display_video_stream)

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1


def test_main_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если display_video_stream поднимает KeyboardInterrupt, main должен завершаться с кодом 0."""

    def fake_load_config():
        return {"camera": {"index": 0}}

    def fake_display_video_stream(*args, **kwargs):  # noqa: ARG001
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_module, "load_config", fake_load_config)
    monkeypatch.setattr(main_module, "display_video_stream", fake_display_video_stream)

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 0


def test_main_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если display_video_stream поднимает неожиданное исключение, main должен завершаться с кодом 1."""

    def fake_load_config():
        return {"camera": {"index": 0}}

    def fake_display_video_stream(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "load_config", fake_load_config)
    monkeypatch.setattr(main_module, "display_video_stream", fake_display_video_stream)

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1


def test_main_module_entrypoint() -> None:
    """Точка входа: в модуле main есть вызываемая функция main."""
    assert callable(getattr(main_module, "main"))

