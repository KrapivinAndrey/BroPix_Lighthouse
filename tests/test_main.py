"""
Тесты для модуля main.

Этот файл использует unittest, но не обращается к реальной камере,
так как функции, работающие с камерой, замоканы.
"""

import unittest
from unittest.mock import patch

import src.main as main_module


class TestMain(unittest.TestCase):
    """Тесты для функции main в стиле unittest."""

    def test_main_does_not_touch_real_camera(self):
        """main() должен завершаться без обращения к реальной камере."""

        def fake_load_config():
            return {"camera": {"index": 0}}

        def fake_display_video_stream(*args, **kwargs):  # noqa: ARG001
            return None

        with patch.object(main_module, "load_config", fake_load_config), patch.object(
            main_module,
            "display_video_stream",
            fake_display_video_stream,
        ):
            # Если бы main обращался к реальной камере, тест завис бы или упал.
            main_module.main()


if __name__ == "__main__":
    unittest.main()
