from __future__ import annotations

"""
Конкретные реализации LighthouseController.

- UILighthouseController: виртуальный маяк для веб-интерфейса
- GPIOLighthouseController: управление GPIO на Raspberry Pi (будущая реализация)
- USBRelayLighthouseController: управление USB-реле (будущая реализация)
"""

import threading
from typing import Optional

from src.lighthouse.controller import LighthouseController


class UILighthouseController(LighthouseController):
    """
    Виртуальный маяк для веб-интерфейса.

    Хранит состояние в памяти и используется для отображения
    индикатора в веб-интерфейсе через API /api/lamp.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_exceeded: bool = False

    def set_state(self, is_exceeded: bool) -> None:
        with self._lock:
            self._is_exceeded = bool(is_exceeded)

    def get_state(self) -> bool:
        with self._lock:
            return self._is_exceeded

    def cleanup(self) -> None:
        # Для UI-маяка очистка не требуется
        pass


class GPIOLighthouseController(LighthouseController):
    """
    Управление маяком через GPIO на Raspberry Pi.

    TODO: Реализовать управление GPIO портом для включения/выключения
    светодиода или реле, подключённого к маяку.
    """

    def __init__(self, gpio_pin: int = 18) -> None:
        """
        Инициализация GPIO контроллера.

        Args:
            gpio_pin: Номер GPIO пина для управления маяком
        """
        self._gpio_pin = gpio_pin
        self._is_exceeded = False
        # TODO: Инициализировать GPIO библиотеку (RPi.GPIO или gpiozero)

    def set_state(self, is_exceeded: bool) -> None:
        self._is_exceeded = bool(is_exceeded)
        # TODO: Установить состояние GPIO пина (HIGH/LOW)

    def get_state(self) -> bool:
        return self._is_exceeded

    def cleanup(self) -> None:
        # TODO: Освободить GPIO ресурсы
        pass


class USBRelayLighthouseController(LighthouseController):
    """
    Управление маяком через USB-реле.

    TODO: Реализовать управление USB-реле для включения/выключения
    маяка через USB-устройство.
    """

    def __init__(self, device_path: Optional[str] = None) -> None:
        """
        Инициализация USB-реле контроллера.

        Args:
            device_path: Путь к USB-устройству (если None, автоопределение)
        """
        self._device_path = device_path
        self._is_exceeded = False
        # TODO: Инициализировать подключение к USB-реле

    def set_state(self, is_exceeded: bool) -> None:
        self._is_exceeded = bool(is_exceeded)
        # TODO: Отправить команду включения/выключения реле

    def get_state(self) -> bool:
        return self._is_exceeded

    def cleanup(self) -> None:
        # TODO: Закрыть соединение с USB-устройством
        pass
