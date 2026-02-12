from __future__ import annotations

"""
Абстрактный интерфейс для управления маяком (лампочкой).

Конкретные реализации находятся в implementations.py.
"""

from abc import ABC, abstractmethod


class LighthouseController(ABC):
    """
    Абстрактный контроллер маяка.

    Управляет состоянием светового сигнала (зелёный/красный)
    в зависимости от превышения скорости объектов.
    """

    @abstractmethod
    def set_state(self, is_exceeded: bool) -> None:
        """
        Установить состояние маяка.

        Args:
            is_exceeded: True если скорость превышена (красный),
                        False если в норме (зелёный)
        """
        pass

    @abstractmethod
    def get_state(self) -> bool:
        """
        Получить текущее состояние маяка.

        Returns:
            True если маяк красный (превышение скорости),
            False если зелёный (в норме)
        """
        pass

    def cleanup(self) -> None:
        """
        Очистка ресурсов при завершении работы.

        По умолчанию ничего не делает, но может быть переопределён
        в конкретных реализациях (например, для GPIO).
        """
        pass
