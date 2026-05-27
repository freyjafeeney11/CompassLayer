from dataclasses import dataclass, field
from typing import List, Tuple
from core import i18n

@dataclass
class Setting:
    key: str
    value: float
    min_val: float
    max_val: float
    step: float
    unit_key: str = 'unit_none'
    is_lang: bool = False

    def increase(self) -> float:
        self.value = min(self.max_val, round(self.value + self.step, 2))
        return self.value

    def decrease(self) -> float:
        self.value = max(self.min_val, round(self.value - self.step, 2))
        return self.value

    def announce(self) -> str:
        name = i18n.get_text(self.key)
        if self.is_lang:
            val_str = i18n.get_text(f'lang_value_{int(self.value)}')
            return f'{name}, {val_str}'
        unit_str = i18n.get_text(self.unit_key)
        if unit_str:
            return f'{name}, {self.value} {unit_str}'
        return f'{name}, {self.value}'


class SettingsMenu:
    def __init__(self):
        self.items: List[Setting] = [
            Setting(key='language', value=0.0, min_val=0.0, max_val=1.0, step=1.0, is_lang=True),
            Setting(key='ping_rate', value=1.0, min_val=0.5, max_val=2.0, step=0.25, unit_key='unit_times'),
            Setting(key='ping_volume', value=1.0, min_val=0.1, max_val=2.0, step=0.1, unit_key='unit_times'),
            Setting(key='tts_volume', value=60, min_val=10, max_val=100, step=10),
            Setting(key='tts_speed', value=0, min_val=-5, max_val=5, step=1),
        ]
        self.index: int = 0
        self.active: bool = False

    @property
    def current(self) -> Setting:
        return self.items[self.index]

    @property
    def language_code(self) -> str:
        return 'en' if self.items[0].value == 1.0 else 'fr'

    @property
    def pulse_rate(self) -> float:
        return self.items[1].value

    @property
    def ping_volume(self) -> float:
        return self.items[2].value

    @property
    def tts_volume(self) -> int:
        return int(self.items[3].value)

    @property
    def tts_speed(self) -> int:
        return int(self.items[4].value)

    def next_item(self) -> Setting:
        self.index = (self.index + 1) % len(self.items)
        return self.current

    def prev_item(self) -> Setting:
        self.index = (self.index - 1) % len(self.items)
        return self.current
