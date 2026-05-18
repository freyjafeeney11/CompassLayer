import pyo
import numpy as np
import time
import threading
import os
from dataclasses import dataclass
from typing import Optional
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print('pyttsx3 not found — TTS disabled. Install with: pip install pyttsx3')
    TTS_AVAILABLE = False

@dataclass
class NavIcon:
    icon_type: str
    direction: str
    offset: float
    distance_m: Optional[float]

def from_algo_data(raw: dict) -> Optional[NavIcon]:
    dist_str = raw.get('distance', 'N/A')
    distance: Optional[float] = None
    if dist_str and dist_str != 'N/A':
        try:
            parsed = float(dist_str.replace('m', '').strip())
            if parsed > 0.0:
                distance = parsed
        except ValueError:
            pass
    direction = raw.get('direction', 'Straight').lower()
    offset = abs(raw.get('rel_offset', 0.0))
    if direction == 'straight':
        direction = 'center'
    label = raw.get('label', 'main_quest')
    if label not in ('main_quest', 'treasure', 'stockpile'):
        label = 'main_quest'
    return NavIcon(icon_type=label, direction=direction, offset=offset, distance_m=distance)

def from_algo_batch(rows: list[dict]) -> list[NavIcon]:
    return [icon for icon in (from_algo_data(r) for r in rows) if icon is not None]

def generate_test_scenario(scenario: str='approaching') -> list[NavIcon]:
    scenarios = {'approaching': [NavIcon('main_quest', 'center', 0.05, 27.0)], 'veer_left': [NavIcon('main_quest', 'left', 0.6, 80.0)], 'veer_right': [NavIcon('main_quest', 'right', 0.7, 95.0)], 'treasure_nearby': [NavIcon('main_quest', 'right', 0.1, 45.0), NavIcon('treasure', 'left', 0.3, 18.0)], 'multi': [NavIcon('main_quest', 'left', 0.4, 60.0), NavIcon('treasure', 'right', 0.5, 22.0)], 'stockpile_close': [NavIcon('main_quest', 'center', 0.05, 40.0), NavIcon('stockpile', 'right', 0.35, 12.0)], 'stockpile_far': [NavIcon('main_quest', 'left', 0.3, 55.0), NavIcon('stockpile', 'right', 0.6, 80.0)], 'all_icons': [NavIcon('main_quest', 'center', 0.05, 30.0), NavIcon('treasure', 'left', 0.4, 20.0), NavIcon('stockpile', 'right', 0.5, 14.0)], 'label_missing': [NavIcon('main_quest', 'right', 0.6, None)], 'zero_dist': [NavIcon('main_quest', 'center', 0.0, 0.0)]}
    return scenarios.get(scenario, scenarios['approaching'])

class TTSEngine:

    def __init__(self):
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active = False
        self._backend = None
        self._fr_voice_token = None
        try:
            import win32com.client
            self._backend = 'win32com'
            self._active = True
            try:
                voices = win32com.client.Dispatch('SAPI.SpVoice')
                token_enum = voices.GetVoices()
                FR_HINTS = ('hortense', 'paul', ' fr', 'french', 'français')
                for i in range(token_enum.Count):
                    token = token_enum.Item(i)
                    desc = token.GetDescription().lower()
                    if any((hint in desc for hint in FR_HINTS)):
                        self._fr_voice_token = token
                        print(f'  [+] TTS voice: {token.GetDescription()}')
                        break
                if self._fr_voice_token is None:
                    print("  [i]  No French SAPI5 voice found — using system default. Install 'Microsoft Hortense Desktop' from Windows language settings.")
            except Exception:
                pass
            print('  [+] TTS engine ready (win32com / SAPI5)')
            return
        except ImportError:
            pass
        if TTS_AVAILABLE:
            try:
                import pyttsx3
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate', 200)
                engine.setProperty('volume', 0.95)
                voices = engine.getProperty('voices')
                FR_HINTS = ('hortense', 'paul', 'fr_', 'french', 'français')
                for v in voices:
                    if any((hint in v.name.lower() or hint in v.id.lower() for hint in FR_HINTS)):
                        engine.setProperty('voice', v.id)
                        print(f'  [+] TTS voice: {v.name}')
                        break
                else:
                    print('  [i]  No French voice found in pyttsx3 — using default.')
                self._pyttsx3_engine = engine
                self._backend = 'pyttsx3'
                self._active = True
                print('  [+] TTS engine ready (pyttsx3 fallback)')
            except Exception as e:
                print(f'  [!]  TTS init failed: {e}')
        else:
            print('  No TTS backend available (install pywin32 or pyttsx3)')

    def speak(self, text: str) -> None:
        if not self._active:
            return
        with self._lock:
            self._queue.append(text)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._drain_queue, daemon=True)
            self._thread.start()

    def speak_icon(self, icon: 'NavIcon') -> None:
        self.speak(_build_tts_phrase(icon))

    def wait(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def stop(self) -> None:
        with self._lock:
            self._queue.clear()

    def _drain_queue(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    break
                phrase = self._queue.pop(0)
            try:
                if self._backend == 'win32com':
                    self._speak_win32(phrase)
                else:
                    self._speak_pyttsx3(phrase)
            except Exception as e:
                print(f'  TTS error: {e}')

    def _speak_win32(self, phrase: str) -> None:
        import win32com.client
        voice = win32com.client.Dispatch('SAPI.SpVoice')
        if self._fr_voice_token is not None:
            voice.Voice = self._fr_voice_token
        voice.Rate = 0
        voice.Volume = 60
        voice.Speak(phrase, 0)

    def _speak_pyttsx3(self, phrase: str) -> None:
        self._pyttsx3_engine.say(phrase)
        self._pyttsx3_engine.runAndWait()

def _build_tts_phrase(icon: NavIcon) -> str:
    if icon.icon_type == 'main_quest':
        label = 'Objectif principal'
    elif icon.icon_type == 'treasure':
        label = 'Trésor'
    elif icon.icon_type == 'stockpile':
        label = 'Réserve'
    else:
        label = icon.icon_type.replace('_', ' ')
    if icon.direction == 'center' or icon.offset < 0.05:
        position = 'droit devant'
    else:
        side = 'gauche' if icon.direction == 'left' else 'droite'
        if icon.offset > 0.55:
            position = f'loin à {side}'
        elif icon.offset > 0.25:
            position = f'à {side}'
        else:
            position = f'légèrement à {side}'
    dist = int(icon.distance_m) if icon.distance_m is not None else 0
    distance = f"à {dist} mètre{('s' if dist != 1 else '')}"
    suffix = ', contient du bois, des cultures ou des minéraux' if icon.icon_type == 'stockpile' else ''
    return f'{label}, {position}, {distance}{suffix}'

class AudioEngine:
    SAMPLE_RATE = 44100
    VOLUME_QUEST = 0.8
    VOLUME_TREASURE = 2.5
    VOLUME_STOCKPILE = 0.15

    def __init__(self):
        self.server = pyo.Server(audio='portaudio', sr=self.SAMPLE_RATE, nchnls=2, duplex=0).boot()
        self.server.start()
        from config import resource_path
        audio_dir = resource_path(os.path.join('assets', 'audio'))
        self.quest = self._load(os.path.join(audio_dir, 'koto_note.wav'), self.VOLUME_QUEST)
        self.treasure = self._load(os.path.join(audio_dir, 'koto_trill.wav'), self.VOLUME_TREASURE)
        self.stockpile = self._load(os.path.join(audio_dir, 'koto_stockpile.wav'), self.VOLUME_STOCKPILE)
        self.tts = TTSEngine()
        self._pulse_active = False
        self._pulse_thread: Optional[threading.Thread] = None
        self._current_icon: Optional[NavIcon] = None
        self._lock = threading.Lock()

    def shutdown(self):
        self.server.stop()
        self.server.shutdown()

    def _load(self, filename: str, volume: float) -> Optional[dict]:
        if not os.path.exists(filename):
            print(f'  [!] Missing audio file: {filename}')
            return None
        sf = pyo.SfPlayer(filename, loop=False)
        sf.stop()
        sf_mono = sf.mix(1)
        amp = pyo.Sig(volume)
        pan = pyo.Sig(0.5)
        panned = pyo.Pan(sf_mono * amp, outs=2, pan=pan).out()
        print(f'  [+] Loaded {filename}')
        return {'sf': sf, 'pan': pan, 'panned': panned, 'amp': amp, 'vol': volume}

    def _play_sound(self, player: Optional[dict], icon: NavIcon, volume_override: Optional[float]=None, is_center: bool=False):
        if not player:
            return
        clamped = min(1.0, icon.offset)
        if icon.direction == 'center':
            pan_pos = 0.5
        elif icon.direction == 'right':
            pan_pos = 0.5 + clamped * 0.5
        else:
            pan_pos = 0.5 - clamped * 0.5
        player['pan'].value = pan_pos
        player['sf'].setSpeed(1.059463 if is_center else 1.0)
        v = volume_override if volume_override is not None else player['vol']
        player['amp'].value = v
        player['sf'].play()

    def _pulse_interval(self, distance_m: float) -> float:
        interval = 2.5 * (distance_m / 100.0) ** 0.8
        return round(max(0.25, min(3.0, interval)), 2)

    def start_quest_pulse(self, icon: NavIcon):
        with self._lock:
            self._current_icon = icon
            if not self._pulse_active:
                self._pulse_active = True
                self._pulse_thread = threading.Thread(target=self._pulse_loop, daemon=True)
                self._pulse_thread.start()

    def update_quest_icon(self, icon: NavIcon):
        with self._lock:
            self._current_icon = icon

    def stop_quest_pulse(self):
        with self._lock:
            self._pulse_active = False

    def _pulse_loop(self):
        my_thread = threading.current_thread()
        while True:
            with self._lock:
                active = self._pulse_active
                icon = self._current_icon
                is_current = my_thread is self._pulse_thread
            if not active or icon is None or (not is_current):
                break
            is_center = icon.direction == 'center'
            self._play_sound(self.quest, icon, is_center=is_center)
            time.sleep(self._pulse_interval(icon.distance_m))

    def play_treasure_earcon(self, icon: NavIcon):
        self._play_sound(self.treasure, icon)

    def play_stockpile_earcon(self, icon: NavIcon):
        self._play_sound(self.stockpile, icon)

    def play_scan(self, icons: list[NavIcon]):
        print('\n  SCAN MODE — sweeping compass icons...\n')

        def sort_key(i):
            return -i.offset if i.direction == 'left' else i.offset
        sorted_icons = sorted(icons, key=sort_key)
        for icon in sorted_icons:
            if icon.icon_type == 'main_quest':
                self._play_sound(self.quest, icon, is_center=icon.direction == 'center')
            elif icon.icon_type == 'treasure':
                self._play_sound(self.treasure, icon)
            elif icon.icon_type == 'stockpile':
                self._play_sound(self.stockpile, icon)
            arrow = '<-' if icon.direction == 'left' else '->' if icon.direction == 'right' else '.'
            print(f'    {icon.icon_type:<14}  {arrow} {icon.direction:<8}  {icon.distance_m:.0f}m')
            time.sleep(0.65)
        time.sleep(0.2)
        for icon in sorted_icons:
            phrase = _build_tts_phrase(icon)
            print(f'      "{phrase}"')
            self.tts.speak(phrase)

class DistanceCache:
    DEFAULT_DIST: float = 50.0
    MIN_DIST: float = 1.0
    MAX_DIST: float = 500.0

    def __init__(self):
        self._cache: dict[str, float] = {}

    def resolve(self, icon_type: str, raw: Optional[float]) -> float:
        if raw is not None and self.MIN_DIST <= raw <= self.MAX_DIST:
            self._cache[icon_type] = raw
            return raw
        if icon_type in self._cache:
            return self._cache[icon_type]
        return self.DEFAULT_DIST

    def resolve_icon(self, icon: NavIcon) -> NavIcon:
        resolved = self.resolve(icon.icon_type, icon.distance_m)
        if resolved == icon.distance_m:
            return icon
        from dataclasses import replace
        return replace(icon, distance_m=resolved)

    def invalidate(self, icon_type: str) -> None:
        self._cache.pop(icon_type, None)

class NavigationController:
    TREASURE_THRESHOLDS = [30.0, 20.0, 10.0, 5.0]
    STOCKPILE_THRESHOLDS = [15.0, 10.0, 5.0]

    def __init__(self):
        self.audio = AudioEngine()
        self._dist_cache = DistanceCache()
        self._last_treasure_thresh: Optional[float] = None
        self._last_stockpile_thresh: Optional[float] = None
        self._last_quest_seen: Optional[float] = None
        self._quest_arrived: bool = False

    def update(self, icons: list[NavIcon]):
        resolved = [self._dist_cache.resolve_icon(i) for i in icons]
        quest = next((i for i in resolved if i.icon_type == 'main_quest'), None)
        treasure = next((i for i in resolved if i.icon_type == 'treasure'), None)
        stockpile = next((i for i in resolved if i.icon_type == 'stockpile'), None)
        now = time.time()
        if quest:
            self._last_quest_seen = now
            if quest.distance_m is not None and quest.distance_m <= 5.0:
                if not self._quest_arrived:
                    self._quest_arrived = True
                    self.audio.stop_quest_pulse()
                    self.audio.tts.speak("Vous êtes arrivé à l'objectif.")
            else:
                if self._quest_arrived and quest.distance_m is not None and (quest.distance_m > 8.0):
                    self._quest_arrived = False
                if not self._quest_arrived:
                    self.audio.update_quest_icon(quest)
                    if not self.audio._pulse_active:
                        self.audio.start_quest_pulse(quest)
        elif self._last_quest_seen is not None and now - self._last_quest_seen > 6.0:
            self.audio.stop_quest_pulse()
            self._last_quest_seen = None
        if treasure:
            crossed = self._crossed_threshold(treasure.distance_m, self.TREASURE_THRESHOLDS, self._last_treasure_thresh)
            if crossed is not None:
                self.audio.play_treasure_earcon(treasure)
                self._last_treasure_thresh = crossed
        else:
            self._last_treasure_thresh = None
        if stockpile:
            crossed = self._crossed_threshold(stockpile.distance_m, self.STOCKPILE_THRESHOLDS, self._last_stockpile_thresh)
            if crossed is not None:
                self.audio.play_stockpile_earcon(stockpile)
                self._last_stockpile_thresh = crossed
        else:
            self._last_stockpile_thresh = None

    def scan(self, icons: list[NavIcon]):
        resolved = [self._dist_cache.resolve_icon(i) for i in icons]
        self.audio.play_scan(resolved)
        self.audio.tts.wait()

    def stop(self) -> None:
        self.audio.stop_quest_pulse()

    @staticmethod
    def _crossed_threshold(distance_m: float, thresholds: list[float], last_thresh: Optional[float]) -> Optional[float]:
        active = [t for t in thresholds if distance_m <= t]
        if not active:
            return None
        tightest = min(active)
        if last_thresh is not None and tightest >= last_thresh:
            return None
        return tightest

def run_demo():
    print('=' * 55)
    print('  AC Shadows — Audio Navigation Tool')
    print('  Accessibility feature demo')
    print('=' * 55)
    scenarios = [('approaching', 'Quest marker ahead, nearly centered — close (27m)'), ('veer_right', 'Player drifted — quest is far right (95m)'), ('veer_left', 'Player drifted — quest is far left (80m)'), ('treasure_nearby', 'Treasure nearby on left (18m) + quest right (45m)'), ('multi', 'Both icons visible — quest left, treasure right'), ('stockpile_close', 'Stockpile at 12m right — ambient earcon should fire'), ('stockpile_far', 'Stockpile at 80m — too far for ambient, quiet'), ('all_icons', 'All three icons — quest center, treasure left, stockpile right'), ('label_missing', 'Distance label invisible (None) — cache fallback active')]
    for scenario_key, description in scenarios:
        icons = generate_test_scenario(scenario_key)
        print(f'  >  {description}')
        for icon in icons:
            resolved = controller._dist_cache.resolve_icon(icon)
            bar = _direction_bar(icon.direction, icon.offset)
            raw_str = f'{icon.distance_m:.0f}m' if icon.distance_m is not None else 'N/A'
            print(f'     {icon.icon_type:<14}  {bar}  raw={raw_str}  resolved={resolved.distance_m:.0f}m  (pulse every ~{controller.audio._pulse_interval(resolved.distance_m):.1f}s)')
        start = time.time()
        while time.time() - start < 6.0:
            controller.update(icons)
            time.sleep(0.2)
        controller.stop()
        time.sleep(0.5)
        print()
    print('Running scan mode demo (all three icon types + TTS)...')
    all_icons = generate_test_scenario('all_icons')
    controller.scan(all_icons)
    print()
    print('  Demo done!')
    controller.audio.shutdown()

def _direction_bar(direction: str, offset: float) -> str:
    width = 20
    center = width // 2
    if direction == 'right':
        pos = center + int(offset * center)
    elif direction == 'left':
        pos = center - int(offset * center)
    else:
        pos = center
    pos = max(0, min(width - 1, pos))
    bar = ['-'] * width
    bar[center] = '|'
    bar[pos] = 'O'
    return '[' + ''.join(bar) + ']'
if __name__ == '__main__':
    run_demo()