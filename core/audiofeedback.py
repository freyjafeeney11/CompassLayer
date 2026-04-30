"""
AC Shadows - Audio Navigation Accessibility Tool
=================================================
Provides spatial audio feedback for blind/visually impaired players.

SETUP:
    pip install pyo numpy pyttsx3

AUDIO FILES:
    Place GarageBand exports in the same folder as this script:
        - koto_note.wav       -> main quest pulse tone
        - koto_trill.wav      -> treasure earcon
        - koto_stockpile.wav  -> stockpile earcon

HOW IT WORKS:
    - Main quest icon:  continuous panning pulse, rate increases as you get closer
    - Treasure icon:    distinct trill earcon, panned to its direction (on proximity)
    - Stockpile icon:   soft percussion earcon — scan-mode preferred, ambient only when
                        very close (15 m) with a long cooldown (20 s) to avoid clutter
    - Scan mode:        press SPACE to sweep all current icons left-to-right,
                        then TTS describes each icon's location naturally
    - All sounds pan left/right based on icon position on the compass

PANNING NOTES:
    Uses equal-power panning (cos/sin curve) so volume stays perceptually
    consistent across the stereo field — no harsh jump between ears.
"""

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
    print("pyttsx3 not found — TTS disabled. Install with: pip install pyttsx3")
    TTS_AVAILABLE = False


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class NavIcon:
    """Represents a single icon on the AC Shadows compass bar."""
    icon_type:  str    # "main_quest" | "treasure" | "stockpile"
    direction:  str    # "left" | "right" | "center"
    offset:     float  # 0.0 (center) -> 1.0 (far edge)
    distance_m: float  # distance in metres, e.g. 27.0


# ─────────────────────────────────────────────
#  ALGO DATA ADAPTER
# ─────────────────────────────────────────────

def from_algo_data(raw: dict) -> Optional[NavIcon]:
    """
    Converts a row from detection algorithm output into a NavIcon.

    Expected input:
        {'id': 0, 'label': 'main_quest', 'rel_offset': -0.047,
         'direction': 'Left', 'distance': '28m'}

    Valid labels: 'main_quest', 'treasure', 'stockpile'
    """
    dist_str = raw.get('distance', 'N/A')
    distance = 0.0
    if dist_str != 'N/A':
        try:
            distance = float(dist_str.replace('m', '').strip())
        except ValueError:
            pass

    direction = raw.get('direction', 'Straight').lower()
    offset    = abs(raw.get('rel_offset', 0.0))

    if direction == 'straight':
        direction = 'center'

    label = raw.get('label', 'main_quest')
    if label not in ('main_quest', 'treasure', 'stockpile'):
        label = 'main_quest'

    return NavIcon(
        icon_type  = label,
        direction  = direction,
        offset     = offset,
        distance_m = distance,
    )


def from_algo_batch(rows: list[dict]) -> list[NavIcon]:
    return [icon for icon in (from_algo_data(r) for r in rows) if icon is not None]


# ─────────────────────────────────────────────
#  TEST DATA GENERATOR
# ─────────────────────────────────────────────

def generate_test_scenario(scenario: str = "approaching") -> list[NavIcon]:
    scenarios = {
        "approaching":     [NavIcon("main_quest", "center", 0.05, 27.0)],
        "veer_left":       [NavIcon("main_quest", "left",   0.6,  80.0)],
        "veer_right":      [NavIcon("main_quest", "right",  0.7,  95.0)],
        "treasure_nearby": [NavIcon("main_quest", "right",  0.1,  45.0),
                            NavIcon("treasure",   "left",   0.3,  18.0)],
        "multi":           [NavIcon("main_quest", "left",   0.4,  60.0),
                            NavIcon("treasure",   "right",  0.5,  22.0)],
        "stockpile_close": [NavIcon("main_quest", "center", 0.05, 40.0),
                            NavIcon("stockpile",  "right",  0.35, 12.0)],
        "stockpile_far":   [NavIcon("main_quest", "left",   0.3,  55.0),
                            NavIcon("stockpile",  "right",  0.6,  80.0)],
        "all_icons":       [NavIcon("main_quest", "center", 0.05, 30.0),
                            NavIcon("treasure",   "left",   0.4,  20.0),
                            NavIcon("stockpile",  "right",  0.5,  14.0)],
    }
    return scenarios.get(scenario, scenarios["approaching"])


# ─────────────────────────────────────────────
#  TTS ENGINE
# ─────────────────────────────────────────────

class TTSEngine:
    """

    Why win32com instead of pyttsx3?
    ---------------------------------
    pyttsx3's SAPI5 driver shares one COM SpVoice object across calls,
    which means a second say() issued before the first finishes will
    silently drop the queued phrase.  Calling win32com.client.Dispatch
    directly gives us a fresh, isolated SpVoice per phrase — guaranteed
    sequential delivery with no drops.

    pyttsx3 is kept as a fallback if pywin32 is not installed.

    Install:
        pip install pywin32      <- preferred
        pip install pyttsx3      <- fallback
    """

    def __init__(self):
        self._queue:  list[str] = []
        self._lock    = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active  = False
        self._backend = None   # "win32com" | "pyttsx3"
        self._fr_voice_token = None   # French SAPI5 voice token (Hortense, Paul, etc.)

        try:
            import win32com.client
            self._backend = "win32com"
            self._active  = True

            try:
                voices    = win32com.client.Dispatch("SAPI.SpVoice")
                token_enum = voices.GetVoices()
                FR_HINTS  = ("hortense", "paul", " fr", "french", "français")
                for i in range(token_enum.Count):
                    token = token_enum.Item(i)
                    desc  = token.GetDescription().lower()
                    if any(hint in desc for hint in FR_HINTS):
                        self._fr_voice_token = token
                        print(f"  [+] TTS voice: {token.GetDescription()}")
                        break
                if self._fr_voice_token is None:
                    print("  [i]  No French SAPI5 voice found — using system default. "
                          "Install 'Microsoft Hortense Desktop' from Windows language settings.")
            except Exception:
                pass

            print("  [+] TTS engine ready (win32com / SAPI5)")
            return
        except ImportError:
            pass

        if TTS_AVAILABLE:
            try:
                import pyttsx3
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate',   200)
                engine.setProperty('volume', 0.95)
                voices   = engine.getProperty('voices')
                FR_HINTS = ("hortense", "paul", "fr_", "french", "français")
                for v in voices:
                    if any(hint in v.name.lower() or hint in v.id.lower()
                           for hint in FR_HINTS):
                        engine.setProperty('voice', v.id)
                        print(f"  [+] TTS voice: {v.name}")
                        break
                else:
                    print("  [i]  No French voice found in pyttsx3 — using default.")
                self._pyttsx3_engine = engine
                self._backend = "pyttsx3"
                self._active  = True
                print("  [+] TTS engine ready (pyttsx3 fallback)")
            except Exception as e:
                print(f"  [!]  TTS init failed: {e}")
        else:
            print("  No TTS backend available (install pywin32 or pyttsx3)")

    # ── Public API ──────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Queue a phrase. Non-blocking — drains sequentially on a background thread."""
        if not self._active:
            return
        with self._lock:
            self._queue.append(text)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._drain_queue, daemon=True)
            self._thread.start()

    def speak_icon(self, icon: "NavIcon") -> None:
        self.speak(_build_tts_phrase(icon))

    def wait(self) -> None:
        """Block until all queued phrases have been spoken."""
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def stop(self) -> None:
        """Clear the pending queue without waiting for in-progress speech."""
        with self._lock:
            self._queue.clear()

    # ── Internal ────────────────────────────────────────────────────

    def _drain_queue(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    break
                phrase = self._queue.pop(0)
            try:
                if self._backend == "win32com":
                    self._speak_win32(phrase)
                else:
                    self._speak_pyttsx3(phrase)
            except Exception as e:
                print(f"  TTS error: {e}")

    def _speak_win32(self, phrase: str) -> None:
        import win32com.client
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        if self._fr_voice_token is not None:
            voice.Voice = self._fr_voice_token
        voice.Rate   = 0
        voice.Volume = 60
        voice.Speak(phrase, 0)   # 0 = synchronous

    def _speak_pyttsx3(self, phrase: str) -> None:
        self._pyttsx3_engine.say(phrase)
        self._pyttsx3_engine.runAndWait()


def _build_tts_phrase(icon: NavIcon) -> str:
    """
    Construit une phrase de localisation naturelle en français.

    Exemples
    --------
    "Objectif principal, légèrement à gauche, à 27 mètres"
    "Trésor, loin à droite, à 18 mètres"
    "Réserve, à droite, à 14 mètres, contient du bois, des cultures ou des minéraux"
    "Objectif principal, droit devant, à 5 mètres"
    """
    if icon.icon_type == "main_quest":
        label = "Objectif principal"
    elif icon.icon_type == "treasure":
        label = "Trésor"
    elif icon.icon_type == "stockpile":
        label = "Réserve"
    else:
        label = icon.icon_type.replace("_", " ")

    if icon.direction == "center" or icon.offset < 0.05:
        position = "droit devant"
    else:
        side = "gauche" if icon.direction == "left" else "droite"
        if icon.offset > 0.55:
            position = f"loin à {side}"
        elif icon.offset > 0.25:
            position = f"à {side}"
        else:
            position = f"légèrement à {side}"

    dist     = int(icon.distance_m)
    distance = f"à {dist} mètre{'s' if dist != 1 else ''}"
    suffix   = ", contient du bois, des cultures ou des minéraux" \
               if icon.icon_type == "stockpile" else ""

    return f"{label}, {position}, {distance}{suffix}"


# ─────────────────────────────────────────────
#  AUDIO ENGINE
# ─────────────────────────────────────────────

class AudioEngine:
    """Handles all spatial audio output with pyo (Binaural HRTF)."""

    SAMPLE_RATE = 44100

    # Volume Levels
    VOLUME_QUEST     = 0.4
    VOLUME_TREASURE  = 2.5  # Heavily boosted to compensate for quiet source file
    VOLUME_STOCKPILE = 0.15 # Lowered to keep ambient footprint minimal

    def __init__(self):
        # Boot pyo server for real-time DSP and HRTF
        self.server = pyo.Server(audio="portaudio", sr=self.SAMPLE_RATE, duplex=0).boot()
        self.server.start()

        # Set up paths relative to this file's location (core/audiofeedback.py)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        audio_dir = os.path.join(base_dir, "assets", "audio")

        self.quest     = self._load(os.path.join(audio_dir, "koto_note.wav"),      self.VOLUME_QUEST)
        self.treasure  = self._load(os.path.join(audio_dir, "koto_trill.wav"),     self.VOLUME_TREASURE)
        self.stockpile = self._load(os.path.join(audio_dir, "koto_stockpile.wav"), self.VOLUME_STOCKPILE)

        self.tts = TTSEngine()

        self._pulse_active = False
        self._pulse_thread: Optional[threading.Thread] = None
        self._current_icon: Optional[NavIcon] = None
        self._lock = threading.Lock()

    def shutdown(self):
        self.server.stop()
        self.server.shutdown()

    # ── Sound loading ───────────────────────────────────────────────

    def _load(self, filename: str, volume: float) -> Optional[dict]:
        if not os.path.exists(filename):
            print(f"  [!] Missing audio file: {filename}")
            return None
            
        sf = pyo.SfPlayer(filename, loop=False)
        sf.stop()
        sf_mono = sf.mix(1) # Mix to mono for HRTF
        
        # Apply initial volume scaling
        amp = pyo.Sig(volume)
        binaural = pyo.Binaural(sf_mono * amp, azimuth=0, elevation=0).out()
        
        print(f"  [+] Loaded {filename}")
        return {"sf": sf, "binaural": binaural, "amp": amp, "vol": volume}

    # ── HRTF Panning & Pitch ────────────────────────────────────────

    def _play_sound(self, player: Optional[dict], icon: NavIcon, volume_override: Optional[float] = None, is_center: bool = False):
        if not player: return
        
        # Azimuth mapping for Binaural: center=0, right=+90, left=-90
        if icon.direction == "center":
            azimuth = 0.0
        elif icon.direction == "right":
            azimuth = min(1.0, icon.offset) * 90.0
        elif icon.direction == "left":
            azimuth = -min(1.0, icon.offset) * 90.0
            
        player["binaural"].setAzimuth(azimuth)
        
        # Pitch up 1 semitone by increasing playback speed (2**(1/12) ≈ 1.059463)
        player["sf"].setSpeed(1.059463 if is_center else 1.0)
        
        v = volume_override if volume_override is not None else player["vol"]
        player["amp"].value = v
        
        player["sf"].play()

    # ── Pulse rate ──────────────────────────────────────────────────

    def _pulse_interval(self, distance_m: float) -> float:
        """
        Map distance -> pulse interval (seconds).
        Exponential curve: floors at 0.25s (not too annoying) and maxes around 2.5s
        """
        distance_m = max(1.0, distance_m)
        interval = 2.5 * ((distance_m / 100.0) ** 0.8)
        return round(max(0.25, min(3.0, interval)), 2)

    # ── Continuous quest pulse ──────────────────────────────────────

    def start_quest_pulse(self, icon: NavIcon):
        with self._lock:
            self._current_icon = icon
            if not self._pulse_active:
                self._pulse_active = True
                self._pulse_thread = threading.Thread(
                    target=self._pulse_loop, daemon=True
                )
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
                icon   = self._current_icon
                is_current = (my_thread is self._pulse_thread)
            if not active or icon is None or not is_current:
                break
            
            is_center = (icon.direction == "center")
            self._play_sound(self.quest, icon, is_center=is_center)
            time.sleep(self._pulse_interval(icon.distance_m))

    # ── Treasure earcon ─────────────────────────────────────────────

    def play_treasure_earcon(self, icon: NavIcon):
        self._play_sound(self.treasure, icon)

    # ── Stockpile earcon ────────────────────────────────────────────

    def play_stockpile_earcon(self, icon: NavIcon):
        self._play_sound(self.stockpile, icon)

    # ── Scan mode ───────────────────────────────────────────────────

    def play_scan(self, icons: list[NavIcon]):
        """
        Sweep all visible icons left-to-right with audio, then speak
        each icon's location via TTS.
        """
        print("\n  SCAN MODE — sweeping compass icons...\n")

        def sort_key(i):
            return -i.offset if i.direction == "left" else i.offset

        sorted_icons = sorted(icons, key=sort_key)

        for icon in sorted_icons:
            if icon.icon_type == "main_quest":
                self._play_sound(self.quest, icon, is_center=(icon.direction == "center"))
            elif icon.icon_type == "treasure":
                self._play_sound(self.treasure, icon)
            elif icon.icon_type == "stockpile":
                self._play_sound(self.stockpile, icon)

            arrow = "<-" if icon.direction == "left" else ("->" if icon.direction == "right" else ".")
            print(f"    {icon.icon_type:<14}  {arrow} {icon.direction:<8}  {icon.distance_m:.0f}m")
            time.sleep(0.65)

        time.sleep(0.2)
        for icon in sorted_icons:
            phrase = _build_tts_phrase(icon)
            print(f"      \"{phrase}\"")
            self.tts.speak(phrase)


# ─────────────────────────────────────────────
#  NAVIGATION CONTROLLER
# ─────────────────────────────────────────────

class NavigationController:
    """
    Orchestrates all icons: runs the quest pulse continuously,
    fires treasure/stockpile earcons on distance thresholds, handles scan mode.
    """

    TREASURE_THRESHOLDS  = [30.0, 20.0, 10.0, 5.0]
    STOCKPILE_THRESHOLDS = [15.0, 10.0, 5.0]

    def __init__(self):
        self.audio = AudioEngine()
        self._last_treasure_dist = 999.0
        self._last_stockpile_dist = 999.0

    def update(self, icons: list[NavIcon]):
        quest     = next((i for i in icons if i.icon_type == "main_quest"), None)
        treasure  = next((i for i in icons if i.icon_type == "treasure"),   None)
        stockpile = next((i for i in icons if i.icon_type == "stockpile"),  None)

        if quest:
            self.audio.update_quest_icon(quest)
            if not self.audio._pulse_active:
                self.audio.start_quest_pulse(quest)

        if treasure:
            active_thresh = [t for t in self.TREASURE_THRESHOLDS if treasure.distance_m <= t]
            if active_thresh:
                closest = min(active_thresh)
                if closest < self._last_treasure_dist:
                    self.audio.play_treasure_earcon(treasure)
                    self._last_treasure_dist = closest
            else:
                self._last_treasure_dist = 999.0

        if stockpile:
            active_thresh = [t for t in self.STOCKPILE_THRESHOLDS if stockpile.distance_m <= t]
            if active_thresh:
                closest = min(active_thresh)
                if closest < self._last_stockpile_dist:
                    self.audio.play_stockpile_earcon(stockpile)
                    self._last_stockpile_dist = closest
            else:
                self._last_stockpile_dist = 999.0

    def scan(self, icons: list[NavIcon]):
        self.audio.play_scan(icons)
        self.audio.tts.wait()

    def stop(self) -> None:
        self.audio.stop_quest_pulse()


# ─────────────────────────────────────────────
#  DEMO / TEST RUNNER
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 55)
    print("  AC Shadows — Audio Navigation Tool")
    print("  Accessibility feature demo")
    print("=" * 55)
    print()
    print("  Loading audio...")

    controller = NavigationController()

    print()
    print("  Each scenario runs for ~6 seconds automatically.")
    print()

    scenarios = [
        ("approaching",     "Quest marker ahead, nearly centered — close (27m)"),
        ("veer_right",      "Player drifted — quest is far right (95m)"),
        ("veer_left",       "Player drifted — quest is far left (80m)"),
        ("treasure_nearby", "Treasure nearby on left (18m) + quest right (45m)"),
        ("multi",           "Both icons visible — quest left, treasure right"),
        ("stockpile_close", "Stockpile at 12m right — ambient earcon should fire"),
        ("stockpile_far",   "Stockpile at 80m — too far for ambient, quiet"),
        ("all_icons",       "All three icons — quest center, treasure left, stockpile right"),
    ]

    for scenario_key, description in scenarios:
        icons = generate_test_scenario(scenario_key)

        print(f"  >  {description}")
        for icon in icons:
            arrow = "->" if icon.direction == "right" else ("<-" if icon.direction == "left" else ".")
            bar   = _direction_bar(icon.direction, icon.offset)
            print(f"     {icon.icon_type:<14}  {bar}  {icon.distance_m:.0f}m  "
                  f"(pulse every ~{controller.audio._pulse_interval(icon.distance_m):.1f}s)")

        start = time.time()
        while time.time() - start < 6.0:
            controller.update(icons)
            time.sleep(0.2)

        controller.stop()
        time.sleep(0.5)
        print()

    print("Running scan mode demo (all three icon types + TTS)...")
    all_icons = generate_test_scenario("all_icons")
    controller.scan(all_icons)

    print()
    print("  Demo done!")
    controller.audio.shutdown()


def _direction_bar(direction: str, offset: float) -> str:
    width  = 20
    center = width // 2
    if direction == "right":
        pos = center + int(offset * center)
    elif direction == "left":
        pos = center - int(offset * center)
    else:
        pos = center
    pos    = max(0, min(width - 1, pos))
    bar    = ["-"] * width
    bar[center] = "|"
    bar[pos]    = "O"
    return "[" + "".join(bar) + "]"


if __name__ == "__main__":
    run_demo()