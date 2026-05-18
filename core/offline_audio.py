import numpy as np
import os
import scipy.io.wavfile as wavfile
from typing import List
from core.audiofeedback import NavIcon

class OfflineAudioRenderer:
    SAMPLE_RATE = 44100

    def __init__(self, total_duration_sec: float):
        self.total_samples = int(total_duration_sec * self.SAMPLE_RATE)
        self.audio_buffer = np.zeros((self.total_samples, 2), dtype=np.float32)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        audio_dir = os.path.join(base_dir, 'assets', 'audio')
        self.sounds = {'quest': self._load_sound(os.path.join(audio_dir, 'koto_note.wav'), 0.4), 'quest_center': self._load_sound(self._get_or_create_center_sound(audio_dir), 0.4), 'treasure': self._load_sound(os.path.join(audio_dir, 'koto_trill.wav'), 0.4), 'stockpile': self._load_sound(os.path.join(audio_dir, 'koto_stockpile.wav'), 0.2)}

    def _get_or_create_center_sound(self, audio_dir: str) -> str:
        orig_path = os.path.join(audio_dir, 'koto_note.wav')
        center_path = os.path.join(audio_dir, 'koto_note_center.wav')
        if not os.path.exists(center_path) and os.path.exists(orig_path):
            sr, data = wavfile.read(orig_path)
            factor = 2 ** (1.0 / 12.0)
            orig_indices = np.arange(len(data))
            new_indices = np.linspace(0, len(data) - 1, int(len(data) / factor))
            if len(data.shape) == 2:
                new_data = np.zeros((len(new_indices), 2), dtype=data.dtype)
                new_data[:, 0] = np.interp(new_indices, orig_indices, data[:, 0])
                new_data[:, 1] = np.interp(new_indices, orig_indices, data[:, 1])
            else:
                new_data = np.interp(new_indices, orig_indices, data).astype(data.dtype)
            wavfile.write(center_path, sr, new_data)
        return center_path if os.path.exists(center_path) else orig_path

    def _load_sound(self, path: str, volume: float) -> np.ndarray:
        if not os.path.exists(path):
            print(f'  [OfflineAudio] Warning: Could not find {path}')
            return np.zeros((1, 2), dtype=np.float32)
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if len(data.shape) == 1:
            data = np.column_stack((data, data))
        return data * volume

    def add_sound_event(self, sound_type: str, icon: NavIcon, time_sec: float):
        if sound_type not in self.sounds:
            return
        sound_data = self.sounds[sound_type]
        if icon.direction == 'center':
            pan = 0.0
        elif icon.direction == 'left':
            pan = -min(1.0, icon.offset)
        else:
            pan = min(1.0, icon.offset)
        theta = (pan + 1.0) / 2.0 * (np.pi / 2.0)
        left_gain = float(np.cos(theta))
        right_gain = float(np.sin(theta))
        panned_sound = sound_data.copy()
        panned_sound[:, 0] *= left_gain
        panned_sound[:, 1] *= right_gain
        start_idx = int(time_sec * self.SAMPLE_RATE)
        end_idx = start_idx + panned_sound.shape[0]
        if start_idx >= self.total_samples:
            return
        if end_idx > self.total_samples:
            valid_len = self.total_samples - start_idx
            panned_sound = panned_sound[:valid_len]
            end_idx = self.total_samples
        self.audio_buffer[start_idx:end_idx] += panned_sound

    def export_wav(self, output_path: str):
        final_audio = np.clip(self.audio_buffer, -1.0, 1.0)
        final_audio_int16 = (final_audio * 32767).astype(np.int16)
        wavfile.write(output_path, self.SAMPLE_RATE, final_audio_int16)

class OfflineNavigationController:
    TREASURE_THRESHOLDS = [30.0, 20.0, 10.0, 5.0]
    STOCKPILE_THRESHOLDS = [15.0, 10.0, 5.0]

    def __init__(self, total_duration_sec: float):
        self.renderer = OfflineAudioRenderer(total_duration_sec)
        self._last_treasure_dist = 999.0
        self._last_stockpile_dist = 999.0
        self._last_quest_pulse = -999.0

    def _pulse_interval(self, distance_m: float) -> float:
        distance_m = max(1.0, distance_m)
        interval = 2.5 * (distance_m / 100.0) ** 0.8
        return round(max(0.25, min(3.0, interval)), 2)

    def update(self, icons: List[NavIcon], current_time: float):
        quest = next((i for i in icons if i.icon_type == 'main_quest'), None)
        treasure = next((i for i in icons if i.icon_type == 'treasure'), None)
        stockpile = next((i for i in icons if i.icon_type == 'stockpile'), None)
        if quest:
            interval = self._pulse_interval(quest.distance_m)
            if current_time - self._last_quest_pulse >= interval:
                sound_type = 'quest_center' if quest.direction == 'center' else 'quest'
                self.renderer.add_sound_event(sound_type, quest, current_time)
                self._last_quest_pulse = current_time
        if treasure:
            active_thresh = [t for t in self.TREASURE_THRESHOLDS if treasure.distance_m <= t]
            if active_thresh:
                closest = min(active_thresh)
                if closest < self._last_treasure_dist:
                    self.renderer.add_sound_event('treasure', treasure, current_time)
                    self._last_treasure_dist = closest
            else:
                self._last_treasure_dist = 999.0
        if stockpile:
            active_thresh = [t for t in self.STOCKPILE_THRESHOLDS if stockpile.distance_m <= t]
            if active_thresh:
                closest = min(active_thresh)
                if closest < self._last_stockpile_dist:
                    self.renderer.add_sound_event('stockpile', stockpile, current_time)
                    self._last_stockpile_dist = closest
            else:
                self._last_stockpile_dist = 999.0

    def export(self, output_path: str):
        self.renderer.export_wav(output_path)