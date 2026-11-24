import os
import threading
import time
import sounddevice as sd
import soundfile as sf

class AudioAlert:

    def __init__(self, audio_file: str, cooldown: float = 2.0):
        self.audio_path = os.path.abspath(audio_file)
        self.cooldown = cooldown
        self.last_played_time = 0.0
        self._lock = threading.Lock()
        self.data = None
        self.samplerate = 44100
        self.duration = 0.0

        self.data, self.samplerate = sf.read(self.audio_path, dtype="float32")
        self.duration = len(self.data) / float(self.samplerate)
        print("Audio loaded")

    def is_playing(self) -> bool:
        try:
            stream = sd.get_stream()
            if stream and stream.active:
                return True
        except Exception:
            pass
        return False

    def play(self):
        if self.data is None:
            return

        with self._lock:
            if self.is_playing():
                return

            now = time.time()
            min_interval = max(self.duration, self.cooldown)
            if now - self.last_played_time < min_interval:
                return

            self.last_played_time = now
            try:
                sd.play(self.data, self.samplerate)
            except Exception as e:
                print(f"Warning: Error during audio playback: {e}")
