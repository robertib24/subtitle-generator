from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from typing import Iterable

model_size="large-v2"      
model = WhisperModel(model_size, device="cuda", compute_type="float16")

def format_timestamp(seconds: float, always_include_hours: bool = True):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

transcription, _ = model.transcribe(
            str("onlymp3.to - Esaret 171. Bölüm Redemption Episode 171-QO2R2qKSGek-192k-1695684116.mp3"),
            language="tr",
            vad_filter="True",
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=True,
        )


for segment in transcription:
      print(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)} {segment.text}")
      audio_basename = os.path.basename(str("onlymp3.to - Esaret 171. Bölüm Redemption Episode 171-QO2R2qKSGek-192k-1695684116.mp3")).rsplit(".", 1)[0]
