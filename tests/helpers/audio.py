import io
import math
import struct
import wave


def wav(
        seconds: float, sample_rate: int, amplitude: int = 6000, frequency: float = 300, gate: int | None = None
) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                    if gate is None or (i // gate) % 2 == 0 else 0)
        for i in range(int(seconds * sample_rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return buffer.getvalue()
