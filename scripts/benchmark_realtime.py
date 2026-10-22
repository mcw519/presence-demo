"""Real-time localhost replay using synthetic blank video and PCM, with cloud services off.

Compare the same duration before/after restarting the server. FPS follows state
recommendations exactly as the repaired browser does; missing recommendations
fall back to 1 fps, reproducing the old protocol. No camera or microphone is used.
"""

import argparse
import asyncio
import base64
import json
import struct
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.client import connect


async def main(args):
    seconds = args.seconds
    image = np.full((480, 640, 3), 80, np.uint8)
    if args.scene in ("face", "twofaces"):
        portrait = Path(__file__).resolve().parents[1] / "tests" / "data" / "face.jpg"
        image[60:420, 200:440] = cv2.resize(cv2.imread(str(portrait)), (240, 360))
        if args.scene == "twofaces":
            image[:] = 80
            for left in (60, 360):
                image[60:420, left : left + 220] = cv2.resize(cv2.imread(str(portrait)), (220, 360))
    jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    frame = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    pcm = (np.random.default_rng(42).normal(0, 0.01, 320) * 32767).astype("<i2").tobytes()
    pcm_blocks = [pcm]
    if args.input_wav:
        with wave.open(str(args.input_wav)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError("--input-wav requires mono PCM16 at 16000 Hz")
            data = wav.readframes(wav.getnframes())
        pcm_blocks = [data[i : i + 640] for i in range(0, len(data) - 639, 640)]
        if not pcm_blocks:
            raise ValueError("--input-wav needs at least 20 ms")
    rate = args.fps or 1.0
    rates = set()
    sent_frames = 0
    sent_video_bytes = 0
    audio = []
    samples = []
    errors = []
    async with connect(args.url, max_size=2**22) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "config",
                    "config": {
                        "audio_pipeline": "puresound",
                        "pose_enabled": True,
                        "adaptive_fps": not bool(args.fps),
                        "motion_gate": not args.full_vision,
                        "fps_low": 0.5,
                        "fps_high": 2,
                        "reid_enabled": False,
                        "asr_provider": "none",
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "start"}))
        start = time.perf_counter()

        async def receive():
            nonlocal rate
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                if not args.fps and m["type"] == "state" and m.get("recommended_fps", 0) > 0:
                    rate = m["recommended_fps"]
                if m["type"] == "audio" and m.get("available"):
                    audio.append(m)
                if m["type"] == "stats":
                    samples.append(m)
                if m["type"] == "error":
                    errors.append(m)

        rx = asyncio.create_task(receive())

        async def camera():
            nonlocal sent_frames, sent_video_bytes
            while time.perf_counter() - start < seconds:
                rates.add(rate)
                timestamp = time.perf_counter() * 1000
                if args.transport == "binary":
                    packet = struct.pack("<BdB", 4, timestamp, 0) + jpeg
                else:
                    packet = json.dumps({"type": "frame", "speech": False, "t": timestamp, "data": frame})
                await ws.send(packet)
                sent_video_bytes += len(packet) if isinstance(packet, bytes) else len(packet.encode())
                sent_frames += 1
                await asyncio.sleep(1 / rate)

        async def mic():
            for i in range(seconds * 50):
                await asyncio.sleep(max(0, start + i * 0.02 - time.perf_counter()))
                await ws.send(b"\x01" + struct.pack("<d", time.perf_counter() * 1000) + pcm_blocks[i % len(pcm_blocks)])

        await asyncio.gather(camera(), mic())
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "stop"}))
        rx.cancel()
        recent = audio[len(audio) // 2 :]
        result = {
            "seconds": seconds,
            "frames_sent": sent_frames,
            "transport": args.transport,
            "scene": args.scene,
            "audio_source": "wav_replay" if args.input_wav else "repeated_noise_block",
            "video_application_bytes_sent": sent_video_bytes,
            "optim": samples[-1]["conn"]["optim"],
            "fps_settings": sorted(rates),
            "second_half_median_cumulative_rtf": float(np.median([m["rtf"] for m in recent])),
            "second_half_median_recent_rtf": float(np.median([m["rtf_recent"] for m in recent])),
            "audio_metric_samples": len(audio),
            "audio_queue_max": max(m["queue"] for m in audio),
            "audio_dropped": audio[-1]["dropped"],
            "server_cpu_avg": float(np.mean([m["system"]["proc_cpu"] for m in samples[2:]])),
            "vision_total_avg_ms": samples[-1]["conn"]["latency_ms"]["total"]["avg"],
            "errors": errors,
            "last_audio_metrics": {
                k: audio[-1].get(k) for k in ["rtf_recent", "proc_p95_ms", "queue_wait_ms_avg", "queue_wait_ms_p95"]
            },
            "last_vision_metrics": samples[-1]["conn"]["latency_ms"],
        }
        print(json.dumps(result, indent=2))
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--seconds", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("json", "binary"), default="json")
    parser.add_argument("--scene", choices=("blank", "face", "twofaces"), default="blank")
    parser.add_argument("--input-wav", type=Path, help="optional local mono PCM16 16 kHz replay; no cloud")
    parser.add_argument("--fps", type=float, default=0, help="fixed FPS; 0 follows server recommendation")
    parser.add_argument(
        "--full-vision", action="store_true", help="disable face motion gate for a controlled comparison"
    )
    args = parser.parse_args()
    if args.seconds < 6:
        parser.error("--seconds must be at least 6")
    if not 0 <= args.fps <= 10:
        parser.error("--fps must be in 0..10")
    asyncio.run(main(args))
