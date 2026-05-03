# Faster Local Voice AI — OpenClaw Edition

A real-time voice agent that runs on a Raspberry Pi 5 (or any Linux box). Mic in, speaker out, with a face-recognition wake gate, conversational follow-ups, and the actual brain provided by an agentic LLM with tools — not a chatbot.

This is a fork of [m15-ai/Faster-Local-Voice-AI](https://github.com/m15-ai/Faster-Local-Voice-AI) with the local Vosk + Piper + Ollama pipeline swapped for **Deepgram** (streaming STT + Aura TTS) and the LLM swapped for **[OpenClaw](https://docs.openclaw.ai/)** — talking to a long-running agent via the **Agent Client Protocol (ACP)**. The ACP bridge keeps a single Node process warm across turns, which is how we get from ~14s/turn (per-call CLI) down to **~5–8s/turn** on a Pi 5.

## What it does

- You sit down in front of the camera → it says "Hello *Name*" and opens a conversation window.
- You talk normally; Deepgram transcribes in real time.
- Your OpenClaw agent runs the turn — including any tool calls (read files, run shell, etc.).
- Aura streams the reply back; you hear it through the speaker.
- Walk away or fall silent and it goes back to sleep.

Wake modes are pluggable: face, phrase ("hey girl"-style), or either.

## Architecture

```
   ┌────────────┐    PCM 16k    ┌────────────┐ ws+JSON  ┌──────────────┐
   │  client.py │ ───────────▶ │  server.py │ ───────▶ │ Deepgram STT │
   │  (JACK)    │ ◀─────────── │            │ ◀─────── │  (streaming) │
   └────────────┘   PCM 48k     │            │          └──────────────┘
        │ │                    │            │
        │ └──── camera ───────▶│ FaceWatcher│
        │                       │            │  stdin/stdout JSON-RPC
        │                       │            │  ┌───────────────────┐
        │                       │            │─▶│ openclaw acp      │
        │                       │            │◀─│  (your agent)     │
        │                       │            │  └───────────────────┘
        │                       │            │  HTTP
        │                       │            │  ┌───────────────────┐
        ▼                       │            │─▶│ Deepgram Aura TTS │
   speaker (USB DAC)            │            │◀─│  (linear16 48k)   │
                                └────────────┘  └───────────────────┘
```

- **`client.py`** — JACK audio I/O (uses PipeWire's JACK compatibility layer). Captures from the default source, plays to the default sink. Streams int16 PCM to/from the server over a single WebSocket. JACK port wiring is **dynamic** — physical capture/playback ports are discovered at startup, so a mono USB device (e.g. an all-in-one mic+speaker like the ROCWARE RC08) works without code changes.
- **`server.py`** — Orchestrates everything: Deepgram STT WebSocket, BotWorker (persistent ACP bridge), FaceWatcher (camera + face recognition), Aura TTS HTTP, wake-window state machine.
- **`BotWorker`** — Spawns `openclaw acp --session agent:<id>:<id>` once at startup. JSON-RPC over stdin/stdout: `initialize` → `session/new` → identity-boot prompt (see [Agent identity & workspace boot](#agent-identity--workspace-boot)) → many `session/prompt`. Streamed text chunks arrive as `session/update` notifications.
- **`FaceWatcher`** — Long-lived `cv2.VideoCapture`, one face-recognition tick every 2s in a thread executor. Fires `on_first_match` once per connection (greeting + open window), `on_present` while you stay in frame (refresh window). A configurable absence grace (default 300s) means brief head-turns or hand-blocks don't end the session, and the greeting is **never replayed** on reacquisition.

## Prerequisites

- **Linux** with PipeWire (tested on Raspberry Pi OS Bookworm, aarch64)
- **Python 3.11**
- **Deepgram** account (you'll need an API key — free tier works)
- **[OpenClaw](https://docs.openclaw.ai/)** installed and a working agent (this fork assumes the agent ID is `main`; configurable)
- **System packages**: `pipewire-jack`, `sox`, `pactl` (PulseAudio CLI)
- **USB mic + speaker + (optional) USB webcam** — see audio routing below

## Install

```bash
git clone https://github.com/<you>/Faster-Local-Voice-AI.git
cd Faster-Local-Voice-AI
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# face recognition (skip if you only want phrase mode):
.venv/bin/pip install opencv-python-headless face_recognition
sudo apt install pipewire-jack sox
```

> **Heads-up:** `face_recognition` pulls dlib, which compiles from source on aarch64 (Pi 5). Allow ~10 minutes the first time.

Drop your Deepgram key into `.env`:

```
DEEPGRAM_API_KEY=your_key_here
```

## Configure

Copy `config.json` → `.config.json` (the loader looks at the dotfile) and edit:

```json
{
  "volume": 65,
  "mute_mic_during_playback": true,
  "fade_duration_ms": 70,
  "aura_voice": "aura-asteria-en",
  "wake_mode": "face",
  "wake_phrase": "hey girl",
  "user_name": "You",
  "user_name_spoken": "",
  "greeting": "Hello",
  "stt_endpointing_ms": 1000,
  "stt_keyterms": [],
  "bot_bin": "/path/to/openclaw",
  "bot_agent": "main",
  "bot_cwd": "/home/you",
  "bot_thought_level": "off",
  "bot_timeout_secs": 120,
  "bot_identity_path": "",
  "workspace_docs": [],
  "face_absence_grace_secs": 300
}
```

### Audio / playback
| Key | What it does |
|---|---|
| `volume` | System master volume (0-100) set via `amixer` at client startup |
| `mute_mic_during_playback` | Hard mute mic while TTS audio is playing — cheap echo suppression in lieu of WebRTC AEC |
| `fade_duration_ms` | Crossfade tail when stopping playback (avoids clicks) |
| `aura_voice` | Deepgram Aura voice id, e.g. `aura-asteria-en`, `aura-luna-en`. See the [Deepgram voice catalog](https://developers.deepgram.com/docs/tts-models). |

### Wake gate
| Key | What it does |
|---|---|
| `wake_mode` | `"face"` (face watcher only), `"phrase"` (utterance must start with `wake_phrase`), or `"either"` |
| `wake_phrase` | Spoken trigger when `wake_mode` includes phrase |
| `face_absence_grace_secs` | How long the face must be missing before the session times out. Default 300s — brief head-turns or hand-blocks won't kick you out. |

> **Cost gate:** in `wake_mode = "face"` only, the server stops forwarding mic bytes to Deepgram once the wake window closes (audio doesn't open the window in face-only mode — only the camera does). The Deepgram STT WebSocket is kept alive via the existing keepalive task and resumes instantly when you return. In `phrase` or `either` modes, audio must keep streaming so the wake phrase can still be heard, so this gate doesn't apply.

### User & TTS personalization
| Key | What it does |
|---|---|
| `user_name` | Display name (logs, "Marwick arrived", greeting prefix) |
| `user_name_spoken` | Optional **phonetic** spelling for TTS (e.g. `"Mar-wick"`). The boot prompt tells the agent to use this when addressing you aloud, without affecting the display name. Leave empty to disable. |
| `greeting` | What the agent says (via TTS) on first arrival. Keep it TTS-friendly. |

### STT (Deepgram)
| Key | What it does |
|---|---|
| `stt_endpointing_ms` | How much trailing silence before Deepgram emits a final transcript. Lower = snappier turns; higher = better tolerance for thinking pauses. |
| `stt_keyterms` | Words to bias STT against common mishears, e.g. `["Kuri"]` to keep "Kuri" from being transcribed as "Curry". |

### Agent (OpenClaw via ACP)
| Key | What it does |
|---|---|
| `bot_bin` | Absolute path to your `openclaw` binary |
| `bot_agent` | OpenClaw agent id (e.g. `main`) |
| `bot_cwd` | Working directory for the ACP session — typically your agent's workspace |
| `bot_thought_level` | `off` / `minimal` / `low` / `medium` / `high` (passed through to OpenClaw — currently informational only over ACP) |
| `bot_timeout_secs` | Per-turn timeout |
| `bot_identity_path` | Optional path to an `IDENTITY.md` (or similar) injected into the boot prompt. See [Agent identity & workspace boot](#agent-identity--workspace-boot). |
| `workspace_docs` | Optional list of file paths (relative to `bot_cwd`) the agent must read at session start, e.g. `["AGENTS.md", "TOOLS.md"]`. Forces a proof-of-read enumeration so the agent's own conventions don't get lazily skipped. Empty list = inject only IDENTITY. |

## Capture a reference face (for face mode)

Sit in front of the camera and run:

```bash
.venv/bin/python capture_reference.py
```

It does a 3-second countdown, grabs frames until it sees exactly one face, then writes `reference_face.npy` (128-dim embedding) and `reference_face.jpg` (snapshot for sanity).

`user_name` and `greeting` are set in `.config.json` (see [Configure](#configure)).

## Agent identity & workspace boot

The OpenClaw ACP bridge spawns a fresh session that **does not** automatically inherit the agent's IDENTITY config or run the agent's normal session-startup ritual — over ACP, the agent boots as plain Claude with no persona awareness and no knowledge of the workspace conventions you've set up. To work around this, `BotWorker` injects a one-time **boot prompt** right after `session/new` that:

1. Tells the agent its operator-configured display name and role (loaded from `bot_identity_path`, e.g. an `IDENTITY.md` file).
2. Optionally instructs the agent to read a list of workspace docs (`workspace_docs`) — for example, your `AGENTS.md` (workspace map) and `TOOLS.md` (which CLI tools are wired up for this agent).
3. Requires a **proof-of-read** acknowledgment: the agent must reply `ready | loaded: <comma-separated capability names>`. Lazy `ready` answers fail this check, and the enumeration pulls the listed capabilities into conversation history so subsequent turns reliably know about them.
4. Optionally adds a phonetic-pronunciation note (`user_name_spoken`) so the agent addresses you correctly aloud.

The boot is a single round-trip on session start (~5–15s on Pi 5 depending on how many docs the agent reads). Per-turn cost after that is unaffected.

If you don't have an IDENTITY file or any workspace docs, leave `bot_identity_path` empty and `workspace_docs` empty — the boot prompt is skipped and you get the original behavior.

## Run

Two terminals:

```bash
# terminal 1 — server (bot brain, STT/TTS, face watcher)
.venv/bin/python -u server.py

# terminal 2 — client (audio I/O via JACK)
pw-jack .venv/bin/python client.py
```

The server logs latency per turn:

```
[User/window(28s left)]: what's the cpu temperature
[AI] (5.8s): CPU temperature: 52.7°C
[Perf] bot=5.8s tts=0.7s total=6.5s
```

`Ctrl-C` (SIGINT) on the server triggers a graceful shutdown — the long-lived `openclaw acp` Node subprocess is terminated cleanly (SIGTERM, 3s grace, then SIGKILL). No orphans.

## Latency

End-to-end on Pi 5 with claude-haiku-4-5 backing the agent:
- STT: streaming, effectively 0s
- Bot (model + tools): ~4–7s warm
- TTS: ~0.7–1.2s
- **Total: ~5–8s/turn**

The dominant cost is the LLM itself. Faster paths exist:
- Streaming TTS (start audio at first token) → would shave ~0.5s
- LiveKit/Pipecat front-end → barge-in, multi-device, real WebRTC AEC
- Realtime APIs (OpenAI Realtime, Gemini Live) — sub-second turns, but the model becomes the brain (you'd lose the OpenClaw tool surface)

## Known limitations

- **No barge-in.** PipeWire's WebRTC AEC was tested and isn't tight enough on a USB mic/speaker pair to prevent the agent's own voice from feeding back into STT. Mic is muted during playback. Real barge-in needs LiveKit/WebRTC.
- **Face mode requires good lighting.** HOG detector misses you if the room is dim.
- **Single-user reference.** Recognition checks against one stored embedding; multi-user would require a small DB and a different match step.
- **Aura is cloud.** If your network is bad, TTS becomes the slow leg.
- **ACP bypasses the agent's normal startup ritual.** The boot prompt (above) is the workaround. If you change agent identity or workspace docs, restart the server so the next session re-boots fresh.
- **`bot_thought_level` is informational only over ACP.** OpenClaw doesn't currently honor an ACP-set thinking level — the value still lives on the gateway side.

## Acknowledgments

- [m15-ai/Faster-Local-Voice-AI](https://github.com/m15-ai/Faster-Local-Voice-AI) — the original WebSocket client/server skeleton this is forked from.
- [OpenClaw](https://docs.openclaw.ai/) — the agent runtime; this project just provides voice I/O.
- [Deepgram](https://deepgram.com/) — Nova-3 STT and Aura TTS.
- [face_recognition](https://github.com/ageitgey/face_recognition) — dlib bindings that made the identity check a one-liner.

## License

MIT — see [LICENSE](LICENSE).
