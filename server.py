import asyncio
import signal
import websockets
import json
import os
import re
import time
import numpy as np
import httpx
from dotenv import load_dotenv
from utils import load_config

load_dotenv()
DEEPGRAM_API_KEY = os.environ['DEEPGRAM_API_KEY']

# Bot/openclaw connection settings come from config (see BotWorker).

def dg_stt_url(endpointing_ms, keyterms=()):
    base = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&encoding=linear16"
        "&sample_rate=16000"
        "&channels=1"
        "&punctuate=true"
        f"&endpointing={endpointing_ms}"
        "&interim_results=false"
        "&smart_format=true"
    )
    from urllib.parse import quote
    for term in keyterms:
        if term:
            base += f"&keyterm={quote(term)}"
    return base
def dg_tts_url(voice):
    return (
        "https://api.deepgram.com/v1/speak"
        f"?model={voice}"
        "&encoding=linear16"
        "&sample_rate=48000"
        "&container=wav"
    )

LOW_EFFORT_UTTERANCES = {"huh", "uh", "um", "erm", "hmm", "the", "but"}

# Wake mode is configured per session (config.wake_mode):
#   "face"   — face presence opens/refreshes the window (FaceWatcher only)
#   "phrase" — utterance must start with config.wake_phrase to open the window
#              (phrase is also stripped from the payload before sending to the bot)
#   "either" — face presence or wake phrase will open/refresh the window
WAKE_WINDOW_SECS = 300  # default; overridable via config "wake_window_secs"

REFERENCE_FACE_PATH = "reference_face.npy"
# USER_NAME (display only) and GREETING_TEXT (spoken via TTS) are loaded
# from config in main(). The greeting may be spelled phonetically (e.g.
# "Mar-Wick") to nudge Aura's pronunciation without affecting log output.
USER_NAME = "User"
GREETING_TEXT = "Hello"
FACE_CHECK_INTERVAL = 2.0
FACE_COOLDOWN_SECS = 300
FACE_TOLERANCE = 0.55
CAMERA_INDEX = 0


_PUNCT_BETWEEN = r'[\s,.\-!?:;]*'
_wake_re_cache = {}


def _wake_re(phrase):
    re_obj = _wake_re_cache.get(phrase)
    if re_obj is None:
        words = phrase.strip().split()
        pat = r'^\s*' + _PUNCT_BETWEEN.join(re.escape(w) for w in words) + _PUNCT_BETWEEN
        re_obj = re.compile(pat, re.IGNORECASE)
        _wake_re_cache[phrase] = re_obj
    return re_obj


def strip_wake_phrase(text, phrase):
    """Return the text after the configured wake phrase (case-insensitive,
    tolerant of punctuation between words), or None if no match."""
    m = _wake_re(phrase).match(text)
    if not m:
        return None
    return text[m.end():].strip()


def clean_response(text):
    import re
    text = text.replace('’', "'").replace('`', "'").replace("''", "'")
    text = re.sub(r'[\U0001F000-\U0001FFFF\U00002700-\U000027BF\U00002600-\U000026FF]+', '', text)
    text = re.sub(r"[\*]+", '', text)
    text = re.sub(r"\(.*?\)", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


class BotWorker:
    """Persistent openclaw ACP bridge — one Node process, many turns."""

    def __init__(self, bin_path, agent_id, cwd, thought_level="off",
                 timeout_secs=120, identity_path=None,
                 user_name=None, user_name_spoken=None,
                 workspace_docs=None):
        self.bin_path = bin_path
        self.agent_id = agent_id
        self.cwd = cwd
        # thought_level is stored for future use; the active value is set in
        # the openclaw agent config (gateway side). Surfacing here lets us
        # call ACP set-options once the protocol method is wired up.
        self.thought_level = thought_level
        self.timeout_secs = timeout_secs
        # ACP sessions don't carry the agent's persona/IDENTITY config from
        # the gateway, so we boot it in-band on session/new.
        self.identity_path = identity_path
        self.user_name = user_name
        self.user_name_spoken = user_name_spoken
        # Workspace docs (e.g. AGENTS.md, TOOLS.md, USER.md) the agent
        # should read at session start. If empty, only IDENTITY is injected.
        self.workspace_docs = list(workspace_docs or [])
        self.proc = None
        self.session_id = None
        self._next_id = 0
        self._pending = {}
        self._chunks = []
        self._reader_task = None
        self._lock = asyncio.Lock()

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            self.bin_path, 'acp',
            '--session', f'agent:{self.agent_id}:{self.agent_id}',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())

        await self._request('initialize', {'protocolVersion': 1})
        sess = await self._request('session/new',
                                   {'cwd': self.cwd, 'mcpServers': []})
        self.session_id = sess['result']['sessionId']
        print(f"[BotWorker] agent={self.agent_id} cwd={self.cwd} "
              f"session_id={self.session_id}")
        await self._boot_identity()

    async def _boot_identity(self):
        if not self.identity_path or not os.path.exists(self.identity_path):
            return
        try:
            with open(self.identity_path) as f:
                identity_md = f.read().strip()
        except Exception as e:
            print(f"\033[38;5;196m[BotWorker] failed reading identity: {e}\033[0m")
            return
        pronunciation_note = ""
        if self.user_name and self.user_name_spoken \
                and self.user_name_spoken != self.user_name:
            pronunciation_note = (
                f"\nPronunciation note: when addressing the user by name in "
                f"this voice/TTS session, use the phonetic spelling "
                f"'{self.user_name_spoken}' (with the hyphen — it cues the "
                f"TTS engine). The display name is '{self.user_name}'.\n"
            )
        operator = self.user_name or "the operator"
        workspace_block = ""
        if self.workspace_docs:
            file_list = "\n".join(f"  {i+1}. {p}"
                                  for i, p in enumerate(self.workspace_docs))
            workspace_block = (
                "Now run your session-startup ritual. You MUST actually "
                f"read these files from {self.cwd}/ before replying — do "
                "not claim you've read them without doing the tool call:\n"
                f"{file_list}\n\n"
                "These document the workspace conventions, tools, and "
                "user context for this agent.\n\n"
                "After reading them, reply with a SHORT proof-of-read in "
                "this exact format (one line, no markdown):\n"
                "  ready | loaded: <comma-separated list of the "
                "capability/section names you found across those files>\n\n"
                "Listing the capabilities by name forces you to actually "
                "parse the files so future turns have those capabilities "
                "in your working context. A bare 'ready' without the list "
                "is wrong."
            )
        else:
            workspace_block = "Reply with only the word 'ready' to confirm."

        boot = (
            "Operator context for this session. You are running as the "
            f"OpenClaw agent named '{self.agent_id}'. Below is the "
            f"IDENTITY.md that {operator} has configured for this agent "
            "slot — this is your actual configured display name and role "
            "for this deployment, not a roleplay request. Use this name "
            "when asked who you are, and let the described role frame "
            "your responses. You remain Claude, built by Anthropic.\n\n"
            "--- IDENTITY.md ---\n"
            f"{identity_md}\n"
            "--- end IDENTITY.md ---\n"
            f"{pronunciation_note}\n"
            f"{workspace_block}"
        )
        ack = await self.prompt(boot)
        print(f"[BotWorker] identity boot → {ack[:60]!r}")

    async def _read_stdout(self):
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue
                if 'id' in msg and ('result' in msg or 'error' in msg):
                    fut = self._pending.pop(msg['id'], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                else:
                    update = msg.get('params', {}).get('update', {})
                    kind = update.get('sessionUpdate', '')
                    if kind == 'agent_message_chunk':
                        content = update.get('content', {})
                        text = content.get('text', '') if isinstance(content, dict) else ''
                        if text:
                            self._chunks.append(text)
        except Exception as e:
            print(f"\033[38;5;196m[BotWorker reader] {e}\033[0m")

    async def _request(self, method, params):
        _id = self._new_id()
        fut = asyncio.get_event_loop().create_future()
        self._pending[_id] = fut
        msg = {'jsonrpc': '2.0', 'id': _id, 'method': method, 'params': params}
        self.proc.stdin.write((json.dumps(msg) + '\n').encode())
        await self.proc.stdin.drain()
        return await fut

    async def prompt(self, text):
        async with self._lock:
            self._chunks = []
            try:
                resp = await asyncio.wait_for(
                    self._request('session/prompt', {
                        'sessionId': self.session_id,
                        'prompt': [{'type': 'text', 'text': text}],
                    }),
                    timeout=self.timeout_secs,
                )
            except asyncio.TimeoutError:
                print(f"\033[38;5;196m[Bot] prompt timeout after {self.timeout_secs}s\033[0m")
                return ""
            if 'error' in resp:
                print(f"\033[38;5;196m[Bot] error: {resp['error']}\033[0m")
                return ""
            joined = ''.join(self._chunks)
            return clean_response(joined)

    async def close(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.proc.kill()


_bot_worker = None
_bot_worker_kwargs = None  # populated by main() from config


async def get_bot_worker():
    global _bot_worker
    if _bot_worker is None:
        if _bot_worker_kwargs is None:
            raise RuntimeError("BotWorker config not set; call configure_bot_worker() first")
        _bot_worker = BotWorker(**_bot_worker_kwargs)
        await _bot_worker.start()
    return _bot_worker


def configure_bot_worker(**kwargs):
    global _bot_worker_kwargs
    _bot_worker_kwargs = kwargs


async def query_bot(user_text):
    worker = await get_bot_worker()
    return await worker.prompt(user_text)


class FaceWatcher:
    """Long-lived camera + face-recognition loop.

    Fires `on_first_match` once per "presence session" (cold-start greeting),
    and `on_present` on every subsequent matching frame so the wake window
    can be refreshed while the user stays in view. A presence session ends
    after `absence_grace_secs` of consecutive non-matches.
    """

    def __init__(self, reference_path, on_first_match, on_present,
                 absence_grace_secs=300.0):
        self.reference_path = reference_path
        self.on_first_match = on_first_match
        self.on_present = on_present
        self.absence_grace_secs = absence_grace_secs
        self.reference = None
        self._cap = None
        self._task = None
        self._present = False
        # Greeting fires once per connection; brief drops (head turns, hand
        # blocks lens) refresh wake without replaying the greeting.
        self._greeted = False
        self._last_seen_at = 0.0
        self._stopped = False

    def _load_reference(self):
        if not os.path.exists(self.reference_path):
            print(f"\033[38;5;240m[Face] no {self.reference_path}; recognition disabled\033[0m")
            return False
        try:
            self.reference = np.load(self.reference_path)
            print(f"\033[38;5;245m[Face] reference loaded shape={self.reference.shape}\033[0m")
            return True
        except Exception as e:
            print(f"\033[38;5;196m[Face] failed to load reference: {e}\033[0m")
            return False

    async def start(self):
        if not self._load_reference():
            return
        loop = asyncio.get_event_loop()
        import cv2
        self._cap = await loop.run_in_executor(
            None, lambda: cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2))
        if not self._cap.isOpened():
            print("\033[38;5;196m[Face] failed to open camera\033[0m")
            self._cap = None
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._task = asyncio.create_task(self._loop())
        print("\033[38;5;245m[Face] watcher started\033[0m")

    async def _loop(self):
        loop = asyncio.get_event_loop()
        try:
            while not self._stopped:
                await asyncio.sleep(FACE_CHECK_INTERVAL)
                matched = await loop.run_in_executor(None, self._check_match)
                now = time.time()
                if matched:
                    self._last_seen_at = now
                    if not self._present:
                        self._present = True
                        if not self._greeted:
                            print(f"\033[38;5;82m[Face] {USER_NAME} arrived\033[0m")
                            self._greeted = True
                            cb = self.on_first_match
                        else:
                            print(f"\033[38;5;82m[Face] {USER_NAME} reacquired\033[0m")
                            cb = self.on_present
                    else:
                        cb = self.on_present
                    try:
                        await cb()
                    except Exception as e:
                        print(f"\033[38;5;196m[Face] callback error: {e}\033[0m")
                elif self._present and (now - self._last_seen_at) > self.absence_grace_secs:
                    self._present = False
                    print(f"\033[38;5;240m[Face] {USER_NAME} left\033[0m")
        except asyncio.CancelledError:
            pass

    def _check_match(self):
        import cv2
        import face_recognition
        for _ in range(2):
            self._cap.grab()
        ok, frame = self._cap.read()
        if not ok:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model='hog')
        if not locs:
            return False
        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            return False
        # face_recognition.compare_faces returns list of bool per known-face
        return bool(face_recognition.compare_faces(
            [self.reference], encs[0], tolerance=FACE_TOLERANCE)[0])

    async def stop(self):
        self._stopped = True
        if self._task:
            self._task.cancel()
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass


async def aura_speak(text, voice):
    """Returns 48kHz stereo int16 PCM bytes (no WAV header)."""
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            dg_tts_url(voice),
            headers={
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={'text': text},
        )
        resp.raise_for_status()
        wav_bytes = resp.content
    pcm_mono = wav_bytes[44:]  # strip WAV header
    arr = np.frombuffer(pcm_mono, dtype=np.int16)
    stereo = np.repeat(arr, 2)
    return stereo.tobytes()


async def process_connection(client_ws):
    transcript_queue = asyncio.Queue()
    dg_ws = None
    listener_task = None
    bot_task = None
    keepalive_task = None
    face_watcher = None
    aura_voice = "aura-asteria-en"
    wake_mode = "face"
    wake_phrase = "hey girl"
    stt_endpointing_ms = 1000
    stt_keyterms = []
    wake_window_secs = float(WAKE_WINDOW_SECS)

    async def dg_keepalive():
        try:
            while True:
                await asyncio.sleep(8)
                if dg_ws is not None:
                    try:
                        await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                    except websockets.ConnectionClosed:
                        return
        except asyncio.CancelledError:
            pass

    async def dg_listener():
        try:
            async for raw in dg_ws:
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if data.get('type') != 'Results':
                    continue
                if not data.get('is_final'):
                    continue
                alts = data.get('channel', {}).get('alternatives', [])
                text = (alts[0].get('transcript', '') if alts else '').strip()
                if text:
                    await transcript_queue.put(text)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            print(f"\033[38;5;196m[DG listener] {e}\033[0m")

    wake_until = 0.0
    streaming_to_dg = True  # reflected to console only on transitions

    async def bot_loop():
        nonlocal wake_until
        phrase_can_wake = wake_mode in ('phrase', 'either')
        while True:
            user_text = await transcript_queue.get()
            cleaned = user_text.lower().strip(".,!? ")
            if cleaned in LOW_EFFORT_UTTERANCES:
                continue
            now = time.time()
            in_window = now < wake_until
            stripped = strip_wake_phrase(user_text, wake_phrase)
            has_phrase = stripped is not None

            if has_phrase and phrase_can_wake and not in_window:
                wake_until = now + wake_window_secs
                if not stripped:
                    print(f"\033[38;5;240m[Wake] phrase only, no command\033[0m")
                    continue
                payload, source = stripped, "phrase"
            elif in_window:
                payload = stripped if has_phrase else user_text
                source = f"window({wake_until-now:.0f}s left)"
            else:
                reason = "asleep" if phrase_can_wake else "face not present"
                print(f"\033[38;5;240m[Gate] ignored ({reason}): {user_text}\033[0m")
                continue
            if not payload.strip():
                continue
            t_user = time.time()
            print(f"\033[38;5;35m[User/{source}]: {user_text}\033[0m")
            user_text = payload

            t0 = time.time()
            # Voice-mode directive — IDENTITY.md is buried in ~38KB of system
            # prompt and the channel hint there gets outweighed. Inline
            # instruction is short but reliably honored.
            voice_prompt = (
                "(Voice mode. Reply in plain spoken prose — NO markdown, "
                "lists, bullets, or headers — your reply will be spoken "
                "aloud. Default length is 1-2 sentences; go longer only "
                "when the user explicitly asks for a list, summary, "
                "briefing, or explanation (e.g. the morning AI briefing is "
                "~150 words). Act on requests directly: if a request maps "
                "to an available tool, call the tool and report the "
                "outcome — do not ask 'should I…?' or 'do you want me "
                "to…?' for routine actions; the user has standing "
                "authorization. Confirm only for genuinely destructive or "
                "high-blast-radius actions. The user may reference earlier "
                "turns in this same conversation with pronouns like 'that', "
                "'it', 'the first one'; check the prior turns in your "
                "context before claiming you don't know what they mean — "
                "you almost certainly do.) "
                f"{user_text}"
            )
            response = await query_bot(voice_prompt)
            t_bot = time.time() - t0
            if not response:
                response = "Sorry, I didn't catch that. Could you repeat?"
            print(f"\033[38;5;75m[AI] ({t_bot:.1f}s): {response[:200]}\033[0m")

            t1 = time.time()
            audio = await aura_speak(response, aura_voice)
            t_tts = time.time() - t1
            t_total = time.time() - t_user
            print(f"\033[38;5;220m[Perf] bot={t_bot:.1f}s tts={t_tts:.1f}s total={t_total:.1f}s\033[0m")

            CHUNK = 4096
            for i in range(0, len(audio), CHUNK):
                await client_ws.send(audio[i:i+CHUNK])
            await client_ws.send("__END__")
            wake_until = time.time() + wake_window_secs

    try:
        async for message in client_ws:
            if isinstance(message, str):
                if message.strip() == "__done__":
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if data.get('type') == 'config_sync':
                    cfg = data.get('config', {}) or {}
                    aura_voice = cfg.get('aura_voice', aura_voice)
                    wake_mode = cfg.get('wake_mode', wake_mode)
                    if wake_mode not in ('face', 'phrase', 'either'):
                        print(f"[Server] unknown wake_mode={wake_mode!r}, defaulting to 'face'")
                        wake_mode = 'face'
                    wake_phrase = cfg.get('wake_phrase', wake_phrase)
                    wake_window_secs = float(cfg.get('wake_window_secs', wake_window_secs))
                    stt_endpointing_ms = cfg.get('stt_endpointing_ms', stt_endpointing_ms)
                    stt_keyterms = cfg.get('stt_keyterms', stt_keyterms) or []
                    print(f"[Server] Config synced (voice={aura_voice}, "
                          f"wake={wake_mode}, phrase={wake_phrase!r}, "
                          f"endpointing={stt_endpointing_ms}ms, "
                          f"keyterms={stt_keyterms}); opening Deepgram STT stream")
                    dg_ws = await websockets.connect(
                        dg_stt_url(stt_endpointing_ms, stt_keyterms),
                        extra_headers={'Authorization': f'Token {DEEPGRAM_API_KEY}'},
                    )
                    listener_task = asyncio.create_task(dg_listener())
                    keepalive_task = asyncio.create_task(dg_keepalive())
                    bot_task = asyncio.create_task(bot_loop())

                    async def on_face_first():
                        nonlocal wake_until
                        print(f"\033[38;5;75m[AI greeting]: {GREETING_TEXT}\033[0m")
                        audio = await aura_speak(GREETING_TEXT, aura_voice)
                        CHUNK = 4096
                        for i in range(0, len(audio), CHUNK):
                            await client_ws.send(audio[i:i+CHUNK])
                        await client_ws.send("__END__")
                        wake_until = time.time() + wake_window_secs

                    async def on_face_present():
                        nonlocal wake_until
                        wake_until = time.time() + wake_window_secs

                    if wake_mode in ('face', 'either'):
                        face_grace = float(cfg.get('face_absence_grace_secs', 300.0))
                        face_watcher = FaceWatcher(
                            REFERENCE_FACE_PATH,
                            on_first_match=on_face_first,
                            on_present=on_face_present,
                            absence_grace_secs=face_grace,
                        )
                        await face_watcher.start()
                continue

            if not isinstance(message, bytes):
                continue
            if dg_ws is None:
                continue
            # Cost gate: in face-only mode, audio doesn't open the window
            # (only the camera does), so streaming bytes to Deepgram while
            # the wake window is closed is paying for STT we'll throw away.
            # Phrase / either modes must keep streaming so the wake phrase
            # can be heard.
            if wake_mode == 'face' and time.time() >= wake_until:
                if streaming_to_dg:
                    print("\033[38;5;240m[DG] mic stream paused "
                          "(no wake window; face-only mode)\033[0m")
                    streaming_to_dg = False
                continue
            if not streaming_to_dg:
                print("\033[38;5;245m[DG] mic stream resumed\033[0m")
                streaming_to_dg = True
            try:
                await dg_ws.send(message)
            except websockets.ConnectionClosed:
                print("[DG] STT WS closed; reconnecting")
                dg_ws = await websockets.connect(
                    dg_stt_url(stt_endpointing_ms, stt_keyterms),
                    extra_headers={'Authorization': f'Token {DEEPGRAM_API_KEY}'},
                )
                for t in (listener_task, keepalive_task):
                    if t:
                        t.cancel()
                listener_task = asyncio.create_task(dg_listener())
                keepalive_task = asyncio.create_task(dg_keepalive())
    finally:
        for t in (listener_task, bot_task, keepalive_task):
            if t:
                t.cancel()
        if face_watcher:
            await face_watcher.stop()
        if dg_ws:
            try:
                await dg_ws.close()
            except Exception:
                pass


async def warm_up_bot():
    print("[Server] Pinging bot...")
    try:
        response = await query_bot("Reply with only: ready")
        print(f"[Server] Bot ready: '{response[:60]}'")
    except Exception as e:
        print(f"[Server] Bot warm-up failed: {e}")


async def shutdown():
    """Tear down long-lived resources on exit."""
    global _bot_worker
    if _bot_worker is not None:
        try:
            await _bot_worker.close()
            print("[Server] BotWorker closed")
        except Exception as e:
            print(f"[Server] BotWorker close error: {e}")
        _bot_worker = None


async def main():
    global USER_NAME, GREETING_TEXT
    config = load_config()
    USER_NAME = config["user_name"]
    GREETING_TEXT = config["greeting"]
    configure_bot_worker(
        bin_path=config["bot_bin"],
        agent_id=config["bot_agent"],
        cwd=config["bot_cwd"],
        thought_level=config["bot_thought_level"],
        timeout_secs=config["bot_timeout_secs"],
        identity_path=config.get("bot_identity_path"),
        user_name=config.get("user_name"),
        user_name_spoken=config.get("user_name_spoken"),
        workspace_docs=config.get("workspace_docs", []),
    )
    await warm_up_bot()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda s=sig: (
                print(f"\n[Server] {s.name} received, shutting down..."),
                stop_event.set(),
            ),
        )

    print("[Server] Listening on ws://0.0.0.0:8765 ...")
    async with websockets.serve(process_connection, "0.0.0.0", 8765,
                                ping_timeout=None, ping_interval=None):
        await stop_event.wait()
    await shutdown()
    print("[Server] Bye.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Backstop: signal handler should normally win, but on some
        # platforms a fast Ctrl-C during startup can still surface here.
        pass
