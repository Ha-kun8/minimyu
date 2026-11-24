# -*- coding: utf-8 -*-
"""
ミニミュー ウェイクワード常駐プログラム
「ミニミュー」と声をかけると起きて、続く言葉を Open WebUI のミニミュー(makase)に送り、
返事を音声で読み上げる。PCのマイクとスピーカーを使う自宅専用の仕組み。

使い方:
  1. setup.bat を実行（初回だけ。依存ライブラリを入れる）
  2. run.bat を実行
  3. 初回だけ Open WebUI のログイン情報（メールとパスワード）を聞かれる
     → config.json に保存される（このPCの中だけ）
  4. 「ミニミュー」と話しかける → ピッと鳴ったら続けて用件を言う
     同じ一息で「ミニミュー、今日の天気は？」でもOK
  5. やめるときは Ctrl+C
"""
import asyncio
import getpass
import json
import os
import re
import sys
import tempfile
import threading
import time
import queue
import unicodedata

import numpy as np
import requests
import sounddevice as sd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
OWUI_URL = "http://localhost:8080"
MODEL_ID = "makase"
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000

WAKE_WORDS = ["ミニミュー", "ミニミュウ", "ミニみゅー", "みにみゅー", "みにみゅう", "ミューズ", "ミュース", "みゅーず", "みゅーす"]

# ---------------------------------------------------------------- 状態
state = {
    "token": None,
    "history": [],          # Open WebUIへ送る会話履歴
    "whisper_tiny": None,   # ウェイクワード検知用
    "whisper_base": None,   # 用件の文字起こし用
    "noise_floor": 0.01,    # 環境ノイズの目安（起動時に測る）
    "mic_index": None,
    "speaking_text": "",    # いま読み上げている文（エコー除け用）
    "barge_rest": None,     # 割り込み時にウェイクワードの後ろに聞こえた用件
    "barge_event": threading.Event(),   # 読み上げ中にウェイクワードを検知したら立つ
    "monitor_stop": threading.Event(),
    "monitor_thread": None,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 設定
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    print("初回設定: Open WebUI のログイン情報を入力してください（このPCにだけ保存されます）")
    email = input("メールアドレス: ").strip()
    password = getpass.getpass("パスワード: ")
    cfg = {"email": email, "password": password, "mic_index": None}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def signin(cfg):
    r = requests.post(f"{OWUI_URL}/api/v1/auths/signin",
                      json={"email": cfg["email"], "password": cfg["password"]}, timeout=30)
    r.raise_for_status()
    state["token"] = r.json()["token"]
    log("Open WebUI にログインしました")


# ---------------------------------------------------------------- 音声まわり
def calibrate_noise(mic_index):
    """起動直後の静かな2秒で環境ノイズを測る"""
    log("周りの音を測っています（2秒、静かにしてください）…")
    rec = sd.rec(int(SAMPLE_RATE * 2), samplerate=SAMPLE_RATE, channels=1,
                 dtype="float32", device=mic_index)
    sd.wait()
    rms = float(np.sqrt(np.mean(rec ** 2)))
    state["noise_floor"] = max(rms, 0.004)
    log(f"ノイズ基準: {state['noise_floor']:.4f}（しきい値 {threshold():.4f}）")


def threshold():
    return state["noise_floor"] * 3.0


def record_segment(mic_index, max_sec=15.0, end_silence_sec=1.0, start_timeout=None):
    """発話1区間を録る。声が出たら始まり、無音が続いたら終わる。
    音声が始まらなければ None、始まれば float32 配列を返す。"""
    th = threshold()
    frames = []
    preroll = []
    speaking = False
    silence_run = 0
    voiced_run = 0
    waited = 0.0
    end_silence_frames = int(end_silence_sec * 1000 / FRAME_MS)
    max_frames = int(max_sec * 1000 / FRAME_MS)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_LEN, device=mic_index) as stream:
        while True:
            data, _ = stream.read(FRAME_LEN)
            frame = data[:, 0].copy()
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if not speaking:
                if rms > th:
                    if voiced_run == 0 and preroll:
                        frames.extend(preroll)
                        preroll = []
                    voiced_run += 1
                    frames.append(frame)
                    if voiced_run >= 3:  # 90ms 声が続いたら発話開始
                        speaking = True
                        silence_run = 0
                else:
                    voiced_run = 0
                    frames.clear()
                    preroll.append(frame)
                    if len(preroll) > 10:
                        preroll.pop(0)
                    waited += FRAME_MS / 1000
                    if start_timeout and waited > start_timeout:
                        return None
                continue
            frames.append(frame)
            if rms > th:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= end_silence_frames:
                    break
            if len(frames) >= max_frames:
                break
    if not speaking or not frames:
        return None
    return np.concatenate(frames)


def beep(freq=880, sec=0.12):
    # ウェイクワード検知音「ピコん」: 2音の上昇チャイム
    # sounddevice経由は読み上げ（miniaudio）再生後に鳴らなくなるため winsound を使う
    import winsound
    winsound.Beep(1046, 90)
    winsound.Beep(1568, 150)


# ---------------------------------------------------------------- 文字起こし
def load_whisper():
    from faster_whisper import WhisperModel
    log("音声認識モデルを読み込み中（初回はダウンロードが入ります）…")

    def load(name):
        # まずGPU（CUDA）を試し、ダメならCPUに自動で戻す
        try:
            m = WhisperModel(name, device="cuda", compute_type="float16")
            # 推論まで通るか無音で試す（DLL不足は読み込みでなく推論で判明するため）
            for _ in m.transcribe(np.zeros(SAMPLE_RATE, dtype="float32"),
                                  language="ja")[0]:
                pass
            log(f"{name}: GPUで読み込み")
            return m
        except Exception as e:
            log(f"{name}: GPUが使えないためCPUで読み込みます ({e})")
            return WhisperModel(name, device="cpu", compute_type="int8")

    state["whisper_base"] = load("base")
    # 用件の文字起こし用に、ひと回り大きいモデルも用意する（失敗したら base で代用）
    try:
        state["whisper_small"] = load("small")
        log("読み込み完了（用件用: small）")
    except Exception as e:
        state["whisper_small"] = None
        log(f"small モデルの読み込みに失敗、base で代用します: {e}")
        log("読み込み完了")


def transcribe(audio, model, fast=False):
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.01:
        audio = audio * (0.9 / peak)
    # 語尾対策1: 末尾に無音を足す（音声が語尾でぷつっと切れると最後の音を落とすため）
    pad = np.zeros(int(SAMPLE_RATE * 0.4), dtype="float32")
    audio = np.concatenate([audio, pad])
    if fast:
        segments, _ = model.transcribe(audio, language="ja", beam_size=1,
                                       vad_filter=False)
    else:
        # 語尾対策2: ビーム幅を広げ、話し言葉の語尾（〜？／〜かな等）を前提にした
        # プロンプトで、語尾上がりの文末まで拾いやすくする
        segments, _ = model.transcribe(
            audio, language="ja", beam_size=3, vad_filter=False,
            initial_prompt="今日の天気は？　それ、ちょっと調べてほしいんだけど。うん、わかったよ。")
    return "".join(s.text for s in segments).strip()


# ---------------------------------------------------------------- ウェイクワード
def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s、。,．・!！?？「」『』（）()\[\]…]", "", text)
    # カタカナ→ひらがなに寄せて照合
    text = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    return text


WAKE_NORM = [normalize(w) for w in WAKE_WORDS]


def find_wake(text):
    """ウェイクワードが含まれていたら (位置, 長さ) を返す。なければ None"""
    n = normalize(text)
    for w in WAKE_NORM:
        i = n.find(w)
        if i >= 0:
            return i, len(w), n
    # 完全一致しなければ、あいまい照合（短い発話だけ対象）
    import difflib
    if 2 <= len(n) <= 10:
        for w in WAKE_NORM:
            if difflib.SequenceMatcher(None, n, w).ratio() >= 0.6:
                return 0, len(n), n
    return None, None, n


# ---------------------------------------------------------------- 割り込み監視
def watch_barge():
    """読み上げ中にマイクを監視し、ウェイクワードを聞いたら割り込みを立てる。
    自分の読み上げ音声（エコー）では誤発しないよう、読み上げ中の文と
    ほぼ一致する聞き取りは無視する。"""
    while not state["monitor_stop"].is_set():
        seg = record_segment(state["mic_index"], max_sec=3.0,
                             end_silence_sec=0.4, start_timeout=1.0)
        if seg is None:
            continue
        heard = transcribe(seg, state["whisper_base"], fast=True)
        if not heard:
            continue
        sp = normalize(state.get("speaking_text") or "")
        hn = normalize(heard)
        if sp and hn and (hn in sp or sp in hn):
            continue  # 読み上げのエコー
        pos, wlen, norm = find_wake(heard)
        if pos is None:
            continue
        if state["monitor_stop"].is_set():
            return
        state["barge_rest"] = norm[pos + wlen:].strip("ー ")
        state["barge_event"].set()
        log(f"割り込み検知: 「{heard}」")
        return


# ---------------------------------------------------------------- ミニミューと会話
def ask_minimyu(text):
    headers = {"Authorization": f"Bearer {state['token']}"}
    msgs = state["history"][-12:] + [{"role": "user", "content": text + " /no_think"}]
    r = requests.post(f"{OWUI_URL}/api/chat/completions", headers=headers,
                      json={"model": MODEL_ID, "messages": msgs, "stream": True},
                      timeout=600, stream=True)
    r.raise_for_status()
    # 読み上げ中の割り込み（バージイン）監視を開始
    state["barge_event"].clear()
    state["barge_rest"] = None
    state["monitor_stop"].clear()
    state["monitor_thread"] = threading.Thread(target=watch_barge, daemon=True)
    state["monitor_thread"].start()
    # 読み上げパイプライン: 文章ができ次第キューに入れ、音声合成と再生を
    # 別スレッドで並行に進める（文と文の間の「合成待ち」の間をなくす）
    text_q = queue.Queue()
    audio_q = queue.Queue()

    def synth_worker():
        try:
            while True:
                sent = text_q.get()
                if sent is None:
                    return
                if state["barge_event"].is_set():
                    continue
                item = synth_to_file(sent)
                if item:
                    audio_q.put(item)
        finally:
            audio_q.put(None)

    def play_worker():
        while True:
            item = audio_q.get()
            if item is None:
                return
            sent, path = item
            play_file(sent, path)

    synth_t = threading.Thread(target=synth_worker, daemon=True)
    play_t = threading.Thread(target=play_worker, daemon=True)
    synth_t.start()
    play_t.start()
    full = ""
    buf = ""
    for raw in r.iter_lines():
        if state["barge_event"].is_set():
            break
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
        except Exception:
            continue
        full += delta
        buf += delta
        while True:
            m = re.search(r"[。！？\n]", buf)
            if not m or m.end() < 8:
                break
            sent, buf = buf[:m.end()], buf[m.end():]
            text_q.put(sent)
    if buf.strip():
        text_q.put(buf)
    text_q.put(None)
    synth_t.join()
    play_t.join()
    r.close()
    state["monitor_stop"].set()
    if state["monitor_thread"] is not None:
        state["monitor_thread"].join(timeout=4)
    reply = full.strip()
    state["history"] += [{"role": "user", "content": text},
                         {"role": "assistant", "content": reply}]
    return reply


def clean_for_speech(text):
    text = re.sub(r"\[(\d+)\]", "", text)                 # 出典 [1]
    text = re.sub(r"https?://\S+", "", text)              # URL
    text = re.sub(r"[*#`>_~|]", "", text)                 # マークダウン記号
    text = re.sub(r"\n+", "。", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 280:
        text = text[:280] + "。続きは画面で見て。"
    return text


async def _tts_to_file(text, path):
    import edge_tts
    communicate = edge_tts.Communicate(text, "ja-JP-NanamiNeural")
    await communicate.save(path)


def synth_to_file(text):
    """読み上げ用に整えた文を音声ファイル化する。(整えた文, パス) か None を返す"""
    text = clean_for_speech(text)
    if not text:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            path = tf.name
        asyncio.run(_tts_to_file(text, path))
        return text, path
    except Exception as e:
        log(f"音声合成に失敗: {e}")
        return None


def play_file(text, path):
    """音声ファイルを再生する。割り込みが立ったら即座に止める"""
    if state["barge_event"].is_set():
        try:
            os.unlink(path)
        except Exception:
            pass
        return
    state["speaking_text"] = text
    print(f"ミニミュー: {text}", flush=True)
    try:
        import miniaudio
        decoded = miniaudio.decode_file(path)
        pcm = np.frombuffer(decoded.samples, dtype=np.int16).astype("float32") / 32768.0
        if decoded.nchannels > 1:
            pcm = pcm.reshape(-1, decoded.nchannels).mean(axis=1)
        sd.play(pcm, decoded.sample_rate)
        while True:
            if state["barge_event"].is_set():
                sd.stop()
                break
            try:
                if not sd.get_stream().active:
                    break
            except Exception:
                break
            time.sleep(0.05)
    except Exception as e:
        log(f"読み上げに失敗（文章は上の通り）: {e}")
    state["speaking_text"] = ""
    try:
        os.unlink(path)
    except Exception:
        pass


def speak(text):
    if state["barge_event"].is_set():
        return  # 割り込み済みなら読まない
    item = synth_to_file(text)
    if item:
        sent, path = item
        play_file(sent, path)


# ---------------------------------------------------------------- メインループ
def main():
    cfg = load_config()
    mic_index = cfg.get("mic_index")
    state["mic_index"] = mic_index
    print("=== 入力デバイス一覧 ===")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = " ← 既定" if sd.default.device[0] == i else ""
            print(f"  {i}: {d['name']}{mark}")
    print("========================")
    for attempt in range(30):
        try:
            signin(cfg)
            break
        except Exception as e:
            log(f"Open WebUI への接続待ち ({attempt + 1}/30)")
            time.sleep(10)
    else:
        raise RuntimeError("Open WebUI に接続できませんでした")
    load_whisper()
    calibrate_noise(mic_index)
    log("待機中。「ミニミュー」と話しかけてください（Ctrl+Cで終了）")
    while True:
        state["barge_event"].clear()
        state["monitor_stop"].set()
        seg = record_segment(mic_index, max_sec=6.0, end_silence_sec=0.35)
        if seg is None:
            continue
        heard = transcribe(seg, state["whisper_base"], fast=True)
        if not heard:
            continue
        pos, wlen, norm = find_wake(heard)
        if pos is None:
            continue
        log(f"ウェイクワード検知: 「{heard}」")
        beep(880, 0.1)
        rest = norm[pos + wlen:].strip("ー ")
        if len(rest) >= 2:
            command = rest
        else:
            log("用件をどうぞ")
            # 語尾対策3: 文末の無音判定を少し待つ（語尾上がりは間が空いて切られやすい）
            seg2 = record_segment(mic_index, max_sec=15.0, end_silence_sec=0.9,
                                  start_timeout=10.0)
            if seg2 is None:
                log("聞き取れませんでした")
                continue
            cmd_model = state.get("whisper_small") or state["whisper_base"]
            command = transcribe(seg2, cmd_model)
        if not command:
            continue
        print(f"あなた: {command}", flush=True)
        if command in ("リセット", "りせっと", "会話をリセット"):
            state["history"] = []
            speak("会話をリセットしました")
            continue
        try:
            reply = ask_minimyu(command)
        except Exception as e:
            log(f"ミニミューへの送信に失敗: {e}")
            speak("うまく繋がりませんでした")
            continue
        # 読み上げ中に割り込みがあったら、そのまま次の用件を聞く
        while state["barge_event"].is_set():
            state["barge_event"].clear()
            beep(880, 0.1)
            rest = state.get("barge_rest") or ""
            state["barge_rest"] = None
            if len(rest) >= 2:
                command = rest
            else:
                log("用件をどうぞ")
                seg2 = record_segment(mic_index, max_sec=15.0, end_silence_sec=0.9,
                                      start_timeout=10.0)
                if seg2 is None:
                    log("聞き取れませんでした")
                    break
                cmd_model = state.get("whisper_small") or state["whisper_base"]
                command = transcribe(seg2, cmd_model)
            if not command:
                break
            print(f"あなた: {command}", flush=True)
            if command in ("リセット", "りせっと", "会話をリセット"):
                state["history"] = []
                speak("会話をリセットしました")
                break
            try:
                ask_minimyu(command)
            except Exception as e:
                log(f"ミニミューへの送信に失敗: {e}")
                speak("うまく繋がりませんでした")
                break
        log("待機中。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n終了します")
        sys.exit(0)
