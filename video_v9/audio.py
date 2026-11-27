"""V10 étape 3 : voix + funk Artlist (aperçu du catalogue) + vrais bruitages Artlist posés sur les mots clés,
ducking de la musique sous la voix, loudnorm -14 LUFS / -1,5 dBTP.
usage: python3 -I audio.py <source.mp4> <timing.json> <out.wav> <tmpdir> <dossier sons artlist>"""
import sys, json, subprocess, wave, numpy as np
from scipy import signal
SR = 44100
SRC, TJ, OUT, TMP, ART = sys.argv[1:6]
T = json.load(open(TJ)); FPS = T["fps"]; TOTAL = T["total"]
SEGS = [(1.78, 200), (23.12, 203), (35.78, 60), (38.21, 81), (41.08, 176)]
def dec(path, ss=None, t=None):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", str(ss)] if ss else []) + (["-t", str(t)] if t else []) + ["-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32).copy()
def lp(x, f, o=2): return signal.sosfilt(signal.butter(o, f, "lp", fs=SR, output="sos"), x)
# ---------- voix : plans bout à bout (coupes dans les silences, fondu 12 ms)
v = dec(SRC); parts = []
for s, n in SEGS:
    a = v[int(round(s * SR)): int(round(s * SR)) + int(round(n / FPS * SR))].copy(); f = int(.012 * SR)
    a[:f] *= np.linspace(0, 1, f); a[-f:] *= np.linspace(1, 0, f); parts.append(a)
voice = np.concatenate(parts); NS = len(voice)
voice = signal.sosfilt(signal.butter(2, 90, "hp", fs=SR, output="sos"), voice)
env = np.sqrt(lp(voice ** 2, 12, 1) + 1e-9); gain = np.clip((0.12 / (env + 1e-6)) ** 0.45, 0.6, 3.0)
voice = np.tanh(voice * gain * 1.1) * 0.9
voice = voice + 0.12 * signal.sosfilt(signal.butter(2, [2500, 5000], "bp", fs=SR, output="sos"), voice)
# ---------- sons Artlist
def load(name, maxd=None):
    x = dec(f"{ART}/{name}.aac")
    if maxd: x = x[: int(maxd * SR)]; k = int(.12 * SR); x[-k:] *= np.linspace(1, 0, k)
    return x / (np.abs(x).max() + 1e-9)
SFX = dict(whoosh=load("whoosh_81139"), impact=load("impact_86684", 2.0), whoosh2=load("whoosh_124627", 1.6),
           glitch=load("glitch_131289", 1.1), glitch2=load("glitch_85624"), pop=load("pop_111659"), paper=load("paper_60800"))
def pitch(x, r): return np.interp(np.arange(0, len(x) - 1, r), np.arange(len(x)), x)
mix_music = dec(f"{ART}/funk_dealing.aac", 0, TOTAL + 1.5)          # début du morceau (funk commercial, 105 bpm)
music = np.zeros(NS); m = mix_music[:NS]; music[:len(m)] = m; music /= (np.abs(music).max() + 1e-9)
venv = np.sqrt(np.convolve(voice ** 2, np.ones(int(.06 * SR)) / int(.06 * SR), "same"))
duck = lp(1 - 0.6 * np.clip(venv / 0.08, 0, 1), 6, 1)
fade = np.minimum(1, np.arange(NS) / (.3 * SR)) * np.minimum(1, (NS - np.arange(NS)) / (.9 * SR))
mix = voice + music * duck * fade * 0.30
def put(x, t, g, align="peak"):
    sm = np.convolve(np.abs(x), np.ones(int(.02 * SR)) / int(.02 * SR), "same")
    off = int(np.argmax(sm)) if align == "peak" else 0              # le pic (ou le début) tombe pile sur le mot clé
    i = int(t * SR) - off
    if i < 0: x = x[-i:]; i = 0
    j = min(NS, i + len(x)); mix[i:j] += x[:j - i] * g
for e in T["events"]:
    t, k = e["t"], e["kind"]
    if k == "plan1": put(SFX["whoosh"], t - .03, .5); put(SFX["impact"], t, .6)
    elif k == "ultime": put(SFX["whoosh2"], t, .4)
    elif k == "check": put(pitch(SFX["pop"], 1 + .07 * e["n"]), t, .55, "start")
    elif k == "porcs": put(SFX["glitch"], t, .45, "start")
    elif k == "plan2": put(SFX["whoosh"], t - .03, .5); put(SFX["impact"], t, .65); put(SFX["glitch2"], t + .02, .25, "start")
    elif k == "toi": put(SFX["paper"], t - .08, .3)
np.save(f"{TMP}/mix_raw.npy", mix)
def wavout(x, path):
    st = np.stack([np.clip(x, -1, 1)] * 2, 1); pcm = (st * 32767).astype("<i2")
    with wave.open(path, "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
wavout(mix * 0.7, f"{TMP}/pre.wav")
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", f"{TMP}/pre.wav", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(r[r.rindex("{"):r.rindex("}") + 1])
af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}"
      f":measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=44100")
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", f"{TMP}/pre.wav", "-af", af, "-ar", "44100", OUT], check=True)
print("audio ok", OUT, "pre-norm I =", j["input_i"])
