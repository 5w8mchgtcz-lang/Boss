"""V9 étape 3 : bruitages synthétisés (whoosh tech/medium, boom, kaching, glitch, swoosh light),
funk 70s synthétisé, mixage voix + musique (ducking) + sfx sur mots clés, loudnorm -14 LUFS.
usage: python3 -I audio.py <source.mp4> <timing.json> <out.wav> <tmpdir>"""
import sys, json, subprocess, numpy as np
from scipy import signal
SR = 44100
SRC, TJ, OUT, TMP = sys.argv[1:5]
T = json.load(open(TJ)); FPS = T["fps"]; TOTAL = T["total"]
SEGS = [(1.63, 206), (23.10, 204), (35.70, 341)]
rng = np.random.default_rng(7)

def dec(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()
# ---------- voix : 3 plans bout à bout (coupes dans les silences, fondu 12 ms)
v = dec(SRC); parts = []
for s, n in SEGS:
    a = v[int(round(s * SR)): int(round(s * SR)) + int(round(n / FPS * SR))].copy(); f = int(.012 * SR)
    a[:f] *= np.linspace(0, 1, f); a[-f:] *= np.linspace(1, 0, f); parts.append(a)
voice = np.concatenate(parts); NS = len(voice)
# léger traitement voix : coupe-bas 90 Hz + compression douce + présence
sos = signal.butter(2, 90, "hp", fs=SR, output="sos"); voice = signal.sosfilt(sos, voice)
env = np.sqrt(signal.sosfilt(*[signal.butter(1, 12, "lp", fs=SR, output="sos")] * 1, voice ** 2) + 1e-9)
gain = np.clip((0.12 / (env + 1e-6)) ** 0.45, 0.6, 3.0); voice = np.tanh(voice * gain * 1.1) * 0.9
b = signal.butter(2, [2500, 5000], "bp", fs=SR, output="sos"); voice = voice + 0.12 * signal.sosfilt(b, voice)

def lp(x, f, o=2): return signal.sosfilt(signal.butter(o, f, "lp", fs=SR, output="sos"), x)
def hp(x, f, o=2): return signal.sosfilt(signal.butter(o, f, "hp", fs=SR, output="sos"), x)
def bp(x, lo, hi, o=2): return signal.sosfilt(signal.butter(o, [lo, hi], "bp", fs=SR, output="sos"), x)
def tt(d): return np.arange(int(d * SR)) / SR
def norm(x, p=0.9): return x / (np.abs(x).max() + 1e-9) * p
def swept_noise(d, f0, f1, q=3.0, shape=1.0):
    n = rng.standard_normal(int(d * SR)); out = np.zeros_like(n); blk = 512
    fs_ = np.geomspace(f0, f1, len(n) // blk + 1)
    for i in range(0, len(n) - blk, blk):
        fc = fs_[i // blk]; bw = fc / q
        sos = signal.butter(2, [max(fc - bw / 2, 30), min(fc + bw / 2, SR / 2 - 100)], "bp", fs=SR, output="sos")
        out[i:i + blk] = signal.sosfilt(sos, n[i:i + blk])    # (zi réinitialisé : granulosité voulue, effet "air")
    x = np.linspace(0, 1, len(n)); e = np.sin(np.pi * x ** shape) ** 1.5
    return out * e
# ---------- bruitages
def whoosh_tech():
    d = 1.0; x = swept_noise(d, 250, 7000, 4, 1.6); t = tt(d)
    sweep = np.sin(2 * np.pi * np.cumsum(np.geomspace(200, 2400, len(t))) / SR) * np.sin(np.pi * (t / d) ** 1.6) ** 2 * .35
    return norm(x + sweep, .85)
def whoosh_medium():
    d = .7; return norm(swept_noise(d, 180, 2600, 2.5, 1.4), .8)
def boom():
    d = 1.6; t = tt(d); f = 38 + 95 * np.exp(-t * 9)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.6)
    thud = lp(rng.standard_normal(len(t)), 220) * np.exp(-t * 14) * 1.2
    return norm(np.tanh(1.8 * (sub + thud)), .95)
def kaching(pitch=1.0):
    d = 1.1; t = tt(d); x = np.zeros_like(t)
    click = hp(rng.standard_normal(len(t)), 3000) * np.exp(-t * 90) * .5
    for f, a, dec_ in [(2093, 1, 4.5), (3136, .7, 5), (4186, .5, 6.5), (6272, .25, 9)]:
        x += a * np.sin(2 * np.pi * f * pitch * t) * np.exp(-t * dec_)
    t2 = np.clip(t - .11, 0, None)                                            # 2e "ding" = la caisse
    for f, a, dec_ in [(2637, 1, 4), (3951, .6, 5), (5274, .35, 7)]:
        x += (t > .11) * a * np.sin(2 * np.pi * f * pitch * t2) * np.exp(-t2 * dec_)
    return norm(x + click, .8)
def glitch():
    d = .5; n = int(d * SR); x = np.zeros(n)
    pos = 0
    while pos < n:
        L = int(rng.uniform(.012, .05) * SR); f = rng.choice([220, 440, 880, 1760, 3520])
        seg = np.sign(np.sin(2 * np.pi * f * np.arange(L) / SR)) * .6 + rng.standard_normal(L) * .25
        if rng.random() < .35: seg *= 0
        x[pos:pos + L] += seg[:max(0, min(L, n - pos))]; pos += L + int(rng.uniform(0, .02) * SR)
    q = 24; x = np.round(x * q) / q                                            # bit-crush
    x = x[::3].repeat(3)[:n]                                                    # sample-hold
    return norm(hp(x, 200) * np.exp(-np.arange(n) / SR * 3), .75)
def swoosh_light():
    return norm(swept_noise(.45, 1500, 6500, 2.2, 1.3), .6)
SFX = dict(whoosh_tech=whoosh_tech(), whoosh_medium=whoosh_medium(), boom=boom(), kaching=kaching(), glitch=glitch(), swoosh_light=swoosh_light())
# ---------- funk 70s (mi mineur, 100 bpm) synthétisé
def funk(dur, bpm=100):
    step = 60 / bpm / 4; N = int(dur * SR); m = np.zeros(N)
    def put(x, t, g=1.0):
        i = int(t * SR)
        if i < N: j = min(N, i + len(x)); m[i:j] += x[:j - i] * g
    def kick():
        t = tt(.22); return np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-t * 28)) / SR) * np.exp(-t * 13)
    def snare():
        t = tt(.2); return (bp(rng.standard_normal(len(t)), 1400, 5200) * .8 + np.sin(2 * np.pi * 190 * t) * .5) * np.exp(-t * 20)
    def hat(o=False):
        t = tt(.18 if o else .045); return hp(rng.standard_normal(len(t)), 7000) * np.exp(-t * (14 if o else 75)) * .5
    def bass(f, d=.2):
        t = tt(d); x = signal.sawtooth(2 * np.pi * f * t) * .6 + np.sin(2 * np.pi * f * t) * .8
        return lp(x, 380) * np.exp(-t * 7) * np.minimum(1, t * 300)
    def chop(freqs, d=.11, wah=0.0):
        t = tt(d); x = sum(signal.sawtooth(2 * np.pi * f * t + i) for i, f in enumerate(freqs)) / len(freqs)
        y = bp(x, 700 + 1500 * wah, 1800 + 2500 * wah, 2) * np.exp(-t * 22) * np.minimum(1, t * 800)
        return y
    E2, G2, A2, B2, D3, E3 = 82.41, 98.0, 110.0, 123.47, 146.83, 164.81
    bass_pat = [E2, 0, E2, 0, G2, E2, 0, D3, 0, E2, 0, E2, A2, 0, G2, B2,
                E2, 0, E2, 0, G2, E2, 0, D3, 0, B2, A2, 0, G2, 0, E2, 0]
    kick_pat = [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0, 1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,0]
    chop_pat = [0,0,1,0, 0,1,0,1, 0,0,1,0, 0,1,0,0, 0,0,1,0, 0,1,0,1, 0,0,1,0, 1,0,1,0]
    nsteps = int(dur / step) + 1
    for k in range(nsteps):
        t = k * step + (step * .18 if k % 2 else 0); s = k % 32; bar = (k // 16) % 2
        if kick_pat[s]: put(kick(), t, .9)
        if s % 8 == 4: put(snare(), t, .8)
        put(hat(), t, .55 if k % 2 == 0 else .3)
        if s % 4 == 2: put(hat(True), t, .3)
        if bass_pat[s]: put(bass(bass_pat[s]), t, .75)
        if chop_pat[s]:
            ch = [E3 * 2, G2 * 4, B2 * 4, D3 * 4] if bar == 0 else [A2 * 4, E3 * 2, G2 * 4, B2 * 4]
            put(chop(ch, wah=.5 + .5 * np.sin(k * .6)), t, .5)
    return m
music = funk(TOTAL + 1)[:NS]
music = music - np.convolve(music, np.ones(2000) / 2000, "same") * 0                            # (placeholder neutre)
music = norm(music, .9)
# ---------- mixage : ducking de la musique sous la voix
venv = np.sqrt(np.convolve(voice ** 2, np.ones(int(.06 * SR)) / int(.06 * SR), "same"))
duck = 1 - 0.55 * np.clip(venv / 0.08, 0, 1); duck = lp(duck, 6, 1)
fade = np.minimum(1, np.arange(NS) / (.25 * SR)) * np.minimum(1, (NS - np.arange(NS)) / (.9 * SR))
mix = voice * 1.0 + music * duck * fade * 0.30
def sfx(name, t, g=1.0, pre=0.0):
    x = SFX[name] if not isinstance(name, np.ndarray) else name
    i = int((t - pre) * SR)
    if i < 0: x = x[-i:]; i = 0
    j = min(NS, i + len(x)); mix[i:j] += x[:j - i] * g
for e in T["events"]:
    t = e["t"]; k = e["kind"]
    if k == "plan1": sfx("whoosh_tech", t, .55, pre=.62); sfx("boom", t, .75)
    elif k == "ultime": sfx("whoosh_medium", t, .5, pre=.45)
    elif k == "check": sfx(kaching(1.0 + .12 * e["n"]), t, .42)
    elif k == "porcs": sfx("glitch", t, .5)
    elif k == "plan2": sfx("whoosh_tech", t, .6, pre=.62); sfx("boom", t, .85); sfx("glitch", t, .3)
    elif k == "toi": sfx("swoosh_light", t, .45, pre=.2)
# sortie de plan plein écran : swoosh
for e in T["events"]:
    if e["kind"] in ("plan1", "plan2"): sfx("swoosh_light", e["t"] + (e.get("dur", 1.3)), .4, pre=.15)
np.save(f"{TMP}/mix_raw.npy", mix)
import wave
def wavout(x, path):
    x = np.clip(x, -1, 1); st = np.stack([x, x], 1); pcm = (st * 32767).astype("<i2")
    with wave.open(path, "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
wavout(mix * 0.7, f"{TMP}/pre.wav")
# loudnorm 2 passes -> -14 LUFS / -1.5 dBTP
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", f"{TMP}/pre.wav", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(r[r.rindex("{"):r.rindex("}") + 1])
af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}"
      f":measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=44100")
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", f"{TMP}/pre.wav", "-af", af, "-ar", "44100", OUT], check=True)
print("audio ok", OUT, "pre-norm I =", j["input_i"])
