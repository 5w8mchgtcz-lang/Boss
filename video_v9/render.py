"""V9 étape 4 : rendu vidéo 1080x1920 30 fps.
Fonds animés (B-roll typographique/graphique) DERRIÈRE la silhouette détourée, 2 plans plein écran
(zoom + glitch + flash), sous-titres MAJUSCULES (mots clés en jaune), cadrage fixe.
usage: python3 -I render.py <workdir> <timing.json> <out_video_noaudio.mp4> [a b]   (a b = plage d'images, test)"""
import sys, os, json, math, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
W, H, FPS = 1080, 1920, 30
WD, TJ, OUT = sys.argv[1:4]
T = json.load(open(TJ)); EV = T["events"]; CH = T["chunks"]
NF = int(round(T["total"] * FPS)); SEGN = [200, 203, 60, 81, 176]; SEGC = np.cumsum([0] + SEGN)
FB = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
YEL = (255, 230, 0)
KEY = {"TESTOSTÉRONE", "ULTIME", "TECHNIQUE", "NATURELLEMENT", "HYGIÈNE", "VIE", "SPORT", "L'EAU", "FUMER", "BOIRE", "PORCS", "MERDE", "PLUS", "TOI"}
# scènes : début (s), base, couleur 1, couleur 2, mot défilant
SC = [(0.0, (22, 5, 8), (255, 60, 40), (255, 160, 30), "TESTOSTÉRONE"),
      (4.99, (18, 12, 0), (255, 200, 40), (255, 110, 0), "ULTIME"),
      (6.8667, (2, 16, 18), (0, 230, 150), (0, 150, 255), "HYGIÈNE DE VIE"),
      (13.6667, (16, 2, 22), (210, 0, 255), (255, 0, 120), "POURQUOI ?"),
      (20.0, (22, 5, 8), (255, 60, 40), (255, 160, 30), "TESTOSTÉRONE")]
def scene_idx(t): return max(i for i, s in enumerate(SC) if t >= s[0] - 1e-6)
def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def ease_in(x): x = min(max(x, 0), 1); return x ** 3
def font(sz): return ImageFont.truetype(FB, int(sz))
def fit(text, maxw, sz, minsz=40):
    while sz > minsz and font(sz).getlength(text) > maxw: sz -= 2
    return sz
# ---------------- bandes de texte défilantes (kinetic type) par scène
ROWH, NR = 275, 7
def make_strip(word, outline):
    f = font(ROWH * .86); unit = word + "   "; per = int(round(f.getlength(unit))); reps = int(math.ceil((per + W) / per)) + 1
    def draw(stroke):
        im = Image.new("L", (per * reps, ROWH), 0); d = ImageDraw.Draw(im)
        for i in range(reps): d.text((i * per, ROWH * .08), unit, font=f, fill=255, stroke_width=stroke)
        return np.array(im)
    a = draw(0).astype(np.int16)
    m = np.clip(draw(4).astype(np.int16) - a, 0, 255) if outline else a
    return m.astype(np.uint8), per
STRIPS = {}
def strips(si):
    if si not in STRIPS:
        STRIPS[si] = [make_strip(SC[si][4], outline=(r % 2 == 0)) for r in range(NR)]
    return STRIPS[si]
gy, gx = np.mgrid[0:192, 0:108].astype(np.float32); gy /= 192; gx /= 108
def background(t, si):
    _, base, c1, c2, _ = SC[si]; beat = math.exp(-(t % 0.6) * 7)
    g = np.zeros((192, 108, 3), np.float32); g[:] = base
    for k in range(3):
        cx = .5 + .45 * math.sin(t * .8 + k * 2.1); cy = .5 + .45 * math.cos(t * .55 + k * 1.7)
        w = np.exp(-(((gx - cx) * 1.0) ** 2 + ((gy - cy) * .56) ** 2) / .06)
        col = np.array(c1 if k % 2 == 0 else c2, np.float32)
        g += w[..., None] * col * (.55 + .3 * beat)
    img = cv2.resize(np.clip(g, 0, 255), (W, H), interpolation=cv2.INTER_CUBIC)
    acc = np.array(c2 if si % 2 == 0 else c1, np.float32); lay = np.zeros((H, W), np.float32)
    for r, (m, per) in enumerate(strips(si)):
        sp = (140 + 40 * (r % 3)) * (1 if r % 2 == 0 else -1); off = int((sp * t) % per)
        lay[r * ROWH: (r + 1) * ROWH if (r + 1) * ROWH <= H else H] += (m[:, off:off + W].astype(np.float32) / 255)[: H - r * ROWH] * (.55 if r % 2 == 0 else .14)
    img += lay[..., None] * acc * (.85 + .3 * beat)
    return img
def rim_and_composite(bg, pers, m, si, t):
    c = np.array(SC[si][2], np.float32)[::-1]; ms = cv2.resize(m, (270, 480), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(ms, (0, 0), 7), (W, H), interpolation=cv2.INTER_LINEAR)
    sh = cv2.resize(cv2.GaussianBlur(ms, (0, 0), 14), (W, H), interpolation=cv2.INTER_LINEAR)
    sh = np.roll(sh, 24, axis=0)
    bg = bg * (1 - .45 * sh[..., None]) + glow[..., None] * c * (.9 + .5 * math.exp(-(t % .6) * 7))
    return bg * (1 - m[..., None]) + pers * m[..., None]
# ---------------- plans plein écran
def txt_layer(img, text, cx, cy, sz, fill, stroke=0, sfill=(0, 0, 0), maxw=980):
    sz = fit(text, maxw, sz); f = font(sz); im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.text((cx, cy), text, font=f, fill=fill, anchor="mm", stroke_width=stroke, stroke_fill=sfill); return np.array(im).astype(np.float32)
def grid_bg(t, c1, c2, base):
    img = np.zeros((H, W, 3), np.float32); img[:] = base
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img += yy ** 2 * np.array(c1, np.float32) * .35 + (1 - yy) * np.array(c2, np.float32) * .10
    off = int(t * 120) % 90
    for x in range(0, W, 90): cv2.line(img, (x, 0), (x, H), (60, 60, 60), 1)
    for y in range(-90 + off, H, 90): cv2.line(img, (0, y), (W, y), (60, 60, 60), 1)
    return img
def glowline(img, pts, col, th):
    l = np.zeros_like(img); cv2.polylines(l, [pts], False, col, th, cv2.LINE_AA)
    img += cv2.GaussianBlur(l, (0, 0), 14) * 1.6; img += l; return img
def plan1(p):
    img = grid_bg(p * 3, (255, 60, 40), (255, 160, 30), (14, 4, 6))
    x0, x1, y0, y1 = 110, 970, 1180, 560
    g = ease_out(min(1, p / .75)); n = 90; xs = np.linspace(0, 1, n); ys = (np.exp(3.2 * xs) - 1) / (np.exp(3.2) - 1)
    ys = ys + 0.025 * np.sin(xs * 38) * (1 - xs * .5)
    k = max(2, int(n * g)); pts = np.stack([x0 + xs[:k] * (x1 - x0), y0 + ys[:k] * (y1 - y0)], 1).astype(np.int32)
    fill = np.zeros((H, W), np.float32); poly = np.vstack([pts, [[pts[-1][0], y0 + 40], [pts[0][0], y0 + 40]]]).astype(np.int32)
    cv2.fillPoly(fill, [poly], 1.0); fill = cv2.GaussianBlur(fill, (0, 0), 3)
    img += fill[..., None] * np.array([255, 70, 40], np.float32) * .28
    img = glowline(img, pts.reshape(-1, 1, 2), (255, 190, 60), 14)
    ex, ey = pts[-1]; cv2.circle(img, (int(ex), int(ey)), 26, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(ex), int(ey)), 52 + int(10 * math.sin(p * 30)), (255, 210, 80), 5, cv2.LINE_AA)
    img = txt_layer(img, "TESTOSTÉRONE", W // 2, 330, 170, YEL, 10)
    s = .8 + .2 * ease_out(p / .3)
    pts2 = np.array([[W // 2, 470], [W // 2 + 90 * s, 560], [W // 2 + 35 * s, 560], [W // 2 + 35 * s, 640], [W // 2 - 35 * s, 640], [W // 2 - 35 * s, 560], [W // 2 - 90 * s, 560]], np.int32)
    cv2.fillPoly(img, [pts2], (255, 255, 255), cv2.LINE_AA)
    return img
def plan2(p):
    img = grid_bg(p * 3, (255, 200, 40), (255, 60, 40), (12, 8, 4))
    img = txt_layer(img, "+ DE TESTOSTÉRONE", W // 2, 330, 120, YEL, 9)
    base = 1170; g = ease_out(p / .6); over = 1 + .06 * math.sin(min(1, p / .6) * math.pi)
    hA, hB = 230 * g, 640 * g * over
    for (x0, x1, h, ca, cb, lab) in [(170, 450, hA, (110, 120, 150), (60, 70, 95), "TOI"), (630, 910, hB, (255, 235, 40), (255, 60, 30), "EUX")]:
        bar = np.zeros((H, W, 3), np.float32); hh = int(h)
        if hh > 2:
            for y in range(0, hh, 2):
                a = y / max(hh, 1); bar[base - y - 2: base - y, x0:x1] = np.array(ca) * (1 - a) + np.array(cb) * a if False else np.array(cb) * (1 - a) + np.array(ca) * a
            m = np.zeros((H, W), np.float32); m[base - hh: base, x0:x1] = 1
            img = np.clip(img * (1 - m[..., None]) + bar * m[..., None] + cv2.GaussianBlur(m, (0, 0), 20)[..., None] * np.array(ca, np.float32) * (.25 if lab == "EUX" else 0), 0, 255)
        img = txt_layer(img, lab, (x0 + x1) // 2, base + 70, 84, (255, 255, 255), 6)
    cv2.line(img, (110, base), (970, base), (255, 255, 255), 6, cv2.LINE_AA)
    return img
# ---------------- carte "check" (sport / eau / fumer / boire)
def check_card(img, label, a):
    sz = fit(label, 760, 128); f = font(sz); tw = f.getlength(label); cw = int(tw + 230); cx, cy = W // 2, 240
    s = 1 + .22 * (1 - ease_out(a / .22)) if a < .22 else 1.0; alpha = min(1, a / .08)
    lay = Image.new("RGBA", (W, 460), (0, 0, 0, 0)); d = ImageDraw.Draw(lay); mid = 230
    d.rounded_rectangle([W // 2 - cw // 2, mid - 100, W // 2 + cw // 2, mid + 100], 46, fill=(8, 12, 14, 215), outline=(0, 230, 150, 255), width=7)
    ccx = W // 2 - cw // 2 + 100; d.ellipse([ccx - 62, mid - 62, ccx + 62, mid + 62], fill=(0, 220, 140, 255))
    d.line([(ccx - 30, mid + 2), (ccx - 8, mid + 26), (ccx + 34, mid - 24)], fill=(255, 255, 255, 255), width=17, joint="curve")
    d.text((ccx + 100 + tw / 2, mid + 4), label, font=f, fill=(255, 255, 255, 255), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0, 255))
    arr = np.array(lay); 
    if s != 1.0: arr = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    h, w = arr.shape[:2]; x = W // 2 - w // 2; y = 150 - (h - 460) // 2 - 0; y0, y1 = max(0, y), min(H, y + h)
    sub = arr[y0 - y: y1 - y, max(0, -x): w - max(0, x + w - W)] if x >= 0 else arr[y0 - y: y1 - y, -x: -x + W]
    x0 = max(x, 0); al = (sub[..., 3:4].astype(np.float32) / 255) * alpha
    img[y0:y1, x0:x0 + sub.shape[1]] = img[y0:y1, x0:x0 + sub.shape[1]] * (1 - al) + sub[..., 2::-1] * al
    return img
# ---------------- sous-titres
CAP = {}
def cap_img(ci):
    if ci in CAP: return CAP[ci]
    ch = CH[ci]; words = [w[0] for w in ch["words"]]; txt = " ".join(words)
    sz = fit(txt, 960, 100, 56); f = font(sz); sp = f.getlength(" "); stroke = max(8, sz // 8)
    ws = [f.getlength(w) for w in words]; tw = int(sum(ws) + sp * (len(words) - 1)); pad = stroke * 3
    im = Image.new("RGBA", (tw + 2 * pad, int(sz * 1.5)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = pad
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); ds = ImageDraw.Draw(sh)
    for w, wl in zip(words, ws):
        k = w.strip(",.?!") in KEY; col = YEL if k else (255, 255, 255)
        ds.text((x, sz * .2 + 9), w, font=f, fill=(0, 0, 0, 200), stroke_width=stroke)
        d.text((x, sz * .2), w, font=f, fill=col + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255)); x += wl + sp
    from PIL import ImageFilter
    sh = sh.filter(ImageFilter.GaussianBlur(6)); sh.alpha_composite(im); CAP[ci] = np.array(sh); return CAP[ci]
def draw_caption(img, t):
    for i, c in enumerate(CH):
        if c["t0"] <= t < c["t1"]:
            a = cap_img(i); age = t - c["t0"]
            s = 1.0 if age > .17 else (.82 + .34 * ease_out(age / .1) if age < .1 else 1.16 - .16 * ((age - .1) / .07))
            if abs(s - 1) > .005: a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            h, w = a.shape[:2]; x = W // 2 - w // 2; y = 1390 - h // 2
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)          # découpe aux bords de l'écran
            a = a[y0 - y: y1 - y, x0 - x: x1 - x]
            al = a[..., 3:4].astype(np.float32) / 255; img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al) + a[..., 2::-1] * al
            break
    return img
# ---------------- effets
def zoom(img, s):
    if abs(s - 1) < 1e-3: return img
    h, w = int(H / s), int(W / s); y0, x0 = (H - h) // 2, (W - w) // 2
    return cv2.resize(img[y0:y0 + h, x0:x0 + w], (W, H), interpolation=cv2.INTER_LINEAR)
def zoomblur(img, s, amt=.07): return (zoom(img, s) + zoom(img, s * (1 + amt)) + zoom(img, s * (1 - amt * .6))) / 3
def rgb_split(img, px):
    px = int(px)
    if px <= 0: return img
    o = img.copy(); o[..., 0] = np.roll(img[..., 0], px, 1); o[..., 2] = np.roll(img[..., 2], -px, 1); return o   # BGR
def slices(img, rng, n, mx):
    o = img.copy()
    for _ in range(n):
        y = rng.integers(0, H - 80); h = rng.integers(20, 140); o[y:y + h] = np.roll(img[y:y + h], int(rng.integers(-mx, mx)), 1)
    return o
def flash(img, a): return img * (1 - a) + 255 * a if a > 0 else img
# ---------------- une image
def pers_frame(fi):
    k = int(np.searchsorted(SEGC, fi, side="right") - 1); j = fi - SEGC[k] + 1
    f = cv2.imread(f"{WD}/frames/s{k}_{j:04d}.jpg"); f = cv2.resize(f, (W, 1923), interpolation=cv2.INTER_CUBIC)[:H].astype(np.float32)
    m = MM[fi].astype(np.float32) / 255; m = cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)
    m = np.clip((m - .1) / .78, 0, 1); m = cv2.GaussianBlur(m, (0, 0), 1.3)
    return f, m
CLIPDIR = os.environ.get("CLIPS_DIR", f"{WD}/clips"); CAPS = {}
def clip_frame(name, tl):
    """image BGR (1080x1920, recadrée) du plan Artlist <name>.mp4 à tl secondes (boucle) ; None si le fichier n'existe pas"""
    path = f"{CLIPDIR}/{name}.mp4"
    if not os.path.exists(path): return None
    if name not in CAPS: CAPS[name] = cv2.VideoCapture(path)
    c = CAPS[name]; n = max(1, int(c.get(cv2.CAP_PROP_FRAME_COUNT))); fps = c.get(cv2.CAP_PROP_FPS) or 30
    c.set(cv2.CAP_PROP_POS_FRAMES, int(tl * fps) % n); ok, f = c.read()
    if not ok: return None
    h, w = f.shape[:2]; k = max(W / w, H / h); f = cv2.resize(f, (int(w * k + .5), int(h * k + .5)), interpolation=cv2.INTER_CUBIC)
    y0, x0 = (f.shape[0] - H) // 2, (f.shape[1] - W) // 2
    return f[y0:y0 + H, x0:x0 + W].astype(np.float32)
CLIPWIN = []   # (nom du plan, début, fin) : le plan passe DERRIÈRE la silhouette
def build_windows():
    chk = [e for e in EV if e["kind"] == "check"]
    for i, e in enumerate(chk):
        CLIPWIN.append((["sport", "eau", "fumer", "boire"][i], e["t"], e["t"] + (chk[i + 1]["t"] - e["t"] if i + 1 < len(chk) else 1.0)))
    for c in CH:
        if c["text"] == "QUE DE LA MERDE": CLIPWIN.append(("fastfood", c["t0"], c["t1"] + .3))
build_windows()
def scene_frame(fi, t):
    f, m = pers_frame(fi)
    f = np.clip((f - 128) * 1.05 + 128 + 3, 0, 255)                            # léger punch, fond d'origine conservé
    out = f
    for name, a, b in CLIPWIN:
        if a <= t < b:
            bg = clip_frame(name, t - a)
            if bg is not None:
                u = min(1, (t - a) / .12); bg = bg * (.55 + .45 * u)               # entrée rapide du plan derrière vous
                sh = cv2.resize(cv2.GaussianBlur(cv2.resize(m, (270, 480), interpolation=cv2.INTER_AREA), (0, 0), 10), (W, H))
                bg = bg * (1 - .35 * np.roll(sh, 18, axis=0)[..., None])
                out = bg * (1 - m[..., None]) + f * m[..., None]
            break
    chk = [e for e in EV if e["kind"] == "check" and e["t"] <= t < e["t"] + 1.05]  # cartes ✓ (illustrations)
    if chk: out = check_card(out, chk[-1]["label"], t - chk[-1]["t"])
    return out
def render(fi):
    t = fi / FPS; rng = np.random.default_rng(1000 + fi); fl = 0.0
    plans = [(e, e["t"], e.get("dur", 1.3)) for e in EV if e["kind"] in ("plan1", "plan2")]
    cur = None
    for e, t0, D in plans:
        if t0 - .13 <= t < t0: cur = ("pre", e, t0, D)
        elif t0 <= t < t0 + D: cur = ("card", e, t0, D)
        elif t0 + D <= t < t0 + D + .28: cur = ("post", e, t0, D)
    if cur and cur[0] == "card":
        _, e, t0, D = cur; p = (t - t0) / D; cf = clip_frame("gym" if e["kind"] == "plan1" else "bodybuilder", t - t0)
        if cf is not None:
            card = txt_layer(cf[..., ::-1].copy(), "TESTOSTÉRONE" if e["kind"] == "plan1" else "+ DE TESTOSTÉRONE", W // 2, 330, 150, YEL, 10)
        else:
            card = plan1(p) if e["kind"] == "plan1" else plan2(p)
        card = card[..., ::-1].copy(); age = t - t0
        img = zoomblur(card, 1 + .4 * (1 - ease_out(age / .25)), .05 * (1 - ease_out(age / .25)))
        if age < .2: img = rgb_split(slices(img, rng, 4, 90), 18 * (1 - age / .2))
        fl = max(0, 1 - age / .14)
    elif cur and cur[0] == "pre":
        _, e, t0, D = cur; u = (t - (t0 - .13)) / .13; img = zoomblur(scene_frame(fi, t), 1 + .55 * ease_in(u), .06 + .05 * u); img = rgb_split(img, 26 * u)
    elif cur and cur[0] == "post":
        _, e, t0, D = cur; age = t - (t0 + D); img = zoomblur(scene_frame(fi, t), 1 + .3 * (1 - ease_out(age / .26)), .05 * (1 - ease_out(age / .26)))
        if age < .12: img = rgb_split(slices(img, rng, 5, 110), 20)
        fl = max(0, .9 * (1 - age / .2)) if age < .2 else 0
        if age < 0.0: fl = 0
    else:
        img = scene_frame(fi, t)
        for e in EV:                                                        # effets ponctuels sur mots clés
            a = t - e["t"]
            if e["kind"] == "porcs" and 0 <= a < .3: img = rgb_split(slices(img, rng, 6, 130), 14 * (1 - a / .3)); fl = max(fl, .35 * (1 - a / .12) if a < .12 else 0)
            if e["kind"] == "ultime" and 0 <= a < .25: fl = max(fl, .55 * (1 - a / .25)); img = zoomblur(img, 1 + .08 * (1 - a / .25), .04)
            if e["kind"] == "toi" and 0 <= a < .3: img = zoomblur(img, 1 + .05 * (1 - a / .3), .03)
            if e["kind"] == "check" and 0 <= a < .2: img = zoom(img, 1 + .018 * (1 - a / .2))
        for b in (SEGC[1], SEGC[2]):                                        # coupes entre plans : micro-glitch
            a = (fi - b) / FPS
            if 0 <= a < .14: img = rgb_split(slices(img, rng, 4, 80), 12 * (1 - a / .14))
    img = draw_caption(img, t)
    img = flash(img, fl)
    img = img + rng.normal(0, 2.0, img.shape[:2])[..., None].astype(np.float32)   # grain léger
    return np.clip(img, 0, 255).astype(np.uint8)
MM = None
def work(args):
    global MM
    a, b, path = args; MM = np.load(f"{WD}/matte.npy", mmap_mode="r")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "13", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", path], stdin=subprocess.PIPE)
    for fi in range(a, b): p.stdin.write(render(fi).tobytes())
    p.stdin.close(); p.wait(); return path
if __name__ == "__main__":
    if len(sys.argv) > 5:                                                    # test: images a..b -> PNG
        a, b = int(sys.argv[4]), int(sys.argv[5]); MM = np.load(f"{WD}/matte.npy", mmap_mode="r")
        for fi in range(a, b, int(sys.argv[6]) if len(sys.argv) > 6 else 1): cv2.imwrite(f"{WD}/test_{fi:04d}.png", render(fi))
    else:
        nw = 4; cuts = np.linspace(0, NF, nw + 1).astype(int); jobs = [(int(cuts[i]), int(cuts[i + 1]), f"{WD}/part{i}.mp4") for i in range(nw)]
        with Pool(nw) as pool: parts = pool.map(work, jobs)
        open(f"{WD}/parts.txt", "w").write("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{WD}/parts.txt", "-c", "copy", OUT], check=True); print("video ok", OUT)
