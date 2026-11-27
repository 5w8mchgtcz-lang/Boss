"""V9 étape 2 : sous-titres (texte corrigé) calés sur l'énergie vocale + événements sur mots clés.
usage: python3 -I timing.py <words.json whisper> <out timing.json>"""
import sys, json, re, numpy as np
FPS = 30
SEGS = [(1.63, 206), (23.10, 204), (35.70, 341)]            # (début source, nb images)
OUT0 = [0, 206 / FPS, 410 / FPS]                           # début de chaque plan sur la timeline finale
W = json.load(open(sys.argv[1]))
W = [(a, b, re.sub(r"[^\wÀ-ÿ']", "", w)) for a, b, w in W]; W = [w for w in W if w[2]]
# groupes: (début voix réel, fin voix réelle) mesurés sur l'énergie audio, fenêtre whisper, texte corrigé, découpe sous-titres
G = [
 (1.88, 6.35, (1.4, 6.1), "SI TU ES UN HOMME ET QUE TU VEUX AUGMENTER NATURELLEMENT TA TESTOSTÉRONE JE VAIS TE DONNER LA TECHNIQUE",
   ["SI TU ES UN HOMME","ET QUE TU VEUX","AUGMENTER","NATURELLEMENT","TA TESTOSTÉRONE","JE VAIS TE DONNER","LA TECHNIQUE"]),
 (6.63, 8.33, (6.2, 8.4), "ULTIME DONT PERSONNE NE PARLE", ["ULTIME","DONT PERSONNE NE PARLE"]),
 (23.25, 29.71, (22.9, 29.8), "LA PLUPART DES TRUCS QUE TU VAS VOIR SUR INTERNET EN GROS C'EST AVOIR UNE BONNE HYGIÈNE DE VIE BON MOI J'AI FAIT DU SPORT BOIRE DE L'EAU PAS FUMER PAS BOIRE",
   ["LA PLUPART DES TRUCS","QUE TU VAS VOIR","SUR INTERNET","EN GROS C'EST","AVOIR UNE BONNE","HYGIÈNE DE VIE","BON, MOI J'AI FAIT","DU SPORT","BOIRE DE L'EAU","PAS FUMER","PAS BOIRE"]),
 (35.85, 37.65, (35.5, 37.8), "OK C'EST BIEN MAIS POURQUOI IL Y A DES GROS PORCS",
   ["OK C'EST BIEN MAIS","POURQUOI IL Y A","DES GROS PORCS"]),
 (38.31, 40.78, (37.8, 40.7), "QUI MANGENT QUE DE LA MERDE QUI SE TOUCHENT TOUTE LA JOURNÉE",
   ["QUI MANGENT","QUE DE LA MERDE","QUI SE TOUCHENT","TOUTE LA JOURNÉE"]),
 (41.18, 41.94, (40.8, 41.7), "QUI BOIVENT", ["QUI BOIVENT"]),
 (42.13, 46.66, (41.7, 46.9), "QUI FUMENT QUI S'EN BATTENT LES COUILLES ET QUI ONT QUAND MÊME PLUS DE TESTOSTÉRONE QUE TOI JE VAIS T'EXPLIQUER",
   ["QUI FUMENT","QUI S'EN BATTENT","LES COUILLES","ET QUI ONT","QUAND MÊME","PLUS DE TESTOSTÉRONE","QUE TOI","JE VAIS T'EXPLIQUER"]),
]
def src2out(t):
    for k, (s, n) in enumerate(SEGS):
        if s - 0.05 <= t <= s + n / FPS + 0.05: return OUT0[k] + (t - s)
    raise ValueError(t)
toks = []   # (mot, début out, fin out)
for on, off, (a, b), text, chunks in G:
    ws = [w for w in W if a <= w[0] < b]
    t0, t1 = ws[0][0], ws[-1][1]
    aff = lambda t: on + (t - t0) * (off - on) / (t1 - t0)       # recale whisper sur début/fin de voix réels
    lens = np.array([len(w[2]) for w in ws], float); cum = np.concatenate([[0], np.cumsum(lens)]) / lens.sum()
    tm = np.array([aff(w[0]) for w in ws] + [off])
    C = text.split(); cl = np.array([len(c) for c in C], float); cc = np.concatenate([[0], np.cumsum(cl)]) / cl.sum()
    stw = np.interp(cc[:-1], cum, np.append(tm[:-1], off))          # timing relatif whisper (recalé)
    stu = on + cc[:-1] * (off - on)                                   # répartition uniforme au nb de lettres
    wg = 0.7 if off - on > 3 else 0.45                                # whisper tasse les petits mots: on le tempère
    st = wg * stw + (1 - wg) * stu
    en = np.append(st[1:] - 0.02, off)
    for c, s_, e_ in zip(C, st, en): toks.append([c, s_, e_, on, off])
# découpe en sous-titres
chunks_out, i = [], 0
for on, off, _, text, chunks in G:
    for ch in chunks:
        n = len(ch.split()); grp = toks[i:i + n]; i += n
        chunks_out.append({"text": ch, "t0": src2out(grp[0][1]), "t1": src2out(grp[-1][2]),
                           "words": [(g[0], src2out(g[1]), src2out(g[2])) for g in grp]})
# fin de chaque sous-titre : tenue jusqu'au suivant si écart < 0.35 s
for a, b in zip(chunks_out, chunks_out[1:]):
    if b["t0"] - a["t1"] < 0.35: a["t1"] = b["t0"] - 0.01
    else: a["t1"] += 0.18
chunks_out[-1]["t1"] += 0.4
def find(chunk_text, word):
    for c in chunks_out:
        if c["text"] == chunk_text:
            for w in c["words"]:
                if w[0].strip(",") == word: return w[1], w[2]
    raise KeyError(word)
ev = []   # événements sur mots clés
def add(kind, ch, wd, **kw): s, e = find(ch, wd); ev.append(dict(kind=kind, t=s, t_end=e, **kw))
add("plan1", "TA TESTOSTÉRONE", "TESTOSTÉRONE")
add("ultime", "ULTIME", "ULTIME")
add("check", "DU SPORT", "SPORT", label="SPORT", n=0)
add("check", "BOIRE DE L'EAU", "L'EAU", label="BOIRE DE L'EAU", n=1)
add("check", "PAS FUMER", "FUMER", label="PAS FUMER", n=2)
add("check", "PAS BOIRE", "BOIRE", label="PAS BOIRE", n=3)
add("porcs", "DES GROS PORCS", "PORCS")
add("plan2", "PLUS DE TESTOSTÉRONE", "PLUS", dur=1.55)
add("toi", "QUE TOI", "TOI")
total = sum(n for _, n in SEGS) / FPS
json.dump({"fps": FPS, "total": total, "seg_out0": OUT0, "chunks": chunks_out, "events": ev}, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
for c in chunks_out: print(f"{c['t0']:6.2f}-{c['t1']:6.2f}  {c['text']}")
print("events:", [(e['kind'], round(e['t'], 2)) for e in ev], "total", total)
