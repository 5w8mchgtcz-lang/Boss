"""V9 étape 1 : extraction des images des 3 plans retenus + détourage MODNet (silhouette).
usage: python3 -I prep.py <source.mp4> <workdir>"""
import sys, os, subprocess, numpy as np, cv2, onnxruntime as ort
from huggingface_hub import hf_hub_download
SRC, WD = sys.argv[1], sys.argv[2]
FPS = 30
# (début source en s, nb d'images) — coupes calées sur les silences réels
SEGS = [(1.63, 206), (23.10, 204), (35.70, 341)]
os.makedirs(f"{WD}/frames", exist_ok=True)
for k, (s, n) in enumerate(SEGS):
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(s),"-i",SRC,"-frames:v",str(n),"-q:v","2",
                    f"{WD}/frames/s{k}_%04d.jpg"], check=True)
N = sum(n for _, n in SEGS)
MW, MH = 512, 896
mm = np.lib.format.open_memmap(f"{WD}/matte.npy", mode="w+", dtype=np.uint8, shape=(N, MH, MW))
p = hf_hub_download("Xenova/modnet", "onnx/model.onnx")
so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count()
sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
i = 0
for k, (s, n) in enumerate(SEGS):
    for j in range(1, n + 1):
        f = cv2.imread(f"{WD}/frames/s{k}_{j:04d}.jpg")
        x = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (MW, MH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - .5) / .5).transpose(2, 0, 1)[None]
        m = sess.run(None, {inp: x})[0][0, 0]
        mm[i] = np.clip(m * 255, 0, 255).astype(np.uint8); i += 1
        if i % 50 == 0: print(i, N, flush=True)
mm.flush(); print("DONE", flush=True)
