import json, os, subprocess, sys, time, math, hashlib

PACK = "/root/pack"; OUT = "/root/out"
# scorer uses the VALIDATOR ffmpeg/ffprobe (inside rootfs, /usr/local/bin). We call the scorer here.
FF = "/usr/local/bin/ffmpeg"; FP = "/usr/local/bin/ffprobe"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, "/app")
from vidaio.core import section
from vidaio.core.config import load_raw_config
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring_worker.config import ScoringWorkerConfig
from vidaio.scoring_worker.service import ScoreRejected, _score_sync, real_backends
from vidaio.services.protocol import ScoreRequest

LINES = []
def R(s):
    LINES.append(s); print("R> " + s, flush=True)
    try: open("/root/RESULT.txt", "w").write("\n".join(LINES))
    except Exception: pass

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

raw = load_raw_config()
cfg = section(raw, "scoring_worker", ScoringWorkerConfig)
sc = section(raw, "scoring", ScoringConfig)
backends = real_backends(cfg, scoring_config=sc)

def score(clip, enc):
    rd, od = sha(clip), sha(enc)
    params = {"vmaf_threshold": 90, "scoring": "streaming", "bits_per_pixel": 0.05,
              "max_keyframe_interval_s": 2.0, "peak_ratio": 1.5, "peak_window_s": 2.0}
    req = ScoreRequest(track="compression", challenge_id="p", item_id=os.path.basename(enc),
                       miner_hotkey="p", reference_path=clip, reference_digest=rd,
                       miner_input_path=clip, miner_input_digest=rd, output_path=enc,
                       output_digest=od, params=params, scorer_version="vidaio-scorer/1")
    try:
        it = _score_sync(req, cfg, sc, backends, req.scorer_version)
    except ScoreRejected as e:
        return {"rejected": str(e)[:70]}
    d = it.model_dump(); m = d.get("metrics") or {}
    return {"score": d["score"], "vmaf": m.get("vmaf"),
            "viol": [v.get("code") if isinstance(v, dict) else str(v) for v in d.get("violations") or []]}

# friend's entry was cloned to /root/friend (encode.py + run.sh). Its Dockerfile uses the mwader
# static ffmpeg (SVT-AV1 4.3) placed at /opt/ff/ffmpeg. Run encode.py with THAT ffmpeg on PATH.
FRIEND = "/root/friend"
FFDIR = "/opt/ff"   # mwader ffmpeg/ffprobe here (set up by onstart)

# Build an input dir of all 8 clips (sha256-named, no ext) as the real evaluator provides.
INP = "/root/inp"; os.makedirs(INP, exist_ok=True)
idx = json.load(open("%s/index.json" % PACK))
name_by_file = {}
for c in idx:
    src = "%s/%s/clip.mkv" % (PACK, c["name"])
    dig = sha(src)
    dst = "%s/%s" % (INP, dig)     # evaluator names items by sha256, no extension
    if not os.path.exists(dst):
        import shutil as _sh; _sh.copy(src, dst)
    name_by_file[dig] = (c["name"], src)

# Run friend's run.sh over the whole batch, timing it (this is the run.sh time → speed term).
fenv = dict(os.environ)
fenv["PATH"] = FFDIR + ":/usr/local/bin:/usr/bin:/bin"   # friend's encode.py picks ffmpeg via PATH
fenv["CONTENDER_APP_DIR"] = FRIEND
R("running friend run.sh over 8 clips (ffmpeg=%s)" % FFDIR)
t0 = time.time()
rc = subprocess.run(["/bin/sh", "%s/run.sh" % FRIEND, INP, OUT],
                    env=fenv, capture_output=True, text=True, timeout=5400)
batch_secs = time.time() - t0
R("run.sh exit=%d wall=%.0fs  stderr_tail=%s" % (rc.returncode, batch_secs, (rc.stderr or '')[-200:]))

# Score each output with the validator scorer.
total = 0.0; n = 0; tot_frames = 0
for dig, (nm, src) in sorted(name_by_file.items(), key=lambda kv: kv[1][0]):
    outp = "%s/%s" % (OUT, dig)
    # count frames of source for speed calc
    try:
        fr = int(subprocess.check_output([FP, "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", src], text=True).strip() or "0")
    except Exception:
        fr = 0
    tot_frames += fr
    if not os.path.exists(outp) or os.path.getsize(outp) == 0:
        R("%s NO_OUTPUT" % nm); n += 1; continue
    r = score(src, outp); scv = r.get("score"); vm = r.get("vmaf")
    R("%s score=%s vmaf=%s viol=%s" % (nm, ("%.4f" % scv) if isinstance(scv, (int, float)) else scv,
      ("%.2f" % vm) if vm else "NA", r.get("viol")))
    total += (scv if isinstance(scv, (int, float)) else 0.0); n += 1

mean = total / n if n else 0.0
spf = batch_secs / tot_frames if tot_frames else 9.9
fast, slow = 1/30, 2.0
spd = 1.0 if spf <= fast else 0.0 if spf >= slow else math.log(slow/spf)/math.log(slow/fast)
R("FRIEND QUALITY_MEAN=%.4f  batch_spf=%.4f  speed=%.3f  FINAL_EST=%.4f  (README claims 0.7789)"
  % (mean, spf, spd, mean * (0.7 + 0.3 * spd)))

print("\n===ALLRESULTS_BEGIN===", flush=True)
for ln in LINES: print("R> " + ln, flush=True)
print("===ALLRESULTS_END===", flush=True)
print("FRIEND_DONE", flush=True)
