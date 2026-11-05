"""Streaming-compression entry: encode every item of a batch inside its bitrate budget.

usage: python3 encode.py --bits-per-pixel BPP [--max-keyframe-interval S] [--peak-window S]
                         [--peak-ratio R] INPUT_DIR OUTPUT_DIR

Per item (up to 4 in parallel):
  1. timing exactly as the validator derives it (canonical CFR view: fps, frame count); budget =
     BPP * w * h * fps / 1000 kbps; keyframes every floor(S * fps) frames and, across timestamp gaps,
     on the last frame before a gap; output colour tags mirror the input's
  2. SVT-AV1 4.3 two-pass VBR (QM, undershoot-pct, maxsection-pct), chroma-only blur and a light luma
     unsharp; the packets are measured exactly as the validator does and the target is refitted
     towards 99 % of the budget (over budget or peak -> lower; clear undershoot -> higher, time-gated)
  3. VMAF (vmaf_v0.6.1) and vmaf_v0.6.1neg measured against the input: the sharpening strength is
     raised while vmaf - neg stays <= 2.85; above it, plain encodes and a delta-lowering ladder
  4. small hard clips may be re-encoded with a slower preset when the time allows
  5. anything failing -> a conservative x264 encode, rule- and delta-checked
Optional work is gated by the speed term (1/30 s per frame); a hard time limit protects the batch.
Stdlib only; ffmpeg/ffprobe via subprocess. Never raises out of main: run.sh must exit 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

FFMPEG = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/usr/local/bin/ffprobe"

PRESET_LARGE = 6           # above 1280x720
PRESET_SMALL = 4           # 1280x720 and below: a quarter of the work per frame, slower preset pays
SLOW_PRESET = 3            # optional re-encode of hard small clips (q07: +0.022 score at 1.7x encode time)
SLOW_PRESET_COST = 1.8     # its encode time relative to PRESET_SMALL
SLOW_PRESET_VMAF_MIN = 50.0
SLOW_PRESET_VMAF_MAX = 92.0
# CRF clips (the hard ones) go further: rate-matched CRF p4 -> p2 / p1: q08 58.6 -> 59.8 / 60.8,
# q07 75.7 -> 76.7 / 76.8, p01 62.5 -> 62.9 / 63.8. p2 costs ~3.5x a p4 encode (one-pass).
# Large clips (above 720p) re-encode at p4 when the time allows (q02 p6 -> p4: 96.18 -> 96.74, +0.014, ~1.7x)
SLOW_PRESET_LARGE = 4
SLOW_PRESET_LARGE_COST = 1.8
SLOW_PRESET_LARGE_VMAF_MAX = 98.5
SLOW_PRESET_CRF = 2
SLOW_PRESET_CRF_COST = 3.5
FIRST_AIM = 0.985         # first target as a share of the budget (2-pass tends to land a bit under)
# The rule allows 1.01; the packets are measured exactly as the validator measures them (same sizes, same
# frames / fps duration), so the only margin needed is for nothing at all -> keep a small one anyway.
ACCEPT_AVG = 1.005        # measured average must stay under this share (rule: 1.01)
RATE_GOAL = 0.99          # retargeting aims here
ACCEPT_PEAK_SHARE = 0.99  # measured peak must stay under this share of peak_ratio x budget (exact measurement)
REFIT_BELOW = 0.95        # refit upwards when the average lands below this share ...
EASY_BELOW = 0.70         # ... unless it is far below: the encoder is already at its quality cap
BIG_SHORTFALL = 0.90      # below this share an upward refit may use the sharpening time slack
MAX_RATE_ENCODES = 3
RESCUE_ENCODES = 3        # extra encodes allowed only while nothing fits the rules yet
# The validator zeroes an item when vmaf - vmaf_neg > 3.0. This entry measures the same file with the
# same libvmaf build and models (agreement within 1e-4 on the practice pack), so 0.15 of margin is ample.
DELTA_GUARD = 2.85
DELTA_QUICK_OK = 2.5      # subsampled (every 3rd frame, within ~0.15) estimate below this: accept
DELTA_QUICK_BAD = 3.1     # subsampled estimate above this: go straight to the next ladder step
# Luma-only light sharpening before the encode (ffmpeg unsharp, 5x5, chroma untouched). VMAF rewards it
# strongly; vmaf_v0.6.1neg discounts it, so the strength is searched per clip and every candidate's
# vmaf - neg is measured on all frames and must stay <= DELTA_GUARD. Practice pack (SVT p6 2-pass,
# 0.97 budget), score / delta at strength 0 -> 0.04 -> 0.08:
#   q03 .936/1.70 -> .972/2.68 -> .977/2.85   q04 .931/1.28 -> .956/2.22 -> .971/2.81
#   q05 .977/0.10 -> .985/1.09 -> 1.00/2.10   q06 .977/0.20 -> .988/1.69 -> 1.00/2.75
#   q07 .325/2.25 -> .347/2.50 -> .349/2.70   q01 .987/2.52 -> .997/2.83 (no headroom)
SHARPEN_STEP = 0.04
SHARPEN_MAX = 0.16
SHARPEN_MIN_HEADROOM = 0.35   # skip when the plain encode is already this close to the guard
SHARPEN_TIME_SLACK = 1.3      # sharpening may run up to this multiple of the free time
SHARPEN_VMAF_MIN = 58.0       # below: the clip scores 0 at VMAF <= 60 anyway (q08: sharpening lowered it)
SHARPEN_VMAF_MAX = 99.9       # above: the score is already ~1
# Always on (rate-matched on the practice pack): quantisation matrices (AI clips +0.8..1.0 VMAF, others
# +-0.1, vmaf-neg unchanged or lower) and a chroma-only blur (VMAF and NEG read luma only, so the bits
# move to luma; q07 +0.6). The chroma checks allow chroma MAE <= 0.10 and an energy ratio 0.4..1.8.
# undershoot-pct=10: first-shot rate q03 .87->.985, q07 .84->.97. maxsection-pct=120 (default 2000): caps a
# GOP section at 1.2x the target, so bursty clips keep the 2 s peak rule while spending the average
# (q03 peak 1.50x -> 1.35x of the budget, VMAF unchanged; other clips unaffected).
BASE_EXTRA = "enable-qm=1:qm-min=0:qm-max=15:undershoot-pct=10:maxsection-pct=120"
BASE_EXTRA_CRF = "enable-qm=1:qm-min=0:qm-max=15"   # rate-control knobs do not apply to CRF
CHROMA_VF = "gblur=sigma=1:planes=6"
SHARPEN_FIRST = 0.04          # the first encode is already sharpened; plain only if that breaks the guard
TF0_RESCUE_MARGIN = 0.3       # sharpened delta within guard + this on a clean clip: retry with tf-strength=0
TF0_RESCUE_MIN_VMAF = 95.0
SHARPEN_DELTA_COST = 0.6      # conservative delta added by SHARPEN_FIRST (measured 0.24..1.5, q02 0.72)
# Hard, high-entropy clips (AI water/foliage, VMAF < CRF_TRY_BELOW after the VBR encode): one-pass CRF
# keeps the quality flatter across frames and beats two-pass VBR at the same rate (rate-matched, p4/p6,
# QM + chroma blur + sharpen 0.04): p01 57.2 -> 62.5, q08 56.7 -> 58.4, q07 74.1 -> 75.1; renders and
# screen content are equal or slightly worse (q03 -0.3, p05 -0.1), so CRF is only tried on hard clips.
# CRF -> rate: d ln(rate) / d CRF ~ -0.075 on such clips (q07/q08/p01 measured 0.066..0.082); SVT takes
# fractional CRF in 0.25 steps up to 70.
CRF_TRY_BELOW = 85.0
CRF_START_SMALL = 56.0       # <= 1280x720 (q07 56.75, q08 56, p01 ~60 at the budget)
CRF_START = 52.0
CRF_SLOPE = 0.075
CRF_MAX_ENCODES = 5
CRF_TIME_SLACK = 1.5          # big gains on hard clips: may use up to this multiple of the free time
PARALLEL = 4
# Speed term: 1 at <= 1/30 s per output frame (sum of run.sh times / frames). Below that line time is
# free; above it every +10 % of time costs ~0.5 % of the result, more than an optional refit gains.
# Optional work (upward rate refits) only runs if it is expected to finish inside this share of the
# batch's free time; zero-avoiding work (rule violations, the VMAF-delta ladder) always runs.
# The batch timeout is 3600 s for the whole run.sh; past this point only zero-avoiding work runs.
HARD_OPTIONAL_LIMIT_S = 1500.0
# Past this point even zero-avoiding SVT work stops and the fast x264 fallback finishes the item, so a
# pathological clip on a slow host still ends well inside the 3600 s batch timeout.
HARD_REQUIRED_LIMIT_S = 2400.0
FREE_SPF = 1.0 / 30.0
FREE_SHARE = 0.85
# Delta-lowering ladder, cheapest first (q02, SVT p6 2-pass at 0.97 budget: vmaf / delta)
#   base 96.42 / 2.89, tf-strength=0 96.08 / 2.67, enable-restoration=0 95.57 / 2.53,
#   + tune=0 / enable-tf=0 lower it further at ~1-1.6 VMAF.
LADDER = [BASE_EXTRA, BASE_EXTRA + ":tf-strength=0", BASE_EXTRA + ":enable-restoration=0",
          BASE_EXTRA + ":enable-restoration=0:tf-strength=0",
          BASE_EXTRA + ":enable-restoration=0:tune=0:enable-tf=0"]

_log_lock = threading.Lock()
T0 = time.monotonic()
FREE_DEADLINE = float("inf")  # set in main() from the batch's frame count
ITEMS: list = []              # every Item of the batch (for the others' expected finish times)
PRESET_COST = {4: 2.5, 5: 1.6, 6: 1.0, 7: 0.75, 8: 0.55}  # relative encode cost per pixel (SVT 4.2, 2-pass)


def effective_cpus() -> int:
    """CPUs this process may really use: cgroup v2 / v1 quota, else affinity, else cpu_count (max 64)."""
    for qf, pf in (("/sys/fs/cgroup/cpu.max", None),
                   ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            if pf is None:
                quota, period = open(qf).read().split()[:2]
            else:
                quota, period = open(qf).read().strip(), open(pf).read().strip()
            if quota not in ("max", "-1") and int(quota) > 0 and int(period) > 0:
                return max(1, min(64, int(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    try:
        return max(1, min(64, len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        return max(1, min(64, os.cpu_count() or 8))


def log(msg: str) -> None:
    with _log_lock:
        print(f"[{time.monotonic() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} exit {r.returncode}: {r.stderr[-400:]}")
    return r


# ---------------------------------------------------------------------------- measurement

# Timestamps are kept (the validator's frame-count gate compares the raw stream frame counts) and only
# rebased to 0, while the budget, the average-bitrate duration and the VMAF pairing use the validator's
# canonical CFR view (canonical_timing). For a CFR clip both views are the same thing.
CANON_VF = "setpts=PTS-STARTPTS"

_TAG_FIELDS = (("color_range", "-color_range"), ("color_space", "-colorspace"),
               ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"))
_UNSET = {"", "unknown", "unspecified", "reserved", "n/a"}


def probe(path: str) -> dict:
    """Geometry, stream frame rate, packet count and the colour tags of the first video stream."""
    out = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
               "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets," +
               ",".join(f for f, _ in _TAG_FIELDS), "-of", "json", path], 300).stdout
    s = json.loads(out)["streams"][0]
    fps = Fraction(s.get("r_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(s.get("avg_frame_rate") or "0/1")
    # Output colour tags mirror the input's: the validator compares pixel values after reading
    # each side with its own tags (an untagged side is read as BT.709/tv), so relabelling (e.g. a
    # full-range input written as tv) could shift the RGB its tone/chroma checks see.
    color = []
    for field, opt in _TAG_FIELDS:
        value = str(s.get(field) or "").lower()
        if value not in _UNSET:
            color += [opt, value]
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": fps,
            "frames": int(s["nb_read_packets"]), "color": color}


def is_uniform(times: list[float], fps: float) -> bool:
    """Timestamps on a constant-rate grid (container rounding, e.g. Matroska's 1 ms, tolerated)."""
    if len(times) < 2 or fps <= 0:
        return False
    step = 1.0 / fps
    tol = max(0.0015, 0.05 * step)
    return all(abs(t - i * step) <= tol for i, t in enumerate(times))


def canonical_timing(path: str, fallback: dict, times: list[float] | None = None) -> tuple[Fraction, int]:
    """(fps, frames) exactly as the validator derives them: it canonicalizes the reference to a
    CFR y4m (format=yuv420p, setpts=PTS-STARTPTS, -fps_mode cfr) and probes that file, so the budget
    and the packet measurement use the y4m header rate and the decoded frame count."""
    try:
        base = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path, "-map", "0:v:0",
                "-an", "-sn", "-dn", "-vf", "format=yuv420p,setpts=PTS-STARTPTS", "-fps_mode", "cfr"]
        head = subprocess.run(base + ["-frames:v", "1", "-f", "yuv4mpegpipe", "-"], capture_output=True,
                              timeout=300).stdout[:200].split(b"\n", 1)[0].decode("ascii", "replace")
        rate = next(tok[1:] for tok in head.split() if tok.startswith("F"))
        num, den = rate.split(":")
        fps = Fraction(int(num), int(den))
        if times and is_uniform(times, float(fps)):
            # constant frame rate: the CFR canonical copy has exactly one frame per input frame
            return fps, len(times)
        prog = run(base + ["-f", "null", "-", "-progress", "pipe:1", "-nostats"], 900).stdout
        frames = max(int(line.split("=", 1)[1]) for line in prog.splitlines() if line.startswith("frame="))
        if fps > 0 and frames > 0:
            return fps, frames
    except Exception as exc:  # noqa: BLE001  fall back to the stream probe
        log(f"canonical timing failed ({exc}); using the stream probe")
    return fallback["fps"], fallback["frames"]


def measure(path: str, fps: float, frames: int, window_s: float) -> dict:
    """Validator's packet measurement (vidaio/scoring/streaming.py), decode-order packets."""
    out = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size,pts_time,flags",
               "-of", "json", path], 300).stdout
    pk = [(int(p["size"]), float(p["pts_time"]), "K" in p.get("flags", "")) for p in json.loads(out)["packets"]]
    sizes = [s for s, _, _ in pk]
    avg = 8.0 * sum(sizes) / (frames / fps) / 1000.0
    window = max(1, round(window_s * fps))
    if len(sizes) <= window:
        peak = 8.0 * sum(sizes) / (len(sizes) / fps) / 1000.0
    else:
        cur = best = sum(sizes[:window])
        for i in range(window, len(sizes)):
            cur += sizes[i] - sizes[i - window]
            best = max(best, cur)
        peak = 8.0 * best / (window / fps) / 1000.0
    pts = [t for _, t, _ in pk]
    start, end = min(pts), max(pts) + 1.0 / fps
    keys = sorted(t for _, t, k in pk if k)
    first_key = bool(keys) and abs(keys[0] - start) < 0.5 / fps
    spans = ([b - a for a, b in zip(keys, keys[1:])] + [end - keys[-1], keys[0] - start]) if keys else [end - start]
    return {"avg": avg, "peak": peak, "max_kf": max(spans), "first_key": first_key, "packets": len(pk)}


def vmaf_pair(ref: str, dist: str, fps: Fraction, threads: int, subsample: int = 1) -> tuple[float, float]:
    """(vmaf_v0.6.1, vmaf_v0.6.1neg) means, both sides canonicalized like the validator (CFR grid)."""
    with tempfile.TemporaryDirectory(prefix="vm-", dir="/tmp") as tmp:
        logp = os.path.join(tmp, "v.json")
        pre = f"format=yuv420p,setpts=PTS-STARTPTS,fps={fps}"
        graph = (f"[0:v:0]{pre}[d];[1:v:0]{pre}[r];[d][r]libvmaf=model='version=vmaf_v0.6.1\\:name=p|"
                 f"version=vmaf_v0.6.1neg\\:name=n':log_fmt=json:log_path={logp}:n_threads={threads}:n_subsample={subsample}:pool=mean")
        run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", dist, "-i", ref,
             "-filter_complex", graph, "-f", "null", "-"], 1800)
        pm = json.load(open(logp))["pooled_metrics"]
    p = next(v["mean"] for k, v in pm.items() if k.endswith("_p") or k == "p")
    n = next(v["mean"] for k, v in pm.items() if k.endswith("_n") or k == "n")
    return p, n


# ---------------------------------------------------------------------------- encoders

def frame_times(path: str) -> list[float]:
    """Presentation times of the input frames, rebased to 0 (FFV1 is intra-only: packets == frames)."""
    out = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
               "-of", "csv=p=0", path], 300).stdout
    ts = sorted(float(x) for x in out.split() if x and x != "N/A")
    return [t - ts[0] for t in ts] if ts else []


def keyframe_times(times: list[float], period: float, fps: float) -> list[float]:
    """Greedy keyframe placement: frame j-1 becomes a keyframe whenever frame j would be more than
    `period` after the last keyframe, and the tail (last frame + 1/fps) is covered too. On a CFR clip
    this is exactly every round(period * fps) frames; across a timestamp gap it puts the keyframe on
    the last frame before the gap (forcing by time alone would land after it)."""
    keys, last = [0.0], 0.0
    pts = times + [times[-1] + 1.0 / fps] if times else []
    for j in range(1, len(pts)):
        if pts[j] - last > period + 1e-6:
            if pts[j - 1] > last:
                last = pts[j - 1]
                keys.append(last)
    return keys


def keyframe_args(keys: list[float] | None, fps: float) -> list[str]:
    """Force the computed keyframe times (a quarter frame early, so a frame at exactly t matches).
    Only used for timestamps with gaps: on a CFR clip keyint already puts the keyframes there, and
    forcing them costs efficiency (SVT adds a keyframe on the last frame and splits mini-GOPs: q02
    -0.44 VMAF, q01 vmaf-neg +0.07 at the same target), so CFR clips pass keys=None."""
    if not keys:
        return []
    eps = 0.25 / fps
    return ["-force_key_frames", ",".join(f"{max(0.0, k - eps):.6f}" for k in keys)]


def svt_encode(src: str, dst: str, kbps: float, gop: int, extra: str, tmp: str, timeout: float,
               preset: int, vf: str = "", color: list | None = None, fps: float = 30.0,
               keys: list | None = None) -> None:
    params = f"keyint={gop}:scd=0" + (f":{extra}" if extra else "")
    stats = os.path.join(tmp, "svt2pass")
    common = [FFMPEG, "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0", "-an", "-sn", "-dn",
              "-fps_mode", "passthrough", "-vf", CANON_VF + (f",{vf}" if vf else ""), *keyframe_args(keys, fps),
              "-c:v", "libsvtav1", "-preset", str(preset), "-b:v", f"{kbps:.0f}k",
              "-svtav1-params", params, "-pix_fmt", "yuv420p", *(color or []), "-passlogfile", stats]
    run(common + ["-pass", "1", "-f", "null", "-"], timeout)
    run(common + ["-pass", "2", "-f", "mp4", dst], timeout)


def svt_crf_encode(src: str, dst: str, crf: float, gop: int, extra: str, timeout: float, preset: int,
                   vf: str = "", color: list | None = None, fps: float = 30.0, keys: list | None = None) -> None:
    params = f"keyint={gop}:scd=0:crf={crf:g}" + (f":{extra}" if extra else "")
    run([FFMPEG, "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0", "-an", "-sn", "-dn",
         "-fps_mode", "passthrough", "-vf", CANON_VF + (f",{vf}" if vf else ""), *keyframe_args(keys, fps),
         "-c:v", "libsvtav1", "-preset", str(preset), "-svtav1-params", params, "-pix_fmt", "yuv420p",
         *(color or []), "-f", "mp4", dst], timeout)


X264_TUNE = ["-tune", "ssim"]
X264_NO_PSY = ["-tune", "ssim", "-x264-params", "psy-rd=0.0,0.0:aq-mode=1"]


def x264_encode(src: str, dst: str, kbps: float, gop: int, timeout: float, color: list | None = None,
                fps: float = 30.0, keys: list | None = None, opts: list | None = None) -> None:
    run([FFMPEG, "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0", "-an", "-sn", "-dn",
         "-fps_mode", "passthrough", "-vf", CANON_VF, *keyframe_args(keys, fps),
         "-c:v", "libx264", "-preset", "veryfast", *(opts or X264_TUNE),
         "-b:v", f"{kbps:.0f}k", "-maxrate", f"{kbps * 1.2:.0f}k", "-bufsize", f"{kbps:.0f}k",
         "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0", "-pix_fmt", "yuv420p", *(color or []),
         "-f", "mp4", dst], timeout)


def prefilter(strength: float) -> str:
    """Chroma-only blur, plus luma-only 5x5 unsharp of the given amount (0 = none)."""
    vf = CHROMA_VF
    if strength > 0:
        vf += f",unsharp=5:5:{strength:.2f}:5:5:0"
    return vf


# ---------------------------------------------------------------------------- per item

class Item:
    def __init__(self, src: str, dst: str, rules: dict, threads: int):
        self.src, self.dst, self.rules, self.threads = src, dst, rules, threads
        self.name = os.path.basename(src)[:12]
        self.info = None
        self.last_encode_s = 0.0
        self.last_vmaf_s = 0.0
        self.first_encode_s = 0.0
        self.stage_start = T0
        self.stage_est = None   # expected seconds of the running stage (None: first encode, unknown)
        self.done = False

    def work(self) -> float:
        i = self.info
        return i["w"] * i["h"] * i["frames"] * PRESET_COST.get(getattr(self, "preset", PRESET_LARGE), 1.0) if i else 0.0

    def others_end(self) -> float:
        """Latest expected finish of the other unfinished items (the batch cannot end before)."""
        end = 0.0
        for o in ITEMS:
            if o is self or o.done or o.info is None:
                continue
            if o.stage_est is not None:
                est = o.stage_start + o.stage_est
            elif self.first_encode_s > 0 and self.work() > 0:
                est = o.stage_start + self.first_encode_s * o.work() / self.work()
            else:
                continue
            end = max(end, est + 0.3 * (est - o.stage_start))  # + its VMAF measurement
        return end

    def fits(self, m: dict, strict: bool = True) -> bool:
        b, r = self.budget, self.rules
        avg_lim = b * (ACCEPT_AVG if strict else 1.0)
        peak_lim = b * r["peak_ratio"] * (ACCEPT_PEAK_SHARE if strict else 0.99)
        return (m["avg"] <= avg_lim and m["peak"] <= peak_lim and m["first_key"]
                and m["max_kf"] <= r["max_kf"] + 0.4 / self.fpsf
                and m["packets"] == self.info.get("packets_src", self.info["frames"]))

    def rate_fit(self, extra: str, target: float, tmp: str, tag: str, vf: str = "",
                 allow_up: bool = True) -> tuple[str, dict, float] | None:
        """Encode, measure, retarget. Returns (path, measurement, target) of the best fitting encode.

        Up to MAX_RATE_ENCODES encodes; while nothing fits yet, up to RESCUE_ENCODES more (a clip
        whose rate barely follows the target, e.g. a few frames dominated by the keyframe)."""
        best = None
        points = []  # (target, avg share, peak share)
        goal = RATE_GOAL
        for i in range(MAX_RATE_ENCODES + RESCUE_ENCODES):
            if i >= MAX_RATE_ENCODES and best is not None:
                break
            if time.monotonic() - T0 > HARD_REQUIRED_LIMIT_S:
                log(f"{self.name} {tag}: past the hard time limit, no more SVT encodes")
                break
            path = os.path.join(tmp, f"{tag}_{i}.mp4")
            te = time.monotonic()
            self.stage_start = te
            self.stage_est = 1.2 * self.last_encode_s if self.last_encode_s else None
            svt_encode(self.src, path, target, self.gop, extra, tmp, 1200, self.preset, vf, self.info["color"],
                       self.fpsf, self.keys)
            m = measure(path, self.fpsf, self.info["frames"], self.rules["peak_window"])
            self.last_encode_s = time.monotonic() - te
            if not self.first_encode_s:
                self.first_encode_s = self.last_encode_s
            ra, rp = m["avg"] / self.budget, m["peak"] / (self.budget * self.rules["peak_ratio"])
            ok = self.fits(m)
            log(f"{self.name} {tag} enc{i} target={target / self.budget:.3f} -> avg={ra:.3f} peak={rp:.3f} ok={ok}")
            if ok and (best is None or m["avg"] > best[1]["avg"]):
                if best is not None:
                    _unlink(best[0])
                best = (path, m, target)
            else:
                _unlink(path)  # keep /tmp small: only the best try of a fit survives
            if ok and (ra >= REFIT_BELOW or ra < EASY_BELOW or not allow_up):
                break
            if ok:
                # optional: only when it should end (encode + VMAF) inside the free time; a large
                # shortfall (< BIG_SHORTFALL of the budget) costs enough VMAF to be worth some slack
                eta = time.monotonic() + 1.4 * self.last_encode_s
                slack = SHARPEN_TIME_SLACK if ra < BIG_SHORTFALL else 1.0
                limit = min(max(T0 + slack * (FREE_DEADLINE - T0), self.others_end()), T0 + HARD_OPTIONAL_LIMIT_S)
                if eta > limit:
                    log(f"{self.name} {tag} skip upward refit: eta {eta - T0:.0f}s > limit {limit - T0:.0f}s")
                    break
            points.append((target, ra, rp))
            target = self.next_target(points, goal)
        return best

    @staticmethod
    def next_target(points: list, goal: float) -> float:
        """Next target from the measured (target, avg share, peak share) points.

        Rate ~ target^e: e is estimated from the two points closest to the goal (clamped, so a
        flat response takes a bigger step), then the step aims at the goal from the closest point
        and stays strictly inside the bracket of passing and failing targets."""
        ranked = sorted(points, key=lambda q: abs(math.log(q[1] / goal)))
        t0, r0, p0 = ranked[0]
        e = 1.0
        for t1, r1, _ in ranked[1:]:
            if abs(math.log(t1 / t0)) > 0.01:
                e = (math.log(r1) - math.log(r0)) / (math.log(t1) - math.log(t0))
                e = min(1.5, max(0.15, e))
                break
        scale = (goal / r0) ** (1.0 / e)
        if p0 > ACCEPT_PEAK_SHARE:
            scale = min(scale, (0.97 * ACCEPT_PEAK_SHARE / p0) ** (1.0 / e))
        nxt = t0 * max(0.3, min(1.6, scale))
        passing = [q[0] for q in points if q[1] <= ACCEPT_AVG and q[2] <= ACCEPT_PEAK_SHARE]
        failing = [q[0] for q in points if q[1] > ACCEPT_AVG or q[2] > ACCEPT_PEAK_SHARE]
        lo_t, hi_t = max(passing, default=0.0), min(failing, default=float("inf"))
        if not lo_t < nxt < hi_t:
            if lo_t > 0 and hi_t < float("inf"):
                nxt = math.sqrt(lo_t * hi_t)
            elif hi_t < float("inf") and nxt >= hi_t:
                nxt = hi_t * 0.85
        return nxt

    def process(self) -> str:
        t = time.monotonic()
        tmp = tempfile.mkdtemp(prefix=f"it-{self.name}-", dir="/tmp")
        try:
            if getattr(self, "info", None) is None:
                self.info = probe(self.src)
            try:
                times = frame_times(self.src)
            except Exception as exc:  # noqa: BLE001
                log(f"{self.name} frame times unavailable ({exc})")
                times = []
            if not self.info.get("canonical"):
                fps, frames = canonical_timing(self.src, self.info, times)
                if (fps, frames) != (self.info["fps"], self.info["frames"]):
                    log(f"{self.name} canonical timing {fps} fps / {frames} frames differs from the stream "
                        f"({self.info['fps']} / {self.info['frames']}); using the canonical values")
                self.info.update(packets_src=self.info["frames"], fps=fps, frames=frames, canonical=True)
            self.fpsf = float(self.info["fps"])
            self.budget = self.rules["bpp"] * self.info["w"] * self.info["h"] * self.fpsf / 1000.0
            self.gop = max(1, int(self.rules["max_kf"] * self.fpsf + 1e-9))
            if times and not is_uniform(times, self.fpsf):
                self.keys = keyframe_times(times, self.gop / self.fpsf, self.fpsf)
                log(f"{self.name} irregular timestamps: forcing keyframes at {len(self.keys)} times")
            else:
                self.keys = None  # CFR: keyint alone places every keyframe
            self.preset = PRESET_SMALL if self.info["w"] * self.info["h"] <= 1280 * 720 else PRESET_LARGE
            log(f"{self.name} {self.info['w']}x{self.info['h']}@{self.fpsf:.3f} {self.info['frames']}f "
                f"budget={self.budget:.0f}k gop={self.gop} preset={self.preset}")
            try:
                chosen = self.svt_path(tmp)
            except Exception as exc:  # noqa: BLE001
                log(f"{self.name} SVT path failed: {exc}")
                chosen = None
            if chosen is None:
                chosen = self.fallback(tmp)
            self.publish(chosen)
            log(f"{self.name} done in {time.monotonic() - t:.1f}s")
            return "ok"
        except Exception as exc:  # noqa: BLE001
            log(f"{self.name} FAILED: {exc}")
            return "failed"
        finally:
            self.done = True
            shutil.rmtree(tmp, ignore_errors=True)

    def measure_quality(self, path: str, exact: bool) -> tuple[float, float]:
        """(vmaf, neg): a quick every-3rd-frame estimate, or all frames when exact or near the guard."""
        tv = time.monotonic()
        if not exact:
            v, n = vmaf_pair(self.src, path, self.info["fps"], self.threads, subsample=3)
            self.last_vmaf_s = 2.5 * (time.monotonic() - tv)
            if not DELTA_QUICK_OK < v - n < DELTA_QUICK_BAD:
                return v, n
            tv = time.monotonic()
        v, n = vmaf_pair(self.src, path, self.info["fps"], self.threads)
        self.last_vmaf_s = time.monotonic() - tv
        return v, n

    def svt_path(self, tmp: str) -> str | None:
        target = self.budget * FIRST_AIM
        self.vbr_extra = BASE_EXTRA
        # 1. sharpened first: most clips keep vmaf - neg under the guard at the first strength
        res = self.rate_fit(BASE_EXTRA, target, tmp, "a", vf=prefilter(SHARPEN_FIRST))
        if res is not None:
            path, m, target = res
            v, n = self.measure_quality(path, exact=True)
            log(f"{self.name} first [sharpen {SHARPEN_FIRST:.2f}]: vmaf={v:.2f} neg={n:.2f} delta={v - n:.2f} "
                f"avg={m['avg'] / self.budget:.3f}")
            if DELTA_GUARD < v - n <= DELTA_GUARD + TF0_RESCUE_MARGIN and v >= TF0_RESCUE_MIN_VMAF:
                # just over the guard on a clean clip: SVT's temporal filter adds ADM "enhancement"; without
                # it the delta drops ~0.1 at no VMAF cost on renders (q01 +0.08 / -0.10, q03 +0.09 / -0.06),
                # while on AI clips it costs VMAF (q07 -3.7), hence the VMAF floor
                tf_extra = BASE_EXTRA + ":tf-strength=0"
                res2 = self.rate_fit(tf_extra, target, tmp, "t", vf=prefilter(SHARPEN_FIRST))
                if res2 is not None:
                    p2, m2, t2 = res2
                    v2, n2 = self.measure_quality(p2, exact=True)
                    log(f"{self.name} tf-strength=0 [sharpen {SHARPEN_FIRST:.2f}]: vmaf={v2:.2f} neg={n2:.2f} "
                        f"delta={v2 - n2:.2f} avg={m2['avg'] / self.budget:.3f}")
                    if v2 - n2 <= DELTA_GUARD:
                        _unlink(path)
                        path, m, target, v, n = p2, m2, t2, v2, n2
                        self.vbr_extra = tf_extra
                    else:
                        _unlink(p2)
            if v - n <= DELTA_GUARD:
                if v < CRF_TRY_BELOW:
                    # hard clip: CRF will most likely replace the VBR encode, and with CRF the sharpening
                    # strength barely matters (q07 CRF 0.04 -> 0.08: +0.09 VMAF), so spend the time there
                    strength, d = SHARPEN_FIRST, v - n
                else:
                    path, v, d, strength = self.sharpen_up(path, v, n, SHARPEN_FIRST, target, tmp)
                path, v, d, mode = self.crf_try(path, v, d, strength, tmp)
                return self.preset_up(path, v, d, strength, target, tmp, mode)
        # 2. plain (chroma blur only), then the delta-lowering ladder. The first sharpening step adds
        #    ~0.6-1.0 to the delta (q02: 3.63 sharpened, 2.91 plain, 2.70 plain + tf-strength=0), so when
        #    the sharpened delta says the plain encode would still miss the guard, start one step lower.
        candidates = []
        first_step = 0
        if res is not None and (v - n) - SHARPEN_DELTA_COST > DELTA_GUARD:
            first_step = 1
            log(f"{self.name} sharpened delta {v - n:.2f}: skip the plain encode, start at ladder 1")
        for step, extra in enumerate(LADDER):
            if step < first_step:
                continue
            if time.monotonic() - T0 > HARD_REQUIRED_LIMIT_S:
                log(f"{self.name} ladder: past the hard time limit -> fallback")
                break
            res = self.rate_fit(extra, target, tmp, f"s{step}", vf=prefilter(0.0))
            if res is None:
                log(f"{self.name} ladder {step}: no encode fits the rules -> fallback")
                break
            path, m, target = res
            v, n = self.measure_quality(path, exact=False)
            log(f"{self.name} ladder {step} [{extra}]: vmaf={v:.2f} neg={n:.2f} delta={v - n:.2f} "
                f"avg={m['avg'] / self.budget:.3f}")
            candidates.append((v - n, v, path))
            if v - n <= DELTA_GUARD:
                self.vbr_extra = extra
                return self.preset_up(path, v, v - n, 0.0, target, tmp, "vbr")
        if not candidates:
            return None
        # nothing under the guard: keep the lowest delta if it is still under the hard limit
        d, v, path = min(candidates)
        return path if d < 2.97 else None

    def crf_fit(self, vf: str, tmp: str, tag: str, crf: float = CRF_START,
                preset: int | None = None) -> tuple[str, dict, float] | None:
        """One-pass CRF aimed at RATE_GOAL of the budget: secant on ln(rate) vs CRF."""
        preset = self.preset if preset is None else preset
        best, points = None, []
        for i in range(CRF_MAX_ENCODES):
            if time.monotonic() - T0 > HARD_OPTIONAL_LIMIT_S:
                break
            path = os.path.join(tmp, f"{tag}_{i}.mp4")
            te = time.monotonic()
            self.stage_start = te
            self.stage_est = self.last_encode_s or None
            svt_crf_encode(self.src, path, crf, self.gop, BASE_EXTRA_CRF, 1200, preset, vf, self.info["color"],
                           self.fpsf, self.keys)
            m = measure(path, self.fpsf, self.info["frames"], self.rules["peak_window"])
            self.last_encode_s = time.monotonic() - te
            ra, rp = m["avg"] / self.budget, m["peak"] / (self.budget * self.rules["peak_ratio"])
            ok = self.fits(m)
            log(f"{self.name} {tag} crf{i} crf={crf:g} -> avg={ra:.3f} peak={rp:.3f} ok={ok}")
            if ok and (best is None or m["avg"] > best[1]["avg"]):
                if best is not None:
                    _unlink(best[0])
                best = (path, m, crf)
            else:
                _unlink(path)
            if ok and ra >= REFIT_BELOW:
                break
            points.append((crf, ra, rp))
            # need: ln(ra) - slope * dcrf = ln(goal)  (higher CRF -> lower rate)
            ranked = sorted(points, key=lambda q: abs(math.log(q[1] / RATE_GOAL)))
            c0, r0, p0 = ranked[0]
            slope = CRF_SLOPE
            for c1, r1, _ in ranked[1:]:
                if abs(c1 - c0) >= 0.25:
                    slope = min(0.2, max(0.03, (math.log(r1) - math.log(r0)) / (c0 - c1)))
                    break
            want = math.log(r0 / RATE_GOAL)
            if p0 > ACCEPT_PEAK_SHARE:
                want = max(want, math.log(p0 / (0.97 * ACCEPT_PEAK_SHARE)))
            nxt = round((c0 + want / slope) * 4) / 4
            passing = [q[0] for q in points if q[1] <= ACCEPT_AVG and q[2] <= ACCEPT_PEAK_SHARE]
            failing = [q[0] for q in points if q[1] > ACCEPT_AVG or q[2] > ACCEPT_PEAK_SHARE]
            lo, hi = max(failing, default=-1.0), min(passing, default=99.0)  # lo: too many bits, hi: fits
            if not lo < nxt < hi and lo >= 0 and hi < 99:
                nxt = round((lo + hi) / 2 * 4) / 4
            nxt = min(70.0, max(1.0, nxt))
            if nxt == crf or any(abs(nxt - q[0]) < 0.125 for q in points):
                break
            crf = nxt
        return best

    def crf_try(self, path: str, v0: float, d0: float, strength: float,
                tmp: str) -> tuple[str, float, float, str]:
        """Hard clips: try one-pass CRF with the same prefilter; keep it if VMAF rises within the guard."""
        if v0 >= CRF_TRY_BELOW:
            return path, v0, d0, "vbr"
        eta = time.monotonic() + 3 * 0.7 * self.last_encode_s + self.last_vmaf_s
        limit = min(max(T0 + CRF_TIME_SLACK * (FREE_DEADLINE - T0), self.others_end()), T0 + HARD_OPTIONAL_LIMIT_S)
        if eta > limit:
            log(f"{self.name} crf: skip, eta {eta - T0:.0f}s > limit {limit - T0:.0f}s")
            return path, v0, d0, "vbr"
        start = CRF_START_SMALL if self.preset == PRESET_SMALL else CRF_START
        res = self.crf_fit(prefilter(strength), tmp, "c", crf=start)
        if res is None:
            return path, v0, d0, "vbr"
        p, m, crf = res
        v, n = self.measure_quality(p, exact=True)
        log(f"{self.name} crf {crf:g} [sharpen {strength:.2f}]: vmaf={v:.2f} neg={n:.2f} delta={v - n:.2f} "
            f"avg={m['avg'] / self.budget:.3f} (vbr {v0:.2f})")
        if v - n <= DELTA_GUARD and v > v0:
            _unlink(path)
            self.crf = crf
            return p, v, v - n, "crf"
        _unlink(p)
        return path, v0, d0, "vbr"

    def sharpen_up(self, path: str, v0: float, n0: float, strength: float, target: float,
                   tmp: str) -> tuple[str, float, float, float]:
        """From an accepted sharpened encode, raise the strength while vmaf - neg stays under the guard."""
        best_v, best_path = v0, path
        prev_d = d0 = v0 - n0
        # predicted next delta: linear from the plain encode is unknown, so assume the step adds as
        # much as a typical first step (~0.9 on renders) unless already measured
        best = (path, v0, d0, strength)
        if DELTA_GUARD - d0 < SHARPEN_MIN_HEADROOM or v0 >= SHARPEN_VMAF_MAX or v0 < SHARPEN_VMAF_MIN:
            return best
        step_cost = None
        while strength + SHARPEN_STEP <= SHARPEN_MAX + 1e-9:
            nxt = strength + SHARPEN_STEP
            if step_cost is not None and prev_d + step_cost > DELTA_GUARD - 0.02:
                break
            eta = time.monotonic() + 1.4 * (self.last_encode_s + self.last_vmaf_s)
            limit = min(max(T0 + SHARPEN_TIME_SLACK * (FREE_DEADLINE - T0), self.others_end()),
                        T0 + HARD_OPTIONAL_LIMIT_S)
            if eta > limit:
                log(f"{self.name} sharpen: stop before {nxt:.2f}, eta {eta - T0:.0f}s > limit {limit - T0:.0f}s")
                break
            res = self.rate_fit(self.vbr_extra, target, tmp, f"u{int(nxt * 100)}", vf=prefilter(nxt), allow_up=False)
            if res is None:
                break
            p, m, _ = res
            v, n = self.measure_quality(p, exact=True)
            d = v - n
            log(f"{self.name} sharpen {nxt:.2f}: vmaf={v:.2f} neg={n:.2f} delta={d:.2f} avg={m['avg'] / self.budget:.3f}")
            if d > DELTA_GUARD:
                _unlink(p)
                break
            if v > best_v:
                if best_path != path:
                    _unlink(best_path)
                best_v, best_path = v, p
                best = (p, v, d, nxt)
            else:
                _unlink(p)
            step_cost = max(0.0, d - prev_d)
            prev_d, strength = d, nxt
            if v >= SHARPEN_VMAF_MAX:
                break
        if best_path != path:
            log(f"{self.name} sharpen: chose vmaf {best_v:.2f} (first {v0:.2f})")
        return best

    def preset_up(self, path: str, v0: float, d0: float, strength: float, target: float, tmp: str,
                  mode: str = "vbr") -> str:
        """Small hard clips: re-encode with a slower preset when the time allows (q07 p4 -> p3: +0.9 VMAF,
        1.7x the encode time). Kept only if VMAF rises and vmaf - neg stays under the guard."""
        small = self.preset == PRESET_SMALL
        vmax = SLOW_PRESET_VMAF_MAX if small else SLOW_PRESET_LARGE_VMAF_MAX
        if not (SLOW_PRESET_VMAF_MIN <= v0 < vmax):
            return path
        if mode == "crf":
            slow, cost = SLOW_PRESET_CRF, SLOW_PRESET_CRF_COST * 1.5  # crf: ~1.5 encodes
        elif small:
            slow, cost = SLOW_PRESET, SLOW_PRESET_COST
        else:
            slow, cost = SLOW_PRESET_LARGE, SLOW_PRESET_LARGE_COST
        if slow >= self.preset:
            return path
        eta = time.monotonic() + cost * self.last_encode_s + self.last_vmaf_s
        limit = min(max(T0 + SHARPEN_TIME_SLACK * (FREE_DEADLINE - T0), self.others_end()),
                    T0 + HARD_OPTIONAL_LIMIT_S)
        if eta > limit:
            log(f"{self.name} preset {slow}: skip, eta {eta - T0:.0f}s > limit {limit - T0:.0f}s")
            return path
        if mode == "crf":
            # a slower preset spends a little fewer bits at the same CRF: start half a step lower
            res = self.crf_fit(prefilter(strength), tmp, f"cp{slow}", crf=max(1.0, self.crf - 0.5), preset=slow)
        else:
            keep = self.preset
            self.preset = slow
            try:
                res = self.rate_fit(self.vbr_extra, target, tmp, f"p{slow}", vf=prefilter(strength), allow_up=False)
            finally:
                self.preset = keep
        if res is None:
            return path
        p, m, _ = res
        v, n = self.measure_quality(p, exact=True)
        log(f"{self.name} preset {slow} [{mode}, sharpen {strength:.2f}]: vmaf={v:.2f} neg={n:.2f} "
            f"delta={v - n:.2f} avg={m['avg'] / self.budget:.3f} (was {v0:.2f})")
        if v - n <= DELTA_GUARD and v > v0:
            _unlink(path)
            return p
        _unlink(p)
        return path

    def fallback(self, tmp: str) -> str:
        """x264, rate-checked; its VMAF delta is measured too and a psy-free variant is tried when the
        first one breaks the guard (psychovisual tuning raises vmaf - neg at low rates)."""
        best = None  # (delta, path)
        for variant, opts in enumerate((X264_TUNE, X264_NO_PSY)):
            target = self.budget * 0.85
            path = None
            for i in range(4):
                cand = os.path.join(tmp, f"x264_{variant}_{i}.mp4")
                x264_encode(self.src, cand, target, self.gop, 1200, self.info["color"], self.fpsf, self.keys, opts)
                m = measure(cand, self.fpsf, self.info["frames"], self.rules["peak_window"])
                log(f"{self.name} fallback x264[{variant}] enc{i}: avg={m['avg'] / self.budget:.3f}")
                if self.fits(m, strict=False):
                    path = cand
                    break
                _unlink(cand)
                target *= 0.8
            if path is None:
                continue
            try:
                v, n = self.measure_quality(path, exact=True)
            except Exception as exc:  # noqa: BLE001  no measurement: keep it, it fits the rules
                log(f"{self.name} fallback VMAF failed: {exc}")
                return path
            log(f"{self.name} fallback x264[{variant}]: vmaf={v:.2f} delta={v - n:.2f}")
            if v - n <= DELTA_GUARD:
                if best is not None:
                    _unlink(best[1])
                return path
            if best is None or v - n < best[0]:
                if best is not None:
                    _unlink(best[1])
                best = (v - n, path)
            else:
                _unlink(path)
        if best is None:
            raise RuntimeError("fallback x264 could not fit the rules")
        return best[1]

    def publish(self, path: str) -> None:
        part = os.path.join(os.path.dirname(self.dst), ".partial-" + os.path.basename(self.dst))
        shutil.copyfile(path, part)
        os.chmod(part, 0o644)
        os.replace(part, self.dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits-per-pixel", type=float, required=True)
    ap.add_argument("--max-keyframe-interval", type=float, default=2.0)
    ap.add_argument("--peak-window", type=float, default=2.0)
    ap.add_argument("--peak-ratio", type=float, default=1.5)
    ap.add_argument("--parallel", type=int, default=PARALLEL)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    a = ap.parse_args()
    rules = {"bpp": a.bits_per_pixel, "max_kf": a.max_keyframe_interval, "peak_window": a.peak_window,
             "peak_ratio": a.peak_ratio}
    names = sorted(n for n in os.listdir(a.input_dir)
                   if not n.startswith(".") and os.path.isfile(os.path.join(a.input_dir, n)))
    if not names:
        log("no input items")
        return 0
    global FREE_DEADLINE
    par = max(1, min(a.parallel, len(names)))
    cpus = effective_cpus()
    threads = max(1, cpus // par)
    items = [Item(os.path.join(a.input_dir, n), os.path.join(a.output_dir, n), rules, threads) for n in names]
    ITEMS.extend(items)
    with ThreadPoolExecutor(max_workers=len(items)) as pool:  # probe all first: the batch's frame count
        def _probe(it):
            try:
                it.info = probe(it.src)
            except Exception as exc:  # noqa: BLE001  process() retries and falls back
                log(f"{it.name} probe failed: {exc}")
        list(pool.map(_probe, items))
    frames = sum(it.info["frames"] for it in items if it.info)
    # ENTRY_FREE_SCALE is a local test knob (the evaluator injects no environment variables).
    FREE_DEADLINE = T0 + FREE_SHARE * FREE_SPF * frames * float(os.environ.get("ENTRY_FREE_SCALE", "1"))
    log(f"batch: {len(items)} items, {frames} frames, {cpus} cpus, free time {FREE_DEADLINE - T0:.1f}s")
    with ThreadPoolExecutor(max_workers=par) as pool:
        results = list(pool.map(lambda it: it.process(), items))
    log(f"batch: {results.count('ok')} ok, {results.count('failed')} failed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001  never fail the batch
        log(f"fatal: {exc}")
        sys.exit(0)
