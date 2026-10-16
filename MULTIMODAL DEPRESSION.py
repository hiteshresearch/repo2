# ============================================================
# MULTIMODAL DEPRESSION DETECTION (TEXT + AUDIO) — V13
# Top-K acoustic segment selection BEFORE the audio encoder (edge / bandwidth-aware)
# ============================================================
# Kaggle: run this in the first cell
# !pip install -q transformers librosa pandas scikit-learn tqdm seaborn matplotlib scipy onnx onnxruntime opensmile
#
# What is new vs V12
#   [S1] Cheap on-device chunk features: per-chunk log-mel statistics (64 x mean/std) + per-chunk
#        eGeMAPS (88). Computed with librosa/openSMILE only, i.e. before any neural encoder.
#   [S2] ChunkScorer: tiny MLP (D_cheap -> 32 -> 1) that scores every chunk.
#   [S3] Selection: K = ceil(KEEP_FRAC * T) chunks are kept, in temporal order.
#        Training: Gumbel noise + relaxed (soft) Top-K, straight-through estimator
#                  (hard Top-K in the forward pass, soft gradient in the backward pass),
#                  temperature annealed TAU_START -> TAU_END, optional self-distillation
#                  from the model's own audio attention over all chunks (SEL_DISTILL).
#        Inference: hard Top-K; the ensemble uses ONE shared selection (mean of per-member
#                  z-scored scores) so WavLM runs only on K chunks per session.
#   [S4] Baseline selectors trained with the same protocol: all / first / random / energy /
#        learned_wavlm (scorer on WavLM embeddings = upper bound with no compute savings).
#   [S5] K-sweep (accuracy vs % audio kept), uplink-bytes table, selection analysis
#        (position, overlap with loudness, acoustic profile of kept chunks).
#   [S6] Deployment: chunk_scorer.onnx + audio_encoder.onnx (K chunks only) + fusion_head.onnx,
#        end-to-end ONNX pipeline with per-stage timing (all chunks vs Top-K).
# NOTE: the cache format changed (per-chunk features, chunk length) -> cache is rebuilt once.
# ============================================================
#!pip install -q transformers librosa pandas scikit-learn tqdm seaborn matplotlib scipy onnx onnxruntime opensmile
import os, re, json, copy, random, warnings, zlib, time
from itertools import combinations
from math import factorial
import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor

warnings.filterwarnings("ignore", category=UserWarning)
try:
    import opensmile
    HAS_SMILE = True
except ImportError:
    HAS_SMILE = False
    print("opensmile not installed -> eGeMAPS branch disabled, scorer uses log-mel features only")

# ============================================================
# CONFIG
# ============================================================
TEXT_PATH  = "/home/hp2/Documents/archive(1)/E_DAIC"
AUDIO_PATH = "/home/hp2/Documents/archive/Audio"
OUT_DIR    = "./kaggle/working/outputs_v13_15_seconds_wavlm"
METADATA_CSV = None   # optional: CSV with Participant_ID, Gender -> subgroup analysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoders
TEXT_MODEL = "/home/hp2/models/all-mpnet-base-v2-transformers-default-v1"
TEXT_POOLING = "mean_norm" if TEXT_MODEL.startswith("sentence-transformers") else "cls_mean"
#WAVLM_NAME = "microsoft/wavlm-base-plus"
WAVLM_NAME = "/home/hp2/models/Wavlm-base-plus"
#AUTO_LAYERS = True
AUTO_LAYERS = False
LAYER_WIN = 2
DEFAULT_LAYERS = (9, 10)
USE_EGEMAPS = HAS_SMILE           # speaker-level eGeMAPS branch (88 floats per session)

# Preprocessing
SR = 16000
CHUNK_SEC = 15                    # shorter chunks = finer selection (5 s is finer still, but the cache grows)
CHUNK_SIZE = SR * CHUNK_SEC
MAX_AUDIO_CHUNKS = 120
USE_PARTICIPANT_SEGMENTS = True
MIN_WORDS = 1
MAX_SENTS = 400
EMB_BATCH, AUDIO_BATCH = 32, 16

# ---- [S1-S3] Top-K acoustic segment selection ----
SELECTOR = "learned"              # learned | learned_wavlm | energy | random | first | all
KEEP_FRAC = 0.2                   # K = ceil(KEEP_FRAC * T); 1.0 = no selection
K_MIN = 2
LEARNED = ("learned", "learned_wavlm")
CHEAP_EGEMAPS = HAS_SMILE         # per-chunk eGeMAPS in the scorer input (else log-mel stats only)
N_MELS, MEL_NFFT, MEL_HOP = 64, 400, 160
SCORER_HIDDEN = 32
GUMBEL_SCALE = 1.0
TAU_START, TAU_END = 1.0, 0.1     # relaxed Top-K temperature, annealed geometrically over epochs
SEL_DISTILL = 0.1                 # weight of KL(audio attention over all chunks || scorer); 0 = off

# Main experiments: (name, branch mode, selector, keep fraction)
MAIN_CONFIGS = [("text", "text", "all", 1.0),
                ("audio", "audio", SELECTOR, KEEP_FRAC),
                ("fusion", "fusion", SELECTOR, KEEP_FRAC),        # proposed system
                ("fusion_all", "fusion", "all", 1.0)]             # same model, no selection
# K-sweep (fusion, fewer seeds to keep the run time reasonable)
RUN_SWEEP = True
SWEEP_FRACS = [0.1, 0.2, 0.3, 0.5]
SWEEP_SELECTORS = ["learned", "learned_wavlm", "energy", "random", "first"]
SWEEP_SEEDS = [0]

# Model
SEQ_LEN, STRIDE = 50, 30
HIDDEN = 64
GM_DIM = 32
IN_DROPOUT, HEAD_DROPOUT = 0.0870004496126405, 0.223751973691394

# Training / evaluation
EPOCHS, MIN_EPOCHS, PATIENCE = 40, 8, 10
LR, WD, ACCUM = 0.000646, 0.00471, 1
N_FOLDS, CV_SEED = 5, 42
SEEDS = [0, 1, 2]
THR_METRIC = "f1_macro"
AUGMENT, AUG_FRAC = True, (0.6, 1.0)
N_VIEWS, TTA_FRAC = 5, 0.8        # TTA views are drawn INSIDE the selected chunks (no extra WavLM calls)
DETERMINISTIC = False

# Interpretability
IG_STEPS = 32
TOPK_SENTS = 12
DEL_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_RANDOM_DEL = 5
N_BOOT = 1000
OPSET = 17

_tag = lambda s: re.sub(r"[^A-Za-z0-9]+", "-", s)
CACHE_DIR = os.path.join(OUT_DIR, "cache",
                         f"{_tag(TEXT_MODEL)}_{_tag(WAVLM_NAME)}_mw{MIN_WORDS}_ps{int(USE_PARTICIPANT_SEGMENTS)}"
                         f"_gm{int(USE_EGEMAPS)}_c{CHUNK_SEC}_mx{MAX_AUDIO_CHUNKS}_mel{N_MELS}_ce{int(CHEAP_EGEMAPS)}")
FIG_DIR, ONNX_DIR = os.path.join(OUT_DIR, "figures"), os.path.join(OUT_DIR, "onnx")
for d in (OUT_DIR, CACHE_DIR, FIG_DIR, ONNX_DIR):
    os.makedirs(d, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = False


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.show(); plt.close()


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, float)))

# ============================================================
# FROZEN ENCODERS
# ============================================================
class TextEncoder(nn.Module):
    def __init__(self, enc, pooling):
        super().__init__(); self.enc, self.pooling = enc, pooling

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.pooling == "mean_norm":
            return mean / mean.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return (h[:, 0] + mean) / 2


class AudioEncoder(nn.Module):
    def __init__(self, wavlm, normalize, l0, l1):
        super().__init__()
        self.wavlm, self.normalize, self.l0, self.l1 = wavlm, normalize, l0, l1

    def hidden_states(self, x):                       # (B, CHUNK_SIZE) -> (13, B, T', 768)
        if self.normalize:
            x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-7)
        return torch.stack(self.wavlm(x, output_hidden_states=True).hidden_states)

    def forward(self, x):                             # (B, CHUNK_SIZE) -> (B, 768)
        return self.hidden_states(x)[self.l0:self.l1].mean(0).mean(1)


def _load(cls, name):
    try:
        return cls.from_pretrained(name, attn_implementation="eager")
    except (TypeError, ValueError):
        return cls.from_pretrained(name)


print("Loading encoders")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_NAME)
WAVLM_NORMALIZE = bool(getattr(feature_extractor, "do_normalize", False))
text_encoder = TextEncoder(_load(AutoModel, TEXT_MODEL), TEXT_POOLING).to(device).eval()
audio_encoder = AudioEncoder(_load(AutoModel, WAVLM_NAME), WAVLM_NORMALIZE, *DEFAULT_LAYERS).to(device).eval()
for p in list(text_encoder.parameters()) + list(audio_encoder.parameters()):
    p.requires_grad = False
D_TEXT = text_encoder.enc.config.hidden_size

SMILE = (opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                         feature_level=opensmile.FeatureLevel.Functionals) if HAS_SMILE else None)
GM_NAMES = list(SMILE.feature_names) if HAS_SMILE else []
N_GM = len(GM_NAMES) if USE_EGEMAPS else 0
CHEAP_NAMES = ([f"logmel{i}_mean" for i in range(N_MELS)] + [f"logmel{i}_std" for i in range(N_MELS)]
               + (GM_NAMES if CHEAP_EGEMAPS else []))
GM_SLICE = slice(2 * N_MELS, 2 * N_MELS + len(GM_NAMES))     # per-chunk eGeMAPS inside the cheap features

# ============================================================
# PREPROCESSING (pure numpy / librosa / openSMILE: identical on the device)
# ============================================================
def load_transcript(csv_path):
    df = pd.read_csv(csv_path)
    lc = {c.lower().strip(): c for c in df.columns}
    if "speaker" in lc:
        df = df[~df[lc["speaker"]].astype(str).str.lower().str.contains("ellie")]
    tcol = lc.get("text", lc.get("value"))
    if tcol is None:
        raise ValueError(f"No text column in {csv_path}")
    texts = df[tcol].astype(str).str.strip().str.lower().tolist()
    segs = []
    if "start_time" in lc and "end_time" in lc:
        segs = list(zip(df[lc["start_time"]].astype(float), df[lc["end_time"]].astype(float)))
    sents = [t for t in texts if t not in ("", "nan") and len(t.split()) >= MIN_WORDS][:MAX_SENTS]
    return sents, segs


def _concat_to_orig(s, offsets):
    for pos, a, length in offsets:
        if pos <= s < pos + length:
            return a + (s - pos)
    return s


def load_participant_audio(wav_path, segs):
    """Returns chunks (T, CHUNK_SIZE), chunk start times (s, original recording), participant signal."""
    y, _ = librosa.load(wav_path, sr=SR)
    offsets, y_use = [], y
    if USE_PARTICIPANT_SEGMENTS and segs:
        pieces, pos = [], 0
        for s, e in segs:
            a, b = int(s * SR), min(int(e * SR), len(y))
            if b - a < int(0.3 * SR):
                continue
            pieces.append(y[a:b]); offsets.append((pos, a, b - a)); pos += b - a
        if pieces and pos >= CHUNK_SIZE:
            y_use = np.concatenate(pieces)
        else:
            offsets = []
    if not offsets:
        y_use, offsets = y, [(0, 0, len(y))]
    y_use = librosa.util.normalize(y_use).astype(np.float32)
    n = len(y_use) // CHUNK_SIZE
    if n == 0:
        return np.zeros((0, CHUNK_SIZE), np.float32), np.zeros(0), y_use
    starts = np.arange(n) * CHUNK_SIZE
    if n > MAX_AUDIO_CHUNKS:
        starts = starts[np.unique(np.linspace(0, n - 1, MAX_AUDIO_CHUNKS).round().astype(int))]
    chunks = np.stack([y_use[s:s + CHUNK_SIZE] for s in starts]).astype(np.float32)
    times = np.array([_concat_to_orig(s, offsets) for s in starts]) / SR
    return chunks, times, y_use


def speaker_egemaps(y_use):
    return np.nan_to_num(SMILE.process_signal(y_use, SR).iloc[0].to_numpy(np.float32))


def chunk_energy_db(c):
    return float(10.0 * np.log10(np.mean(c.astype(np.float64) ** 2) + 1e-10))


def chunk_features(chunks):
    """[S1] Cheap per-chunk features, computed on the device BEFORE any neural encoder.
    (T, CHUNK_SIZE) -> feats (T, D_CHEAP) float32 [log-mel means | log-mel stds | eGeMAPS], energy (T,) dB."""
    feats, energy = [], []
    for c in chunks:
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=c, sr=SR, n_fft=MEL_NFFT, hop_length=MEL_HOP,
                                                               n_mels=N_MELS), ref=1.0, top_db=None)
        f = [m.mean(1), m.std(1)]
        if CHEAP_EGEMAPS:
            f.append(np.nan_to_num(SMILE.process_signal(c, SR).iloc[0].to_numpy(np.float32)))
        feats.append(np.concatenate(f).astype(np.float32))
        energy.append(chunk_energy_db(c))
    return np.stack(feats), np.array(energy, np.float32)


def window_index(n, seq_len=SEQ_LEN, stride=STRIDE):
    if n <= seq_len:
        return np.arange(n)[None, :]
    starts = list(range(0, n - seq_len + 1, stride))
    if starts[-1] + seq_len < n:
        starts.append(n - seq_len)
    return np.stack([np.arange(s, s + seq_len) for s in starts])


def eval_views(sid, n_s, n_c, n_views=N_VIEWS, frac=TTA_FRAC):
    """Deterministic test-time views. n_c = number of SELECTED chunks. View 0 = everything kept."""
    rng = np.random.RandomState(zlib.crc32(str(sid).encode()) % (2 ** 31))
    views = [(np.arange(n_s), np.arange(n_c))]
    for _ in range(n_views - 1):
        views.append((np.sort(rng.choice(n_s, max(1, int(round(frac * n_s))), replace=False)),
                      np.sort(rng.choice(n_c, max(1, int(round(frac * n_c))), replace=False))))
    return views


@torch.no_grad()
def embed_sentences(sents):
    out = []
    for i in range(0, len(sents), EMB_BATCH):
        tok = tokenizer(sents[i:i + EMB_BATCH], return_tensors="pt", truncation=True,
                        padding=True, max_length=128).to(device)
        out.append(text_encoder(tok["input_ids"], tok["attention_mask"]).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_audio_layers(chunks):
    """Per-chunk, per-layer pooled WavLM states (T, 13, 768) fp16 + speaker layer means (13, 768).
    Offline we embed ALL chunks once (needed to train the selector); on the device only K are embedded."""
    out = []
    for i in range(0, len(chunks), AUDIO_BATCH):
        x = torch.from_numpy(chunks[i:i + AUDIO_BATCH]).to(device)
        out.append(audio_encoder.hidden_states(x).mean(2).permute(1, 0, 2).half().cpu())
    L = torch.cat(out)
    return L, L.float().mean(0)


# ============================================================
# CACHE
# ============================================================
def find_transcript(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith("_Transcript.csv"):
                return os.path.join(root, f)
    return None

'''
def build_split(split):
    path = os.path.join(CACHE_DIR, f"{split}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    cache = {}
    tdir, adir = os.path.join(TEXT_PATH, split), os.path.join(AUDIO_PATH, split)
    for cat in sorted(os.listdir(tdir)):
        cp = os.path.join(tdir, cat)
        if not os.path.isdir(cp):
            continue
        label = int(cat[0])
        ap = os.path.join(adir, cat)
        wavs = [f for f in os.listdir(ap) if f.endswith(".wav")] if os.path.isdir(ap) else []
        for subj in tqdm(sorted(os.listdir(cp)), desc=f"{split}/{cat}"):
            sid = subj.split("_")[0]
            tcsv = find_transcript(os.path.join(cp, subj))
            if tcsv is None:
                continue
            sents, segs = load_transcript(tcsv)
            if not sents:
                continue
            wav = next((os.path.join(ap, f) for f in wavs if f.split("_")[0] == sid), None)
            if wav is None:
                continue
            chunks, times, y_use = load_participant_audio(wav, segs)
            if len(chunks) == 0:
                continue
            if sid in cache:
                raise RuntimeError(f"Speaker {sid} appears twice in split {split}")
            layers, layer_means = embed_audio_layers(chunks)
            cheap, energy = chunk_features(chunks)
            cache[sid] = dict(
                sid=sid, label=label, split=split, sents=sents, text=embed_sentences(sents),
                audio_layers=layers, layer_means=layer_means, times=times, tcsv=tcsv, wav=wav,
                gm=speaker_egemaps(y_use) if USE_EGEMAPS else None,
                cheap=torch.from_numpy(cheap), energy=energy)
    torch.save(cache, path)
    return cache
'''
import glob
def build_split(split):
    hits = glob.glob(f"/home/hp2/Documents/kaggle/working/outputs_v13_15sec_segments/cache_base/embeddings/{split}.pt", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{split}.pt not found")
    print("Loading", hits[0])
    return torch.load(hits[0], weights_only=False)

print("Building / loading cache:", CACHE_DIR)
caches = {s: build_split(s) for s in ["Train", "Val", "Test"]}
text_encoder.cpu(); audio_encoder.cpu()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

ids = {k: set(v) for k, v in caches.items()}
for a, b in [("Train", "Val"), ("Train", "Test"), ("Val", "Test")]:
    overlap = ids[a] & ids[b]
    assert not overlap, f"Speaker leakage between {a} and {b}: {sorted(overlap)}"
print("Speaker-disjoint splits verified:", {k: len(v) for k, v in ids.items()})
pd.DataFrame([(sid, s, caches[s][sid]["label"]) for s in caches for sid in caches[s]],
             columns=["speaker_id", "split", "label"]).to_csv(os.path.join(OUT_DIR, "splits.csv"), index=False)

dev_entries = [caches[s][k] for s in ("Train", "Val") for k in sorted(caches[s])]
test_entries = [caches["Test"][k] for k in sorted(caches["Test"])]
ydev = np.array([e["label"] for e in dev_entries]); yte = np.array([e["label"] for e in test_entries])
assert len({e["sid"] for e in dev_entries}) == len(dev_entries)
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=CV_SEED).split(np.zeros(len(ydev)), ydev))
D_CHEAP = dev_entries[0]["cheap"].shape[1]
print(f"Dev (Train+Val): n={len(ydev)} dep={ydev.sum()} | Test: n={len(yte)} dep={yte.sum()} | {N_FOLDS}-fold CV")
print(f"Chunks per speaker (dev): median {np.median([len(e['times']) for e in dev_entries]):.0f} | "
      f"cheap features per chunk: {D_CHEAP}")


# ============================================================
# WAVLM LAYER SELECTION (CV linear probe on the dev pool only)
# ============================================================
def cv_probe_auc(X, y):
    oof = np.zeros(len(y))
    for tr, va in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); oof[va] = clf.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof)


Xlay = np.stack([e["layer_means"].numpy() for e in dev_entries])
layer_auc = np.array([cv_probe_auc(Xlay[:, l], ydev) for l in range(Xlay.shape[1])])
if AUTO_LAYERS:
    cands = range(1, Xlay.shape[1] - LAYER_WIN + 1)
    l0 = max(cands, key=lambda s: layer_auc[s:s + LAYER_WIN].mean())
    WAVLM_LAYERS = (l0, l0 + LAYER_WIN)
else:
    WAVLM_LAYERS = DEFAULT_LAYERS
audio_encoder.l0, audio_encoder.l1 = WAVLM_LAYERS
print("Layer probe CV AUC:", np.round(layer_auc, 3), "-> using hidden_states", WAVLM_LAYERS)
plt.figure(figsize=(5, 3))
plt.plot(range(len(layer_auc)), layer_auc, marker="o")
plt.axvspan(WAVLM_LAYERS[0] - 0.5, WAVLM_LAYERS[1] - 0.5, alpha=0.15, color="tab:orange", label="selected")
plt.xlabel("WavLM hidden state (0 = CNN features)"); plt.ylabel("CV AUC (linear probe)")
plt.legend(); plt.title("Layer-wise depression information"); savefig("wavlm_layer_probe.png")

for e in dev_entries + test_entries:
    e["audio"] = e["audio_layers"][:, WAVLM_LAYERS[0]:WAVLM_LAYERS[1]].float().mean(1)   # (T, 768)


def to_sample(e):
    return dict(sid=e["sid"], y=e["label"], y_t=torch.tensor([e["label"]], device=device),
                text=e["text"].to(device), audio=e["audio"].to(device),
                cheap=e["cheap"].float().to(device), energy=np.asarray(e["energy"]), times=e["times"],
                gm=torch.from_numpy(e["gm"]).float().unsqueeze(0).to(device) if USE_EGEMAPS else None)


dev_s, test_s = [to_sample(e) for e in dev_entries], [to_sample(e) for e in test_entries]
GM_BASE = torch.cat([s["gm"] for s in dev_s]).mean(0, keepdim=True) if USE_EGEMAPS else None

# ============================================================
# MODEL
# ============================================================
class AttnPool(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return (x * w.unsqueeze(-1)).sum(1), w


class ChunkScorer(nn.Module):
    """[S2] Tiny MLP giving one relevance score per chunk.
    'learned'       : input = cheap per-chunk features -> runs on the device BEFORE WavLM.
    'learned_wavlm' : input = WavLM chunk embeddings  -> upper-bound baseline, no compute saving."""
    def __init__(self, d_in, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1).clone())
        self.register_buffer("std", std.view(1, -1).clone())
        self.net = nn.Sequential(nn.Linear(d_in, SCORER_HIDDEN), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(SCORER_HIDDEN, 1))

    def forward(self, x):                                   # (T, d_in) -> (T,)
        return self.net(((x - self.mean) / self.std).clamp(-5, 5)).squeeze(-1)


class DepressionNet(nn.Module):
    """Players: 'text', 'wavlm', 'egemaps'. The audio input of forward() is the SELECTED chunks.
    text (W, L, D_TEXT) windows · audio (1, K, 768) selected chunks · gm (1, N_GM) speaker eGeMAPS."""
    def __init__(self, mode, gm_mean=None, gm_std=None, selector="all", keep_frac=1.0, sc_mean=None, sc_std=None):
        super().__init__()
        self.mode, self.players = mode, []
        self.selector = selector if mode != "text" else "all"
        self.keep_frac = float(keep_frac) if mode != "text" else 1.0
        self.scorer = None
        H, dims = 2 * HIDDEN, {}
        if mode in ("text", "fusion"):
            self.players.append("text"); dims["text"] = H
            self.text_norm, self.text_drop = nn.LayerNorm(D_TEXT), nn.Dropout(IN_DROPOUT)
            self.text_lstm = nn.LSTM(D_TEXT, HIDDEN, batch_first=True, bidirectional=True)
            self.sent_attn, self.win_attn = AttnPool(H), AttnPool(H)
        if mode in ("audio", "fusion"):
            self.players.append("wavlm"); dims["wavlm"] = H
            self.audio_norm, self.audio_drop = nn.LayerNorm(768), nn.Dropout(IN_DROPOUT)
            self.audio_lstm = nn.LSTM(768, HIDDEN, batch_first=True, bidirectional=True)
            self.audio_attn = AttnPool(H)
            if self.selector in LEARNED:
                self.scorer = ChunkScorer(sc_mean.numel(), sc_mean, sc_std)
            if N_GM > 0:
                self.players.append("egemaps"); dims["egemaps"] = GM_DIM
                self.register_buffer("gm_mean", gm_mean.view(1, -1).clone())
                self.register_buffer("gm_std", gm_std.view(1, -1).clone())
                self.gm_mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(N_GM, GM_DIM), nn.GELU())
        d_f = sum(dims[p] for p in self.players)
        self.head = nn.Sequential(nn.LayerNorm(d_f), nn.Dropout(HEAD_DROPOUT), nn.Linear(d_f, 64),
                                  nn.GELU(), nn.Dropout(HEAD_DROPOUT), nn.Linear(64, 2))

    def chunk_scores(self, s, cand):
        x = s["cheap"] if self.selector == "learned" else s["audio"]
        return self.scorer(x[torch.as_tensor(cand, device=x.device)])

    def encode_text(self, text):
        h, _ = self.text_lstm(self.text_drop(self.text_norm(text)))
        win, w_sent = self.sent_attn(h)
        t, w_win = self.win_attn(win.unsqueeze(0))
        return t, w_sent, w_win.squeeze(0)

    def encode_audio(self, audio, st=None):
        x = self.audio_drop(self.audio_norm(audio))
        if st is not None:
            # Straight-through weights are applied AFTER LayerNorm (LayerNorm is scale-invariant,
            # so a weight applied before it would receive zero gradient).
            x = x * st.view(1, -1, 1)
        h, _ = self.audio_lstm(x)
        a, w = self.audio_attn(h)
        return a, w.squeeze(0)

    def encode_gm(self, gm):
        return self.gm_mlp(((gm - self.gm_mean) / self.gm_std).clamp(-5, 5))

    def player_feats(self, text, audio, gm=None, st=None):
        f, att = {}, dict(w_sent=None, w_win=None, w_audio=None)
        if "text" in self.players:
            f["text"], att["w_sent"], att["w_win"] = self.encode_text(text)
        if "wavlm" in self.players:
            f["wavlm"], att["w_audio"] = self.encode_audio(audio, st)
        if "egemaps" in self.players:
            f["egemaps"] = self.encode_gm(gm)
        return f, att

    def classify(self, f):
        return self.head(torch.cat([f[p] for p in self.players], -1))

    def forward(self, text, audio, gm=None, return_attn=False, st=None):
        f, att = self.player_feats(text, audio, gm, st)
        logits = self.classify(f)
        return (logits, att) if return_attn else logits


def zscore(x, dim=-1):
    mu = x.mean(dim, keepdim=True)
    return (x - mu) / ((x - mu).pow(2).mean(dim, keepdim=True).sqrt() + 1e-6)


class Ensemble(nn.Module):
    """Averages member logits and attention maps. Selection is SHARED: mean of per-member z-scored chunk
    scores, so one set of K chunks serves every member (WavLM runs K times per session, not K x members)."""
    def __init__(self, members):
        super().__init__(); self.members = nn.ModuleList(members)
        self.selector, self.keep_frac = members[0].selector, members[0].keep_frac

    def chunk_scores(self, s, cand):
        return zscore(torch.stack([m.chunk_scores(s, cand) for m in self.members])).mean(0)

    def forward(self, text, audio, gm=None, return_attn=False, st=None):
        outs = [m(text, audio, gm, return_attn=True) for m in self.members]
        logits = torch.stack([o[0] for o in outs]).mean(0)
        if not return_attn:
            return logits
        att = {k: (torch.stack([o[1][k] for o in outs]).mean(0) if outs[0][1][k] is not None else None)
               for k in outs[0][1]}
        return logits, att


def logodds(logits):
    return logits[..., 1] - logits[..., 0]


# ============================================================
# [S3] TOP-K SELECTION
# ============================================================
def n_keep(n, frac):
    return int(n) if frac >= 1.0 else int(min(n, max(K_MIN, np.ceil(frac * n - 1e-9))))


def _sid_rng(sid):
    return np.random.RandomState((zlib.crc32(str(sid).encode()) + 1) % (2 ** 31))


@torch.no_grad()
def pick(model, s, cand=None):
    """Hard Top-K used at inference (and on the device). Returns chunk indices in temporal order."""
    cand = np.arange(len(s["audio"])) if cand is None else np.sort(np.asarray(cand))
    k = n_keep(len(cand), model.keep_frac)
    if model.selector == "all" or k >= len(cand):
        return cand
    if model.selector == "first":
        return cand[:k]
    if model.selector == "random":
        return np.sort(_sid_rng(s["sid"]).choice(cand, k, replace=False))
    if model.selector == "energy":
        sc = s["energy"][cand]
    else:
        sc = model.chunk_scores(s, cand).float().cpu().numpy()
    return np.sort(cand[np.argsort(-sc, kind="stable")[:k]])


def relaxed_topk(keys, k, tau):
    """Continuous relaxation of Top-K (Xie & Ermon, 2019): k successive softmaxes; returns a soft k-hot vector."""
    khot, r = torch.zeros_like(keys), keys
    for _ in range(k):
        a = torch.softmax(r / tau, dim=-1)
        khot = khot + a
        r = r + torch.log((1.0 - a).clamp(min=1e-10))
    return khot


def train_select(model, s, cand, tau, rng):
    """Training-time selection -> (selected indices, straight-through weights or None, auxiliary loss).
    Forward pass: hard Top-K of Gumbel-perturbed scores (the LSTM only sees K chunks, as at inference).
    Backward pass: gradient of the relaxed Top-K weights reaches the scorer."""
    k = n_keep(len(cand), model.keep_frac)
    if model.selector == "all" or k >= len(cand):
        return cand, None, 0.0
    if model.selector == "first":
        return cand[:k], None, 0.0
    if model.selector == "random":
        return np.sort(rng.choice(cand, k, replace=False)), None, 0.0
    if model.selector == "energy":
        return np.sort(cand[np.argsort(-s["energy"][cand], kind="stable")[:k]]), None, 0.0
    scores = model.chunk_scores(s, cand)
    u = torch.rand_like(scores).clamp(1e-6, 1 - 1e-6)
    keys = scores + GUMBEL_SCALE * (-torch.log(-torch.log(u)))
    soft = relaxed_topk(keys, k, tau)
    top = torch.topk(keys, k).indices.sort().values
    w = soft[top]
    st = 1.0 + w - w.detach()                               # = 1 in the forward pass
    aux = 0.0
    if SEL_DISTILL > 0:
        with torch.no_grad():
            _, w_full = model.encode_audio(s["audio"][torch.as_tensor(cand, device=device)].unsqueeze(0))
        aux = SEL_DISTILL * (w_full * (w_full.clamp(min=1e-8).log() - torch.log_softmax(scores, -1))).sum()
    return cand[top.cpu().numpy()], st, aux


def head_forward(model, s, sel, keep_s=None, keep_c=None, return_attn=False, st=None):
    """Runs the classifier on the selected chunks `sel`; keep_c indexes INTO sel (TTA / deletion)."""
    text = s["text"] if keep_s is None else s["text"][torch.as_tensor(np.sort(keep_s), device=device)]
    c = np.asarray(sel) if keep_c is None else np.asarray(sel)[np.sort(keep_c)]
    audio = s["audio"][torch.as_tensor(c, device=device)]
    idx = torch.from_numpy(window_index(text.shape[0])).to(device)
    return model(text[idx], audio.unsqueeze(0), s["gm"], return_attn=return_attn, st=st)


def full_inputs(s, sel):
    idx = torch.from_numpy(window_index(s["text"].shape[0])).to(device)
    return s["text"][idx], s["audio"][torch.as_tensor(np.asarray(sel), device=device)].unsqueeze(0), s["gm"]


@torch.no_grad()
def predict_logodds(model, samples, tta=True):
    model.eval(); out = []
    for s in samples:
        sel = pick(model, s)
        views = eval_views(s["sid"], len(s["text"]), len(sel)) if tta else [(None, None)]
        out.append(np.mean([logodds(head_forward(model, s, sel, ks, kc))[0].item() for ks, kc in views]))
    return np.array(out)


# ============================================================
# TRAIN / EVAL UTILITIES
# ============================================================
def best_threshold(y, p, metric=THR_METRIC):
    grid = np.arange(0.05, 0.96, 0.01)
    avg = "macro" if metric == "f1_macro" else "binary"
    scores = [f1_score(y, (p >= t).astype(int), average=avg, zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def compute_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(AUC=roc_auc_score(y, p),
                F1_macro=f1_score(y, yhat, average="macro", zero_division=0),
                F1_pos=f1_score(y, yhat, zero_division=0),
                MCC=matthews_corrcoef(y, yhat), Acc=accuracy_score(y, yhat),
                BalAcc=balanced_accuracy_score(y, yhat),
                Precision=precision_score(y, yhat, zero_division=0),
                Recall=recall_score(y, yhat, zero_division=0),
                Specificity=recall_score(y, yhat, pos_label=0, zero_division=0))


METRICS = ["AUC", "F1_macro", "F1_pos", "MCC", "Acc", "BalAcc", "Precision", "Recall", "Specificity"]


def bootstrap_ci(y, p, thr, n=N_BOOT, seed=0):
    rng, rows = np.random.RandomState(seed), []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        rows.append(compute_metrics(y[i], p[i], thr))
    return pd.DataFrame(rows).quantile([0.025, 0.975])


def random_subset(n, rng):
    k = max(1, int(round(n * rng.uniform(*AUG_FRAC))))
    return np.sort(rng.choice(n, k, replace=False))


def train_member(mode, selector, frac, seed, fold, tr_idx, va_idx, verbose=True):
    set_seed(seed * 100 + fold)
    tr, va = [dev_s[i] for i in tr_idx], [dev_s[i] for i in va_idx]
    y_tr, y_va = ydev[tr_idx], ydev[va_idx]
    gm_mean = gm_std = None
    if N_GM > 0:
        G = torch.cat([s["gm"] for s in tr])
        gm_mean, gm_std = G.mean(0), G.std(0).clamp(min=1e-6)
    sc_mean = sc_std = None
    if mode != "text" and selector in LEARNED:            # scorer normalisation: training fold only
        X = torch.cat([s["cheap" if selector == "learned" else "audio"] for s in tr])
        sc_mean, sc_std = X.mean(0), X.std(0).clamp(min=1e-6)
    model = DepressionNet(mode, gm_mean, gm_std, selector, frac, sc_mean, sc_std).to(device)
    cw = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr),
                      dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    rng = np.random.RandomState(seed * 100 + fold)
    best, bad, hist = dict(auc=-1.0), 0, dict(train_loss=[], val_auc=[])
    for ep in range(EPOCHS):
        tau = TAU_START * (TAU_END / TAU_START) ** (ep / max(1, EPOCHS - 1))
        model.train(); opt.zero_grad()
        tot = torch.zeros((), device=device)
        order = rng.permutation(len(tr))
        for step, i in enumerate(order, 1):
            s = tr[i]
            ks = random_subset(len(s["text"]), rng) if AUGMENT else None
            cand = random_subset(len(s["audio"]), rng) if AUGMENT else np.arange(len(s["audio"]))
            sel, st, aux = train_select(model, s, cand, tau, rng)
            loss = crit(head_forward(model, s, sel, ks, st=st), s["y_t"]) + aux
            (loss / ACCUM).backward(); tot += loss.detach()
            if step % ACCUM == 0 or step == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        sched.step()
        v_auc = roc_auc_score(y_va, predict_logodds(model, va, tta=False))
        hist["train_loss"].append(tot.item() / len(tr)); hist["val_auc"].append(v_auc)
        if v_auc > best["auc"]:
            best = dict(auc=v_auc, epoch=ep + 1, state={k: v.detach().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if ep + 1 >= MIN_EPOCHS and bad >= PATIENCE:
            break
    model.load_state_dict(best["state"]); model.eval()
    if verbose:
        print(f"  [{mode:6s} {model.selector:13s} keep {model.keep_frac:.0%} seed {seed} fold {fold}] "
              f"best epoch {best['epoch']:02d}  fold AUC {best['auc']:.3f}  final train loss {hist['train_loss'][-1]:.3f}")
    return model, dict(val_auc=best["auc"], epoch=best["epoch"], hist=hist, tr_idx=tr_idx)


def run_config(name, mode, selector, frac, seeds, keep_members=True, verbose=True):
    """Trains len(seeds) x N_FOLDS members with one protocol. Test predictions use the ensemble
    (shared selection), thresholds come from out-of-fold dev predictions."""
    print(f"\n=== {name}: mode={mode} selector={selector} keep={frac:.0%} ===")
    oof_lo = np.zeros((len(seeds), len(dev_s)))
    members, infos, rows = [], [], []
    for si, seed in enumerate(seeds):
        seed_members = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            model, info = train_member(mode, selector, frac, seed, fi, tr_idx, va_idx, verbose)
            oof_lo[si, va_idx] = predict_logodds(model, [dev_s[i] for i in va_idx])
            seed_members.append(model); infos.append(info)
        thr_s, _ = best_threshold(ydev, sigmoid(oof_lo[si]))
        p_s = sigmoid(predict_logodds(Ensemble(seed_members).eval(), test_s))
        rows.append(dict(name=name, mode=mode, selector=selector, keep_frac=frac, seed=seed, thr=thr_s,
                         cv_auc=roc_auc_score(ydev, oof_lo[si]), **compute_metrics(yte, p_s, thr_s)))
        members += seed_members
    E = Ensemble(members).eval()
    oof_p, test_p = sigmoid(oof_lo.mean(0)), sigmoid(predict_logodds(E, test_s))
    thr, _ = best_threshold(ydev, oof_p)
    out = dict(name=name, mode=mode, selector=E.selector, frac=E.keep_frac, infos=infos, thr=thr,
               oof_p=oof_p, test_p=test_p, cv=compute_metrics(ydev, oof_p, thr),
               test=compute_metrics(yte, test_p, thr), sel={s["sid"]: pick(E, s) for s in test_s})
    print(f"{name} ensemble: CV AUC {out['cv']['AUC']:.3f} | TEST AUC {out['test']['AUC']:.3f} "
          f"F1-macro {out['test']['F1_macro']:.3f} Acc {out['test']['Acc']:.3f} (thr {thr:.2f})")
    if keep_members:
        out["members"] = members
    else:
        del members, E, seed_members
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out, rows


# ============================================================
# MAIN EXPERIMENTS
# ============================================================
rows, ens = [], {}
for name, mode, selector, frac in MAIN_CONFIGS:
    ens[name], r = run_config(name, mode, selector, frac, SEEDS)
    rows += r


# ============================================================
# LOGISTIC-REGRESSION BASELINES (all chunks, same folds, same threshold protocol)
# ============================================================
def feat_matrix(entries, kind):
    if kind == "text":
        return np.stack([e["text"].mean(0).numpy() for e in entries])
    if kind == "wavlm":
        return np.stack([e["audio"].mean(0).numpy() for e in entries])
    return np.stack([e["gm"] for e in entries])


def lr_baseline(kinds):
    Xd = np.hstack([feat_matrix(dev_entries, k) for k in kinds])
    Xt = np.hstack([feat_matrix(test_entries, k) for k in kinds])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    oof = np.zeros(len(ydev))
    for tr, va in folds:
        oof[va] = mk().fit(Xd[tr], ydev[tr]).predict_proba(Xd[va])[:, 1]
    thr, _ = best_threshold(ydev, oof)
    p = mk().fit(Xd, ydev).predict_proba(Xt)[:, 1]
    return dict(cv_auc=roc_auc_score(ydev, oof), thr=thr, p=p, **compute_metrics(yte, p, thr))


baseline_sets = {"LR-text": ["text"], "LR-wavlm": ["wavlm"], "LR-fusion": ["text", "wavlm"]}
if USE_EGEMAPS:
    baseline_sets.update({"LR-eGeMAPS": ["gm"], "LR-fusion+eGeMAPS": ["text", "wavlm", "gm"]})
baselines = {name: lr_baseline(k) for name, k in baseline_sets.items()}

# ---- Results tables ----
res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT_DIR, "results_per_seed.csv"), index=False)
fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
table = []
for name, *_ in MAIN_CONFIGS:
    r, E_ = res[res["name"] == name], ens[name]
    tag = f"{name} [{E_['selector']}, keep {E_['frac']:.0%}]"
    table.append(dict(model=f"{tag} per-seed", CV_AUC=fmt(r.cv_auc.mean(), r.cv_auc.std()),
                      **{m: fmt(r[m].mean(), r[m].std()) for m in METRICS}))
    table.append(dict(model=f"{tag} ensemble {len(SEEDS)}x{N_FOLDS}", CV_AUC=f"{E_['cv']['AUC']:.3f}",
                      **{m: f"{E_['test'][m]:.3f}" for m in METRICS}))
for name, b in baselines.items():
    table.append(dict(model=name, CV_AUC=f"{b['cv_auc']:.3f}", **{m: f"{b[m]:.3f}" for m in METRICS}))
table = pd.DataFrame(table).set_index("model")
print("\nTEST RESULTS (threshold from dev CV, macro-F1)\n", table.to_string())
table.to_csv(os.path.join(OUT_DIR, "results_summary.csv"))

# ---- Selected system = fusion ensemble with Top-K selection ----
F = ens["fusion"]; THR, p_test, SEL_TEST = F["thr"], F["test_p"], F["sel"]
yhat = (p_test >= THR).astype(int)
print(f"\nFUSION ENSEMBLE ({len(F['members'])} models, {F['selector']} keep {F['frac']:.0%}, threshold {THR:.2f})")
print(classification_report(yte, yhat, digits=4))
ci = bootstrap_ci(yte, p_test, THR)
ci_table = pd.DataFrame({m: [F["test"][m], ci.loc[0.025, m], ci.loc[0.975, m]] for m in METRICS},
                        index=["point", "ci_low", "ci_high"]).T
print(ci_table.round(4)); ci_table.to_csv(os.path.join(OUT_DIR, "fusion_bootstrap_ci.csv"))

if METADATA_CSV:
    meta = pd.read_csv(METADATA_CSV)
    gmap = dict(zip(meta["Participant_ID"].astype(str), meta["Gender"]))
    groups = np.array([gmap.get(s["sid"], "unknown") for s in test_s])
    sub = [dict(group=g, n=int((groups == g).sum()), **compute_metrics(yte[groups == g], p_test[groups == g], THR))
           for g in np.unique(groups) if len(np.unique(yte[groups == g])) == 2]
    print("\nSubgroup performance\n", pd.DataFrame(sub).round(4).to_string(index=False))

# ---- Standard plots ----
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for info in F["infos"]:
    ax[0].plot(info["hist"]["train_loss"], color="tab:blue", alpha=0.3)
    ax[1].plot(info["hist"]["val_auc"], color="tab:orange", alpha=0.3)
ax[0].set_title("Train loss (fusion members)"); ax[1].set_title("Held-out fold AUC (fusion members)")
for a in ax: a.set_xlabel("Epoch")
savefig("training_curves.png")

plt.figure(figsize=(4, 3.5))
sns.heatmap(confusion_matrix(yte, yhat), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-dep", "Dep"], yticklabels=["Non-dep", "Dep"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Fusion Top-K ensemble (test)")
savefig("confusion_matrix.png")

plt.figure(figsize=(4.8, 4.2))
for name in ens:
    fpr, tpr, _ = roc_curve(yte, ens[name]["test_p"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={ens[name]['test']['AUC']:.3f})")
bl = "LR-fusion+eGeMAPS" if USE_EGEMAPS else "LR-fusion"
fpr, tpr, _ = roc_curve(yte, baselines[bl]["p"])
plt.plot(fpr, tpr, "--", label=f"{bl} (AUC={baselines[bl]['AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k:", lw=0.8)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=7); plt.title("ROC (test)")
savefig("roc_modalities.png")


# ============================================================
# [S5] UPLINK / COMPUTE TABLE  (depends only on K, i.e. on the keep fraction)
# ============================================================
def uplink_table(fracs):
    T = np.array([len(s["audio"]) for s in test_s]); n_s = np.array([len(s["text"]) for s in test_s])
    out = []
    for f in fracs:
        K = np.array([n_keep(t, f) for t in T])
        out.append(dict(keep_frac=f, chunks_total=T.mean(), chunks_kept=K.mean(),
                        raw_audio_all_KB=(T * CHUNK_SIZE * 2).mean() / 1024,        # 16-bit PCM
                        raw_audio_topk_KB=(K * CHUNK_SIZE * 2).mean() / 1024,
                        wavlm_emb_all_KB=(T * 768 * 2).mean() / 1024,               # fp16 embeddings
                        wavlm_emb_topk_KB=(K * 768 * 2).mean() / 1024,
                        text_emb_KB=(n_s * D_TEXT * 2).mean() / 1024,
                        egemaps_KB=N_GM * 4 / 1024,
                        wavlm_calls_saved_pct=100 * (1 - K.sum() / T.sum())))
    return pd.DataFrame(out)


up = uplink_table(sorted(set(SWEEP_FRACS + [KEEP_FRAC, 1.0])))
up["raw_reduction_x"] = up.raw_audio_all_KB / up.raw_audio_topk_KB
up["topk_emb_vs_all_raw_x"] = up.raw_audio_all_KB / up.wavlm_emb_topk_KB
print("\nPer-session uplink (test speakers, participant speech only):\n", up.round(2).to_string(index=False))
up.to_csv(os.path.join(OUT_DIR, "uplink_per_session.csv"), index=False)

# ============================================================
# [S5] K-SWEEP: accuracy vs fraction of audio kept, per selector
# ============================================================
if RUN_SWEEP:
    sweep = []
    configs = [(sk, f) for f in SWEEP_FRACS for sk in SWEEP_SELECTORS] + [("all", 1.0)]
    for sk, f in configs:
        E_, _ = run_config(f"sweep_{sk}_{f}", "fusion", sk, f, SWEEP_SEEDS, keep_members=False, verbose=False)
        sweep.append(dict(selector=sk, keep_frac=f, cv_auc=E_["cv"]["AUC"], cv_F1_macro=E_["cv"]["F1_macro"],
                          **{f"test_{m}": E_["test"][m] for m in METRICS}))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(os.path.join(OUT_DIR, "topk_sweep.csv"), index=False)
    print("\nK-sweep (fusion):\n", sweep.round(3).to_string(index=False))
    ref = sweep[sweep.selector == "all"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for sk, g in sweep[sweep.selector != "all"].groupby("selector"):
        g = g.sort_values("keep_frac")
        ls = "-" if sk == "learned" else "--"
        ax[0].plot(g.keep_frac * 100, g.cv_auc, ls, marker="o", label=sk)
        ax[1].plot(g.keep_frac * 100, g.test_F1_macro, ls, marker="o", label=sk)
    ax[0].axhline(ref.cv_auc, c="k", lw=0.8, ls=":", label="all chunks")
    ax[1].axhline(ref.test_F1_macro, c="k", lw=0.8, ls=":", label="all chunks")
    ax[0].set_ylabel("Dev CV AUC"); ax[1].set_ylabel("Test macro-F1")
    for a in ax:
        a.set_xlabel("% of audio chunks kept (K / T)")
    ax[0].legend(fontsize=7); savefig("topk_sweep.png")

# ============================================================
# INTERPRETABILITY (fusion Top-K ensemble, on the selected chunks = view 0)
# ============================================================
M = Ensemble(F["members"]).eval()
PLAYERS = F["members"][0].players
interp = dict(wavlm_layers=list(WAVLM_LAYERS), layer_probe_cv_auc=[round(float(a), 4) for a in layer_auc],
              selector=F["selector"], keep_frac=F["frac"])


# ---------- 1. Exact Shapley values over input branches ----------
def exact_shapley(v, players):
    n, phi = len(players), {}
    for p in players:
        others, val = [q for q in players if q != p], 0.0
        for r in range(n):
            for S in combinations(others, r):
                w = factorial(r) * factorial(n - r - 1) / factorial(n)
                val += w * (v(frozenset(S) | {p}) - v(frozenset(S)))
        phi[p] = val
    return phi


@torch.no_grad()
def member_bases(member, tr_idx):
    feats = [member.player_feats(*full_inputs(dev_s[i], pick(member, dev_s[i])))[0] for i in tr_idx]
    return {p: torch.cat([f[p] for f in feats]).mean(0, keepdim=True) for p in member.players}


bases = [member_bases(m, info["tr_idx"]) for m, info in zip(F["members"], F["infos"])]


@torch.no_grad()
def speaker_shapley(s):
    acc = {p: 0.0 for p in PLAYERS}
    for m, base in zip(F["members"], bases):
        f = m.player_feats(*full_inputs(s, SEL_TEST[s["sid"]]))[0]
        v = lambda S: logodds(m.classify({p: (f[p] if p in S else base[p]) for p in m.players}))[0].item()
        for p, val in exact_shapley(v, m.players).items():
            acc[p] += val / len(F["members"])
    return acc


shap = pd.DataFrame([dict(sid=s["sid"], y=s["y"], p=p, **{f"phi_{k}": v for k, v in speaker_shapley(s).items()})
                     for s, p in zip(test_s, p_test)])
absphi = shap[[f"phi_{p}" for p in PLAYERS]].abs()
share = absphi.div(absphi.sum(1) + 1e-9, axis=0)
share.columns = PLAYERS
shap.join(share.add_prefix("share_")).to_csv(os.path.join(OUT_DIR, "modality_shapley.csv"), index=False)
interp["shapley_mean_share"] = share.mean().round(4).to_dict()
print("\nMean share of evidence per branch:", interp["shapley_mean_share"])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw=dict(width_ratios=[2, 1]))
long = shap.melt(id_vars="y", value_vars=[f"phi_{p}" for p in PLAYERS], var_name="branch", value_name="phi")
long["branch"] = long.branch.str.replace("phi_", ""); long["class"] = long.y.map({0: "Non-dep", 1: "Dep"})
sns.boxplot(data=long, x="branch", y="phi", hue="class", ax=ax[0])
ax[0].axhline(0, c="k", lw=0.6); ax[0].set_ylabel("Shapley value (log-odds of depression)")
share.mean().plot.bar(ax=ax[1], color="tab:gray"); ax[1].set_ylabel("Mean share of |evidence|")
savefig("modality_shapley.png")


# ---------- 2. Integrated Gradients (sentences, selected audio chunks, eGeMAPS) ----------
def integrated_gradients(model, s, sel, steps=IG_STEPS):
    model.eval()
    win, audio, gm = full_inputs(s, sel)
    xs = [win, audio] + ([gm] if gm is not None else [])
    b = [torch.zeros_like(win), torch.zeros_like(audio)] + ([GM_BASE] if gm is not None else [])
    grads = [torch.zeros_like(x) for x in xs]
    with torch.backends.cudnn.flags(enabled=False):
        for k in range(1, steps + 1):
            cur = [(bb + (k / steps) * (x - bb)).requires_grad_(True) for x, bb in zip(xs, b)]
            out = model(cur[0], cur[1], cur[2] if len(cur) > 2 else None)
            gs = torch.autograd.grad(logodds(out)[0], cur, allow_unused=True)
            for g_acc, g in zip(grads, gs):
                if g is not None:
                    g_acc += g
        with torch.no_grad():
            f_x = logodds(model(*xs[:2], xs[2] if len(xs) > 2 else None))[0].item()
            f_0 = logodds(model(*b[:2], b[2] if len(b) > 2 else None))[0].item()
    attrs = [((x - bb) * g / steps).detach() for x, bb, g in zip(xs, b, grads)]
    total = sum(a.sum().item() for a in attrs)
    return dict(win=attrs[0].sum(-1).cpu(), chunk=attrs[1].sum(-1)[0].cpu().numpy(),
                gm=attrs[2][0].cpu().numpy() if len(attrs) > 2 else None,
                f_x=f_x, delta=total - (f_x - f_0))


def per_sentence(values, n_s):
    idx = torch.from_numpy(window_index(n_s))
    return torch.zeros(n_s).index_add_(0, idx.flatten(), values.flatten().float().cpu()).numpy()


print("\nIntegrated Gradients on the fusion ensemble (selected chunks)")
ig = []
for s in tqdm(test_s):
    sel = SEL_TEST[s["sid"]]
    r = integrated_gradients(M, s, sel)
    with torch.no_grad():
        _, att = M(*full_inputs(s, sel), return_attn=True)
    n_s = len(s["text"])
    ig.append(dict(sid=s["sid"], y=s["y"], f_x=r["f_x"], delta=r["delta"],
                   sent_attr=per_sentence(r["win"], n_s), chunk_attr=r["chunk"], gm_attr=r["gm"],
                   att_sent=per_sentence(att["w_win"].unsqueeze(1) * att["w_sent"], n_s),
                   att_audio=att["w_audio"].cpu().numpy()))

rel_delta = np.median([abs(r["delta"]) / (abs(r["f_x"]) + 1e-6) for r in ig])
rho_t = [stats.spearmanr(r["sent_attr"], r["att_sent"]).correlation for r in ig if len(r["sent_attr"]) > 3]
rho_a = [stats.spearmanr(np.abs(r["chunk_attr"]), r["att_audio"]).correlation for r in ig if len(r["chunk_attr"]) > 3]
print(f"IG completeness: median |delta|/|f(x)| = {rel_delta:.4f} (raise IG_STEPS if > 0.05)")
print(f"Attention vs IG (Spearman): text {np.nanmedian(rho_t):.3f}, audio {np.nanmedian(rho_a):.3f}")
interp["ig"] = dict(completeness_rel_delta=float(rel_delta), attn_ig_spearman_text=float(np.nanmedian(rho_t)),
                    attn_ig_spearman_audio=float(np.nanmedian(rho_a)))

if USE_EGEMAPS:
    A = np.stack([r["gm_attr"] for r in ig]); names = np.array(GM_NAMES)
    imp = np.abs(A).mean(0); top = np.argsort(-imp)[:15][::-1]
    signed = A[yte == 1].mean(0)
    plt.figure(figsize=(6, 4.5))
    plt.barh(names[top], imp[top], color=np.where(signed[top] > 0, "tab:red", "tab:green"))
    plt.xlabel("Mean |IG| (log-odds)"); plt.title("Top eGeMAPS features (red = raises P(dep) in depressed)")
    plt.yticks(fontsize=6); savefig("egemaps_ig_importance.png")
    interp["egemaps_top_features"] = [dict(feature=str(names[i]), mean_abs_ig=float(imp[i]),
                                           mean_signed_ig_dep=float(signed[i])) for i in top[::-1]]


# ---------- 3. Faithfulness: deletion curves (audio deletion is within the K selected chunks) ----------
@torch.no_grad()
def logodds_subset(model, s, keep_s, keep_c):
    return logodds(head_forward(model, s, SEL_TEST[s["sid"]], keep_s, keep_c))[0].item()


def deletion_curve(model, modality, seed=0):
    rng, targeted, random_ = np.random.RandomState(seed), [], []
    for s, r in zip(test_s, ig):
        ns, nc = len(s["text"]), len(SEL_TEST[s["sid"]])
        all_s, all_c = np.arange(ns), np.arange(nc)
        attr, n = (r["sent_attr"], ns) if modality == "text" else (r["chunk_attr"], nc)
        base = logodds_subset(model, s, all_s, all_c)
        order = np.argsort(-attr)
        run = (lambda keep: logodds_subset(model, s, keep, all_c)) if modality == "text" else \
              (lambda keep: logodds_subset(model, s, all_s, keep))
        tg, rd = [], []
        for f in DEL_FRACS:
            k = min(int(round(f * n)), n - 1)
            tg.append(base - run(order[k:]))
            rd.append(np.mean([base - run(rng.permutation(n)[k:]) for _ in range(N_RANDOM_DEL)]))
        targeted.append(tg); random_.append(rd)
    return np.array(targeted), np.array(random_)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.5)); interp["deletion"] = {}
for ax, modality in zip(axes, ["text", "audio"]):
    tg, rd = deletion_curve(M, modality)
    x = np.array(DEL_FRACS) * 100
    for arr, lab in [(tg, "IG-ranked"), (rd, "Random")]:
        mu, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
        ax.plot(x, mu, marker="o", label=lab); ax.fill_between(x, mu - se, mu + se, alpha=0.2)
    aopc_t, aopc_r = tg[:, 1:].mean(1), rd[:, 1:].mean(1)
    try:
        p = stats.wilcoxon(aopc_t, aopc_r).pvalue
    except ValueError:
        p = np.nan
    ax.set_title(f"{modality.capitalize()} deletion (AOPC {aopc_t.mean():.2f} vs {aopc_r.mean():.2f}, p={p:.1e})", fontsize=9)
    ax.set_xlabel(f"% {'sentences' if modality == 'text' else 'selected audio chunks'} removed")
    ax.set_ylabel("Drop in log-odds (depressed)"); ax.legend()
    interp["deletion"][modality] = dict(aopc_ig=float(aopc_t.mean()), aopc_random=float(aopc_r.mean()), wilcoxon_p=float(p))
savefig("deletion_curves.png")

# ---------- 4. Linguistic & PHQ-domain grounding ----------
LEX = {
    "first_person_sg": {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"},
    "absolutist": {"always", "never", "nothing", "everything", "completely", "totally", "entire",
                   "constantly", "definitely", "all", "whole", "every", "anything", "nobody", "everyone"},
    "negative_emotion": {"sad", "tired", "depressed", "lonely", "hate", "worry", "worried", "anxious",
                         "hurt", "bad", "cry", "crying", "hopeless", "stress", "stressed", "afraid",
                         "angry", "upset", "scared", "miserable", "empty", "awful", "terrible"},
}
PHQ_DOMAINS = {
    "anhedonia": {"enjoy", "fun", "interest", "interested", "hobby", "hobbies", "pleasure", "bored", "motivation"},
    "mood": {"sad", "depressed", "down", "hopeless", "unhappy", "cry", "crying", "lonely", "upset", "empty"},
    "sleep": {"sleep", "sleeping", "insomnia", "awake", "nightmare", "nightmares", "bed", "nap"},
    "energy": {"tired", "energy", "exhausted", "fatigue", "lazy", "drained", "worn"},
    "appetite": {"eat", "eating", "appetite", "food", "weight", "hungry"},
    "self_worth": {"failure", "fail", "failed", "guilt", "guilty", "worthless", "blame", "ashamed", "regret", "disappointed"},
    "concentration": {"focus", "concentrate", "concentration", "distracted", "forget", "forgetful", "memory"},
    "psychomotor": {"slow", "restless", "fidget", "agitated", "nervous"},
    "diagnosis_treatment": {"therapy", "therapist", "ptsd", "diagnosed", "medication", "counseling",
                            "psychiatrist", "depression", "meds", "treatment"},   # shortcut check
}
tokenize = lambda s: re.findall(r"[a-z']+", s.lower())
lex_rate = lambda sents, lex: (lambda toks: sum(t in lex for t in toks) / max(len(toks), 1))(
    [t for s in sents for t in tokenize(s)])

lrows, drows, trows = [], [], []
for e, r in zip(test_entries, ig):
    order = np.argsort(-r["sent_attr"])
    k = min(TOPK_SENTS, max(1, len(order) // 3))
    top, rest = [e["sents"][i] for i in order[:k]], [e["sents"][i] for i in order[k:]]
    lrows.append(dict(y=e["label"], **{f"{c}_top": lex_rate(top, L) for c, L in LEX.items()},
                      **{f"{c}_rest": lex_rate(rest, L) for c, L in LEX.items()}))
    drows.append(dict(y=e["label"], **{d: np.mean([bool(set(tokenize(s)) & kw) for s in top]) for d, kw in PHQ_DOMAINS.items()}))
    trows += [dict(sid=e["sid"], y=e["label"], rank=j + 1, ig=float(r["sent_attr"][i]), sentence=e["sents"][i])
              for j, i in enumerate(order[:5])]
ling = pd.DataFrame(lrows); interp["linguistic"] = {}
print("\nLinguistic markers: top-IG sentences vs rest (Wilcoxon)")
for y_, name in [(1, "depressed"), (0, "non-depressed")]:
    sub = ling[ling.y == y_]
    for c in LEX:
        try:
            p = stats.wilcoxon(sub[f"{c}_top"], sub[f"{c}_rest"]).pvalue
        except ValueError:
            p = np.nan
        print(f"  {name:13s} {c:17s} top {sub[f'{c}_top'].mean():.4f} vs rest {sub[f'{c}_rest'].mean():.4f} p={p:.4f}")
        interp["linguistic"][f"{name}_{c}"] = dict(top=float(sub[f"{c}_top"].mean()), rest=float(sub[f"{c}_rest"].mean()), p=float(p))
pd.DataFrame(trows).to_csv(os.path.join(OUT_DIR, "PRIVATE_top_sentences.csv"), index=False)
dom = pd.DataFrame(drows).groupby("y").mean().T; dom.columns = ["Non-dep", "Dep"]
dom.plot.barh(figsize=(5, 4)); plt.xlabel("Fraction of top-IG sentences mentioning domain")
plt.title("PHQ-8 domains in most important sentences"); savefig("phq_domains_top_sentences.png")
interp["phq_domains"] = dom.round(4).to_dict()


# ---------- 5. Acoustic grounding of important selected chunks (per-chunk eGeMAPS) ----------
def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def chunk_egemaps_df(e, idx):
    return pd.DataFrame(e["cheap"].numpy()[:, GM_SLICE][np.asarray(idx)], columns=GM_NAMES).reset_index(drop=True)


ACOUSTIC_FEATS = ["F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
                  "loudness_sma3_amean", "jitterLocal_sma3nz_amean", "shimmerLocaldB_sma3nz_amean",
                  "HNRdBACF_sma3nz_amean", "VoicedSegmentsPerSec", "MeanUnvoicedSegmentLength", "loudnessPeaksPerSec"]
if CHEAP_EGEMAPS:
    diffs, hi_means = [], []
    for e, r in zip(test_entries, ig):
        sel = SEL_TEST[e["sid"]]
        if len(sel) < 4:
            continue
        g = chunk_egemaps_df(e, sel)
        feats = [f for f in ACOUSTIC_FEATS if f in g.columns]
        a = r["chunk_attr"]; hi, lo = a >= np.quantile(a, 0.75), a <= np.quantile(a, 0.25)
        diffs.append(dict(y=e["label"], **(g.loc[hi, feats].mean() - g.loc[lo, feats].mean()).to_dict()))
        hi_means.append(dict(y=e["label"], **g.loc[hi, feats].mean().to_dict()))
    if diffs:
        diffs, hi_means = pd.DataFrame(diffs), pd.DataFrame(hi_means)
        feats = [f for f in ACOUSTIC_FEATS if f in diffs.columns]
        arows = []
        for f in feats:
            dd = diffs.loc[diffs.y == 1, f].dropna()
            pw = stats.wilcoxon(dd).pvalue if len(dd) > 5 and (dd != 0).any() else np.nan
            h1, h0 = hi_means.loc[hi_means.y == 1, f].dropna(), hi_means.loc[hi_means.y == 0, f].dropna()
            pb = stats.mannwhitneyu(h1, h0).pvalue if len(h1) > 2 and len(h0) > 2 else np.nan
            arows.append(dict(feature=f, dep_high_minus_low=dd.mean(), p_within=pw,
                              dep_high=h1.mean(), nondep_high=h0.mean(), p_between=pb))
        ac = pd.DataFrame(arows)
        ac["q_within"], ac["q_between"] = bh_fdr(ac.p_within.fillna(1)), bh_fdr(ac.p_between.fillna(1))
        print("\nAcoustic profile of high- vs low-attribution selected chunks (BH-FDR):\n", ac.round(4).to_string(index=False))
        ac.to_csv(os.path.join(OUT_DIR, "acoustic_grounding.csv"), index=False)
        interp["acoustic"] = ac.round(5).to_dict(orient="records")

# ---------- 6. Per-speaker timelines (selected chunks only) ----------
correct = [(i, p) for i, p in enumerate(p_test) if (p >= THR) == yte[i]]
dep_i = max([c for c in correct if yte[c[0]] == 1], key=lambda c: c[1], default=(None,))[0]
ctl_i = min([c for c in correct if yte[c[0]] == 0], key=lambda c: c[1], default=(None,))[0]
cases = [(i, lab) for i, lab in [(dep_i, "Depressed"), (ctl_i, "Non-depressed")] if i is not None]
if cases:
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 2.6 * len(cases)), squeeze=False)
    for row, (i, lab) in enumerate(cases):
        r, e = ig[i], test_entries[i]
        t_all, t_sel = e["times"] / 60, e["times"][SEL_TEST[e["sid"]]] / 60
        axes[row, 0].bar(range(len(r["sent_attr"])), r["sent_attr"], color=np.where(r["sent_attr"] > 0, "tab:red", "tab:green"))
        axes[row, 0].set_title(f"{lab} speaker, P(dep)={p_test[i]:.2f}: sentence IG", fontsize=9)
        axes[row, 0].set_xlabel("Sentence index")
        axes[row, 1].scatter(t_all, np.zeros_like(t_all), marker="|", c="lightgray", label="not sent")
        axes[row, 1].bar(t_sel, r["chunk_attr"], width=CHUNK_SEC / 60,
                         color=np.where(r["chunk_attr"] > 0, "tab:red", "tab:green"))
        axes[row, 1].set_title(f"IG of the {len(t_sel)}/{len(t_all)} selected chunks", fontsize=9)
        axes[row, 1].set_xlabel("Interview time (min)")
    savefig("explanation_timelines.png")

# ---------- 7. [S5] What does the selector keep? ----------
if F["selector"] != "all" and F["frac"] < 1.0:
    pos = np.concatenate([SEL_TEST[s["sid"]] / max(1, len(s["audio"]) - 1) for s in test_s])
    ks_p = stats.kstest(pos, "uniform").pvalue
    jac = []
    for s in test_s:
        a = set(SEL_TEST[s["sid"]].tolist()); k = len(a); T_ = len(s["audio"])
        if k >= T_:
            continue
        b = set(np.argsort(-s["energy"], kind="stable")[:k].tolist())
        c = set(np.random.RandomState(0).choice(T_, k, replace=False).tolist())
        jac.append(dict(vs_energy=len(a & b) / len(a | b), vs_random=len(a & c) / len(a | c)))
    jac = pd.DataFrame(jac)
    print(f"\nSelection position vs uniform: KS p={ks_p:.3g} | Jaccard with loudest-K {jac.vs_energy.mean():.3f}, "
          f"with random-K {jac.vs_random.mean():.3f}")
    interp["selection"] = dict(position_ks_p=float(ks_p), jaccard_vs_energy=float(jac.vs_energy.mean()),
                               jaccard_vs_random=float(jac.vs_random.mean()))
    plt.figure(figsize=(4.5, 3))
    plt.hist(pos, bins=10, range=(0, 1), density=True, color="tab:blue", alpha=0.8)
    plt.axhline(1.0, c="k", ls=":", lw=0.8)
    plt.xlabel("Relative position in interview"); plt.ylabel("Density of kept chunks")
    plt.title("Where the selector keeps audio"); savefig("selection_position.png")
    if CHEAP_EGEMAPS:
        srows = []
        for e in test_entries:
            sel = SEL_TEST[e["sid"]]; T_ = len(e["times"])
            if len(sel) >= T_ or len(sel) < 2:
                continue
            g = chunk_egemaps_df(e, np.arange(T_)); mask = np.zeros(T_, bool); mask[sel] = True
            feats = [f for f in ACOUSTIC_FEATS if f in g.columns]
            srows.append(dict(y=e["label"], **(g.loc[mask, feats].mean() - g.loc[~mask, feats].mean()).to_dict()))
        if srows:
            srows = pd.DataFrame(srows); feats = [f for f in ACOUSTIC_FEATS if f in srows.columns]
            prof = []
            for f in feats:
                d = srows[f].dropna()
                p = stats.wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan
                prof.append(dict(feature=f, kept_minus_dropped=d.mean(), p=p))
            prof = pd.DataFrame(prof); prof["q"] = bh_fdr(prof.p.fillna(1))
            print("Acoustic profile: kept vs dropped chunks (BH-FDR):\n", prof.round(4).to_string(index=False))
            prof.to_csv(os.path.join(OUT_DIR, "selection_acoustic_profile.csv"), index=False)
            interp["selection"]["acoustic_profile"] = prof.round(5).to_dict(orient="records")

with open(os.path.join(OUT_DIR, "interpretability_summary.json"), "w") as f:
    json.dump(interp, f, indent=2, default=float)
print("\nInterpretability results saved to", OUT_DIR)

# ============================================================
# [S6] SAVE + ONNX EXPORT (on-device pipeline)
# ============================================================
#   chunk_scorer.onnx  : chunk_feats (T, D_CHEAP) float32     -> chunk_scores (T,)   [runs BEFORE WavLM]
#   text_encoder.onnx  : input_ids, attention_mask (B,S) int64 -> sent_emb (B,D_TEXT)
#   audio_encoder.onnx : input_values (B,CHUNK_SIZE) float32  -> chunk_emb (B,768)  [only K chunks]
#   fusion_head.onnx   : text (W,L,D_TEXT), audio (1,K,768)[, egemaps (1,88)]
#                        -> prob_depressed, logodds, w_sent, w_win, w_audio
import onnxruntime as ort

torch.save(dict(states=[m.state_dict() for m in F["members"]], threshold=THR, wavlm_layers=WAVLM_LAYERS,
                text_model=TEXT_MODEL, gm_names=GM_NAMES, selector=F["selector"], keep_frac=F["frac"],
                k_min=K_MIN, cheap_names=CHEAP_NAMES, chunk_sec=CHUNK_SEC),
           os.path.join(OUT_DIR, "fusion_topk_ensemble.pt"))


class DeployHead(nn.Module):
    def __init__(self, ens_model):
        super().__init__(); self.ens = ens_model

    def forward(self, text, audio, egemaps=None):
        logits, att = self.ens(text, audio, egemaps, return_attn=True)
        lo = logits[:, 1] - logits[:, 0]
        return torch.sigmoid(lo), lo, att["w_sent"], att["w_win"], att["w_audio"]


class EnsembleScorer(nn.Module):
    """Same arithmetic as Ensemble.chunk_scores: mean of per-member z-scored scores."""
    def __init__(self, scorers):
        super().__init__(); self.scorers = nn.ModuleList(scorers)

    def forward(self, x):
        return zscore(torch.stack([sc(x) for sc in self.scorers])).mean(0)


def export_onnx(module, args, path, input_names, output_names, dynamic_axes):
    kw = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
              opset_version=OPSET, do_constant_folding=True)
    try:
        torch.onnx.export(module, args, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(module, args, path, **kw)
    print("Exported", path)


def verify_onnx(path, module, args, names, atol=1e-3):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = module(*args)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    out = sess.run(None, {n: a.numpy() for n, a in zip(names, args)})
    err = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(out, ref))
    print(f"  parity {os.path.basename(path)}: max |onnx - torch| = {err:.2e} {'OK' if err < atol else 'CHECK'}")
    return sess


head = DeployHead(copy.deepcopy(M).cpu().eval()).eval()
head_inputs = ["text", "audio"] + (["egemaps"] if N_GM else [])
mk_args = lambda w, l, t: tuple([torch.randn(w, l, D_TEXT) * 0.05, torch.randn(1, t, 768)] +
                                ([GM_BASE.cpu().clone()] if N_GM else []))
p_head = os.path.join(ONNX_DIR, "fusion_head.onnx")
export_onnx(head, mk_args(3, SEQ_LEN, 40), p_head, head_inputs,
            ["prob_depressed", "logodds", "w_sent", "w_win", "w_audio"],
            {"text": {0: "n_windows", 1: "win_len"}, "audio": {1: "n_chunks"},
             "w_sent": {0: "n_windows", 1: "win_len"}, "w_win": {0: "n_windows"}, "w_audio": {0: "n_chunks"}})
sess_head = verify_onnx(p_head, head, mk_args(2, 37, 13), head_inputs)

sess_scorer = None
if F["selector"] in LEARNED:
    scorer_cpu = EnsembleScorer([copy.deepcopy(m.scorer).cpu().eval() for m in F["members"]]).eval()
    d_in = D_CHEAP if F["selector"] == "learned" else 768
    p_scorer = os.path.join(ONNX_DIR, "chunk_scorer.onnx")
    export_onnx(scorer_cpu, (torch.randn(30, d_in),), p_scorer, ["chunk_feats"], ["chunk_scores"],
                {"chunk_feats": {0: "n_chunks"}, "chunk_scores": {0: "n_chunks"}})
    ref_x = dev_s[0]["cheap" if F["selector"] == "learned" else "audio"].float().cpu()
    sess_scorer = verify_onnx(p_scorer, scorer_cpu, (ref_x,), ["chunk_feats"])

te_cpu = copy.deepcopy(text_encoder).cpu().eval()
p_text = os.path.join(ONNX_DIR, "text_encoder.onnx")
tok = tokenizer(["i have been feeling tired lately", "no"], return_tensors="pt", padding=True)
export_onnx(te_cpu, (tok["input_ids"], tok["attention_mask"]), p_text, ["input_ids", "attention_mask"], ["sent_emb"],
            {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "sent_emb": {0: "batch"}})
tok2 = tokenizer(["a longer sentence to check that dynamic axes work properly", "ok", "not really"],
                 return_tensors="pt", padding=True)
sess_text = verify_onnx(p_text, te_cpu, (tok2["input_ids"], tok2["attention_mask"]), ["input_ids", "attention_mask"])

ae_cpu = copy.deepcopy(audio_encoder).cpu().eval()
p_audio = os.path.join(ONNX_DIR, "audio_encoder.onnx")
sess_audio = None
try:
    export_onnx(ae_cpu, (torch.randn(2, CHUNK_SIZE) * 0.1,), p_audio, ["input_values"], ["chunk_emb"],
                {"input_values": {0: "batch"}, "chunk_emb": {0: "batch"}})
    sess_audio = verify_onnx(p_audio, ae_cpu, (torch.randn(3, CHUNK_SIZE) * 0.1,), ["input_values"])
except Exception as ex:
    print("WavLM export failed (try another OPSET / transformers version):", repr(ex))

tokenizer.save_pretrained(os.path.join(ONNX_DIR, "tokenizer"))
gm_shape = f",egemaps:1x{N_GM}" if N_GM else ""
deploy_cfg = dict(
    threshold=THR, ensemble_size=len(F["members"]), sample_rate=SR, chunk_samples=CHUNK_SIZE,
    max_audio_chunks=MAX_AUDIO_CHUNKS, use_participant_segments=USE_PARTICIPANT_SEGMENTS,
    min_words=MIN_WORDS, max_sents=MAX_SENTS, max_token_len=128, seq_len=SEQ_LEN, stride=STRIDE,
    n_views=N_VIEWS, tta_frac=TTA_FRAC, view_seed="crc32(speaker_id)", tta_scope="views drawn inside the K selected chunks",
    text_model=TEXT_MODEL, text_pooling=TEXT_POOLING, wavlm=WAVLM_NAME, wavlm_layers=list(WAVLM_LAYERS),
    egemaps=dict(enabled=bool(N_GM), feature_set="eGeMAPSv02 Functionals", computed_on="participant signal",
                 feature_order=GM_NAMES),
    selection=dict(
        selector=F["selector"], keep_frac=F["frac"], k_min=K_MIN,
        k_rule="K = T if keep_frac >= 1 else min(T, max(k_min, ceil(keep_frac * T)))",
        order="descending score (stable sort); kept chunk indices re-sorted in time",
        scorer_onnx="chunk_scorer.onnx" if sess_scorer is not None else None,
        scorer_input="cheap per-chunk features" if F["selector"] == "learned" else "WavLM chunk embeddings",
        cheap_features=dict(log_mel=dict(n_mels=N_MELS, n_fft=MEL_NFFT, hop=MEL_HOP,
                                         power_to_db="ref=1.0, top_db=None", stats=["mean over time", "std over time"]),
                            egemaps_per_chunk=bool(CHEAP_EGEMAPS), order=CHEAP_NAMES),
        energy_db="10*log10(mean(x^2) + 1e-10) per chunk", random_seed="(crc32(speaker_id) + 1) % 2**31"),
    opset=OPSET,
    tensorrt=dict(
        fusion_head=("trtexec --onnx=fusion_head.onnx --saveEngine=fusion_head_fp16.engine --fp16 "
                     f"--minShapes=text:1x1x{D_TEXT},audio:1x1x768{gm_shape} "
                     f"--optShapes=text:8x50x{D_TEXT},audio:1x{max(1, n_keep(60, F['frac']))}x768{gm_shape} "
                     f"--maxShapes=text:16x50x{D_TEXT},audio:1x{MAX_AUDIO_CHUNKS}x768{gm_shape}"),
        text_encoder=("trtexec --onnx=text_encoder.onnx --saveEngine=text_encoder_fp16.engine --fp16 "
                      "--minShapes=input_ids:1x4,attention_mask:1x4 --optShapes=input_ids:32x32,attention_mask:32x32 "
                      "--maxShapes=input_ids:32x128,attention_mask:32x128"),
        audio_encoder=(f"trtexec --onnx=audio_encoder.onnx --saveEngine=audio_encoder_fp16.engine --fp16 "
                       f"--minShapes=input_values:1x{CHUNK_SIZE} --optShapes=input_values:8x{CHUNK_SIZE} "
                       f"--maxShapes=input_values:{AUDIO_BATCH}x{CHUNK_SIZE}")))
with open(os.path.join(ONNX_DIR, "deploy_config.json"), "w") as f:
    json.dump(deploy_cfg, f, indent=2)


def onnx_wavlm(ch):
    return np.concatenate([sess_audio.run(None, {"input_values": np.ascontiguousarray(ch[i:i + AUDIO_BATCH])})[0]
                           for i in range(0, len(ch), AUDIO_BATCH)])


def onnx_select(sid, cheap, energy, chunks, select=True):
    """Device-side selection, identical to pick(). Returns (indices, WavLM embeddings of all chunks or None)."""
    n = len(cheap); k = n_keep(n, F["frac"])
    if not select or F["selector"] == "all" or k >= n:
        return np.arange(n), None
    if F["selector"] == "first":
        return np.arange(k), None
    if F["selector"] == "random":
        return np.sort(_sid_rng(sid).choice(np.arange(n), k, replace=False)), None
    if F["selector"] == "energy":
        return np.sort(np.argsort(-energy, kind="stable")[:k]), None
    emb_all = None
    if F["selector"] == "learned":
        sc = sess_scorer.run(None, {"chunk_feats": cheap.astype(np.float32)})[0]
    else:                                                  # learned_wavlm: needs WavLM on every chunk
        emb_all = onnx_wavlm(chunks)
        sc = sess_scorer.run(None, {"chunk_feats": emb_all.astype(np.float32)})[0]
    return np.sort(np.argsort(-sc, kind="stable")[:k]), emb_all


def onnx_predict(transcript_csv, wav_path, sid, select=True):
    """Full on-device pipeline with per-stage timing. select=False embeds every chunk (cost reference only)."""
    tm, clock = {}, time.perf_counter
    t0 = clock(); sents, segs = load_transcript(transcript_csv)
    chunks, times, y_use = load_participant_audio(wav_path, segs); tm["load_audio"] = clock() - t0
    t0 = clock(); cheap, energy = chunk_features(chunks); tm["cheap_features"] = clock() - t0
    t0 = clock(); sel, emb_all = onnx_select(sid, cheap, energy, chunks, select); tm["select"] = clock() - t0
    t0 = clock(); audio = emb_all[sel] if emb_all is not None else onnx_wavlm(chunks[sel]); tm["wavlm"] = clock() - t0
    t0 = clock(); embs = []
    for i in range(0, len(sents), EMB_BATCH):
        t = tokenizer(sents[i:i + EMB_BATCH], return_tensors="np", truncation=True, padding=True, max_length=128)
        embs.append(sess_text.run(None, {"input_ids": t["input_ids"].astype(np.int64),
                                         "attention_mask": t["attention_mask"].astype(np.int64)})[0])
    text = np.concatenate(embs); tm["text_encoder"] = clock() - t0
    t0 = clock(); gm = speaker_egemaps(y_use)[None] if N_GM else None; tm["speaker_egemaps"] = clock() - t0
    t0 = clock(); los, expl = [], None
    for vi, (ks, kc) in enumerate(eval_views(sid, len(text), len(sel))):
        t_v = text[ks]; idx = window_index(len(t_v))
        feeds = {"text": t_v[idx].astype(np.float32), "audio": audio[kc][None].astype(np.float32)}
        if N_GM:
            feeds["egemaps"] = gm.astype(np.float32)
        prob, lo, w_sent, w_win, w_aud = sess_head.run(None, feeds)
        los.append(float(lo[0]))
        if vi == 0:
            sent_imp = np.zeros(len(text)); np.add.at(sent_imp, idx, w_win[:, None] * w_sent)
            expl = dict(top_sentences=[sents[i] for i in np.argsort(-sent_imp)[:3]],
                        top_audio_times_s=times[sel][np.argsort(-w_aud)[:3]].tolist())
    tm["fusion_head"] = clock() - t0
    p = float(sigmoid(np.mean(los)))
    return dict(prob=p, depressed=p >= THR, n_chunks=len(chunks), n_kept=len(sel),
                uplink_raw_KB=len(sel) * CHUNK_SIZE * 2 / 1024, uplink_emb_KB=len(sel) * 768 * 2 / 1024,
                timings=tm, **expl)


if sess_audio is not None and (F["selector"] not in LEARNED or sess_scorer is not None):
    lat = []
    for i in range(min(3, len(test_entries))):
        e = test_entries[i]
        r_sel = onnx_predict(e["tcsv"], e["wav"], e["sid"], select=True)
        r_all = onnx_predict(e["tcsv"], e["wav"], e["sid"], select=False)
        print(f"  E2E parity speaker {e['sid']}: torch P={p_test[i]:.4f} | onnx P={r_sel['prob']:.4f} | "
              f"kept {r_sel['n_kept']}/{r_sel['n_chunks']} chunks")
        for tag, r in [("top-k", r_sel), ("all", r_all)]:
            lat.append(dict(sid=e["sid"], pipeline=tag, n_chunks_encoded=r["n_kept"], **r["timings"],
                            total_s=sum(r["timings"].values())))
    lat = pd.DataFrame(lat)
    lat_mean = lat.drop(columns="sid").groupby("pipeline").mean()
    print("\nHost CPU (ONNX Runtime) latency per session, seconds — rerun on the Jetson for the paper:\n",
          lat_mean.round(3).to_string())
    lat.to_csv(os.path.join(OUT_DIR, "edge_latency_host.csv"), index=False)

print("\nDone. Results:", OUT_DIR, "| Deployment bundle:", ONNX_DIR)
