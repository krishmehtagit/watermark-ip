#!/usr/bin/env python3
"""Bit-exact golden model for the merged watermark IP (plans/initial_plan.md).

One keyed uniform u per (context, token) feeds every scheme:
  GUMBEL  SAMPLE: argmax(l + G(u))                       Aaronson / exponential-minimum
  GREEN   SAMPLE: argmax(l + delta*green(u) + G(u_free)) Kirchenbauer soft red/green list
  DETECT        : S = sum -ln(1-u)  and  g = sum green(u) over the scored tokens
  MASK          : green(u) for all V indices              (stretch command)
green(u) is the TOP gamma of u, so both detectors respond to both schemes.

Block <-> function:
  seed_unit  step_seed / free_seed     hash_lanes  uniforms
  rom_index  pf_index                  gumbel_rom, score_rom  build_roms
  add_argmax sample                    detect_lanes + score_acc  detect
  mask_packer mask                     host  p_gumbel, p_green, repeat_flags

python3 src/golden/wm_ref.py runs the self-test (a few seconds).
Gumbel ROM, pf_index and the repeat handling are Krish Mehta's; seed and hash
constants and the green test are Fahad Tajiki's; the merge is in plans/design_review.md.
"""
import math

import numpy as np

M32, M64 = (1 << 32) - 1, (1 << 64) - 1
C_IDX = 0x9E3779B1                       # odd: t -> t*C_IDX is a bijection mod 2^32
C_CTX = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D)
GUMBEL, GREEN = 0, 1                     # CONFIG scheme field


def fmix64(k):                           # MurmurHash3 64-bit finaliser (seed_unit, detect_lanes)
    k ^= k >> 33; k = (k * 0xFF51AFD7ED558CCD) & M64
    k ^= k >> 33; k = (k * 0xC4CEB9FE1A85EC53) & M64
    return k ^ (k >> 33)


def fmix32(h):                           # MurmurHash3 32-bit finaliser on a uint64 array (one lane each)
    h = h ^ (h >> np.uint64(16)); h = (h * np.uint64(0x85EBCA6B)) & np.uint64(M32)
    h = h ^ (h >> np.uint64(13)); h = (h * np.uint64(0xC2B2AE35)) & np.uint64(M32)
    return h ^ (h >> np.uint64(16))


def step_seed(key, ctx):                 # keyed seed; ctx = previous k token ids, oldest first
    x = key
    for j, t in enumerate(reversed(list(ctx))):
        x ^= ((int(t) + 1) * C_CTX[j]) & M64
    return fmix64(x)


def free_seed(nonce, step):              # unkeyed seed: repeated contexts, and GREEN-scheme noise
    return fmix64(nonce ^ (((step + 1) * C_CTX[3]) & M64))


def uniforms(seed, idx):                 # hash_lanes: one uint32 u per token index
    t = np.asarray(idx, dtype=np.uint64)
    return fmix32(((t * np.uint64(C_IDX)) & np.uint64(M32)) ^ np.uint64(seed & M32))


def green_thr(gamma):                    # CONFIG: green iff u >= green_thr  (top gamma of [0, 2^32))
    return (1 << 32) - int(gamma * (1 << 32))


def green(u, thr):
    return np.asarray(u, dtype=np.uint64) >= np.uint64(thr)


def pf_index(u):                         # rom_index: pseudo-float address, accurate in both tails
    u = np.asarray(u, dtype=np.uint64)
    half = u >> np.uint64(31)
    w = np.where(half == 1, np.uint64(M32) - u, u)                    # distance to nearer end, < 2^31
    e = (32 - np.frexp(w.astype(np.float64))[1]).astype(np.uint64)  # leading zeros (32 if w == 0)
    m = ((w << (e + np.uint64(1))) & np.uint64(M32)) >> np.uint64(24)  # 8 bits after the leading one
    return (half * np.uint64(33) + e) * np.uint64(256) + m


def build_roms():                        # 16,896 x 16-bit each; the HLS code #includes the same tables
    G = np.zeros(2 * 33 * 256, np.int16)     # Gumbel noise -ln(-ln u), Q8.8
    S = np.zeros(2 * 33 * 256, np.uint16)    # detection score -ln(1-u), Q5.11
    for half in (0, 1):
        for e in range(1, 33):
            for m in range(256):
                if e == 32:
                    if m: continue
                    lo = hi = 0
                elif e <= 23:
                    lo = (256 + m) << (23 - e); hi = lo + (1 << (23 - e)) - 1
                else:
                    if m & ((1 << (e - 23)) - 1): continue           # address never produced
                    lo = hi = (256 + m) >> (e - 23)
                x = ((lo + hi) / 2 + 0.5) / 2**32                    # distance of u~ to nearer end
                neg_ln_u = -math.log1p(-x) if half else -math.log(x)
                neg_ln_1mu = -math.log(x) if half else -math.log1p(-x)
                a = (half * 33 + e) * 256 + m
                G[a] = round(256 * -math.log(neg_ln_u))
                S[a] = round(2048 * neg_ln_1mu)
    return G, S


G_ROM, S_ROM = build_roms()


def sample(key, ctx, logits_q88, scheme=GUMBEL, gamma=0.25, delta_q88=512,
           free=False, nonce=0, step=0):                              # SAMPLE command
    V = len(logits_q88)
    v = logits_q88.astype(np.int32)
    u_free = uniforms(free_seed(nonce, step), range(V))
    if scheme == GUMBEL:
        u_noise = u_free if free else uniforms(step_seed(key, ctx), range(V))
    else:                                                             # GREEN
        v = v + delta_q88 * green(uniforms(step_seed(key, ctx), range(V)), green_thr(gamma))
        u_noise = u_free
    v = v + G_ROM[pf_index(u_noise)]
    t = int(np.argmax(v))                                             # first maximum: smallest index wins
    return t, int(v[t])


def mask(key, ctx, V, gamma=0.25):                                    # MASK command (stretch)
    return green(uniforms(step_seed(key, ctx), range(V)), green_thr(gamma))


def detect(key, tokens, skip, k, gamma=0.25):                         # DETECT: S (Q21.11), g, N
    thr, S, g, N = green_thr(gamma), 0, 0, 0
    for i in range(k, len(tokens)):
        if skip[i]: continue
        u = uniforms(step_seed(key, tokens[i - k:i]), [tokens[i]])
        S = (S + int(S_ROM[pf_index(u)[0]])) & M32
        g += int(green(u, thr)[0]); N += 1
    return S, g, N


def repeat_flags(tokens, k):             # host: True where the k-token window was already seen
    seen, out = set(), []
    for i in range(len(tokens)):
        w = tuple(tokens[max(0, i - k):i])
        out.append(i >= k and w in seen); seen.add(w)
    return out


def p_gumbel(S, N):                      # host: S ~ Gamma(N, 1) for unwatermarked text
    x = S / 2048
    try:
        from scipy.special import gammaincc
        return float(gammaincc(N, x))
    except ImportError:                  # log-space series: P(Poisson(x) < N)
        terms = [j * math.log(x) - math.lgamma(j + 1) - x for j in range(N)]
        mx = max(terms)
        return math.exp(mx + math.log(sum(math.exp(t - mx) for t in terms)))


def p_green(g, N, gamma=0.25):         # host: exact tail P[Binomial(N, gamma) >= g]
    try:
        from scipy.stats import binom
        return float(binom.sf(g - 1, N, gamma))
    except ImportError:
        return sum(math.comb(N, j) * gamma**j * (1 - gamma)**(N - j) for j in range(g, N + 1))


def z_green(g, N, gamma=0.25):           # host: green count ~ Binomial(N, gamma) for unwatermarked text
    return (g - gamma * N) / math.sqrt(N * gamma * (1 - gamma))


def to_q88(x):                           # host: subtract max, clamp, quantise
    x = np.asarray(x, np.float64)
    return np.round(np.clip(x - x.max(), -128, 127) * 256).astype(np.int16)


def _chi2(cnt, p):
    n = cnt.sum(); return float(((cnt - n * p) ** 2 / (n * p)).sum())


def _selftest():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 1 << 32, size=2_000_000, dtype=np.uint64)
    G = G_ROM[pf_index(u)] / 256
    print(f"ROM noise        mean {G.mean():.4f} (0.5772)  var {G.var():.4f} (1.6449)")

    lg = np.array([0.0, -0.5, -1.0, -2.0, -3.0]); q = to_q88(lg); n = 50_000
    cnt = np.zeros(5)
    for c in range(n): cnt[sample(0xABC, [c], q)[0]] += 1
    p = np.exp(lg); p /= p.sum()
    print(f"GUMBEL vs softmax(l)          chi2 {_chi2(cnt, p):6.2f}  (95% cutoff 9.49)")

    gm = mask(0xABC, [7], 5, 0.4)                                     # fixed context: green set fixed
    cnt = np.zeros(5)
    for s in range(n): cnt[sample(0xABC, [7], q, GREEN, 0.4, 512, nonce=99, step=s)[0]] += 1
    p = np.exp(lg + 2.0 * gm); p /= p.sum()
    print(f"GREEN  vs softmax(l+delta*g)  chi2 {_chi2(cnt, p):6.2f}  (95% cutoff 9.49)  green set {gm.astype(int)}")

    V, T, k, key = 4096, 200, 1, 0x1234
    for scheme in ("plain", "GUMBEL", "GREEN"):
        r = np.random.default_rng(7); toks = [int(r.integers(V))]
        for i in range(T):
            ql = to_q88(r.normal(0, 3, V))
            if scheme == "plain":
                pp = np.exp(ql / 256); toks.append(int(r.choice(V, p=pp / pp.sum())))
            else:
                rep = repeat_flags(toks + [0], k)[-1]
                toks.append(sample(key, toks[-k:], ql, GUMBEL if scheme == "GUMBEL" else GREEN,
                                   free=rep, nonce=0x5EED, step=i)[0])
        S, g, N = detect(key, toks, repeat_flags(toks, k), k)
        print(f"{scheme:6s} text, T={T}:  Gumbel p {p_gumbel(S, N):9.2e}   green p {p_green(g, N):9.2e}"
              f"  (z {z_green(g, N):+6.2f})   N {N}")


if __name__ == "__main__":
    _selftest()
