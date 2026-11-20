# Initial Plan: Watermarking Sampler for LLM Text

Repository: https://github.com/krishmehtagit/watermark-ip

## Project Team

- Fahad Tajiki (fxt2021)
- Krish Mehta (km6152)

## IP Definition

### What the IP does

A language model writes text one token at a time. At each step it gives a score
(logit) to every token in its vocabulary of V tokens and samples the next token
from softmax(ℓ). Our IP is this sampler. The difference from a normal sampler is
where the randomness comes from: we compute it from a secret key and the
previous k tokens. Anyone with the key can then detect the watermark from the
tokens alone, without the model.

The IP supports two published watermark schemes, chosen at CONFIG:

- GUMBEL (default): the scheme from Aaronson, also studied by Kuditipudi et al.
  The IP returns argmax_t (ℓ_t + G(u_t)), where G is Gumbel noise made from a
  keyed random number u_t. With a random key this is a normal softmax sample
  (the Gumbel-max trick), so the text quality does not change.
- GREEN: the green-list scheme from Kirchenbauer et al. A fraction γ of tokens
  is green (the ones with high u), and those get +δ on their logit. The IP
  then samples using unkeyed Gumbel noise. This changes the text a little but is
  easy to detect (under 100 tokens in the MarkMyWords benchmark).

The detector recomputes u for every token in a text and returns two numbers in
one pass: the Gumbel score S and the green count g. Both schemes push toward
tokens with high u, so both statistics react to both schemes.

Commands:
- SAMPLE: context and V logits in, chosen token out.
- DETECT: token sequence in, S, g and N (tokens scored) out. The host computes
  the p-values.
- MASK (stretch goal): context in, the V-bit green mask out.

The language model (GPT-2 small, V = 50,257) runs in Python. We record its
logits and replay them in the testbench. The model weights are not changed.

### Mathematical operations

CONFIG sets a 64-bit key K, V, context length k (1 to 4), a 64-bit nonce, the
scheme, the green threshold θ = 2^32 − floor(γ·2^32), and δ in Q8.8. Logits are
int16 Q8.8. The host divides them by the temperature, subtracts the maximum and
clamps them before sending. fmix64 and fmix32 are the MurmurHash3 finalizers.
C_1 to C_4 are fixed odd 64-bit constants (C_CTX in src/golden/wm_ref.py).

1. Seeds, once per step:
   s = fmix64(K xor X), where X is the xor over j = 1..k of
   ((t_{i-j} + 1)·C_j mod 2^64), and t_{i-j} are the previous k tokens.
   s_f = fmix64(nonce xor ((step + 1)·C_4 mod 2^64)) is the unkeyed seed.
2. Uniforms, one per token index t:
   u_t = fmix32(s[31:0] xor (t·0x9E3779B1 mod 2^32)), read as (u + 0.5)/2^32.
   u'_t is the same thing computed from s_f.
3. Noise: G(u) = −ln(−ln u), read from a ROM in Q8.8 (range −3.13 to 22.87).
4. Sample (ties go to the smallest t):
   GUMBEL: argmax_t (ℓ_t + G(u_t)), using u'_t instead if the context repeats.
   GREEN: argmax_t (ℓ_t + δ·[u_t ≥ θ] + G(u'_t)).
5. Detect, over tokens i ≥ k that are not marked as repeats:
   S = sum of −ln(1 − u) (from a second ROM, Q5.11), g = number of tokens with
   u ≥ θ, N = number of tokens scored.
   On normal text each score is Exp(1), so S ~ Gamma(N, 1) and
   p_S = gammaincc(N, S) (Fernandez et al.). Also g ~ Binomial(N, γ), so
   p_g = P[Binomial(N, γ) ≥ g]. The text is called watermarked if p < 10^-6.
6. Repeated contexts: if the same k-token context already appeared in the text,
   the host sets a "free" bit. Generation then uses s_f and detection skips that
   token. This way the noise never repeats and the p-values stay exact.

ROM address: let w be the distance of u from the nearer end of [0, 2^32), e the
number of leading zeros of w, and m the 8 bits after the leading one. The
address is (half·33 + e)·256 + m, which gives 16,896 entries. This works like a
small floating point number, so the table stays accurate near u = 0 and u = 1.
That matters because the winning tokens come from the tail. A simple table on
the top 16 bits of u would cut off everything above 1 − 2^-16, and on average
0.77 of the 50,257 tokens land there at every step.

Golden model outline (the full bit-exact version is src/golden/wm_ref.py):

```python
def sample(ctx, logits, free, step):                            # SAMPLE
    u  = uniforms(step_seed(K, ctx), range(V))                  # keyed
    uf = uniforms(free_seed(nonce, step), range(V))             # unkeyed
    if scheme == GUMBEL:
        v = logits + G_ROM[pf_index(uf if free else u)]
    else:                                                       # GREEN
        v = logits + delta * (u >= theta) + G_ROM[pf_index(uf)]
    return argmax(v)                                            # first maximum

def detect(tokens, skip):                                       # DETECT
    S = g = N = 0
    for i in range(k, len(tokens)):
        if skip[i]: continue
        u = uniforms(step_seed(K, tokens[i-k:i]), [tokens[i]])
        S += S_ROM[pf_index(u)]; g += (u >= theta); N += 1
    return S, g, N     # host: gammaincc(N, S/2048) and binom.sf(g-1, N, gamma)
```

### Why it suits hardware acceleration

- No softmax is needed. Normal sampling needs V exponentials, a sum, a division
  and a search. Here it is V additions and one running maximum, all in integers,
  in one pass over the logits. Both schemes use the same pass.
- The V hashes and ROM reads are independent, so P lanes can run in parallel at
  II = 1, followed by a max tree.
- It runs once for every generated token, so its latency adds directly to the
  time per token.
- The state is small: the key, two seeds and the running maximum. Nothing grows
  with V.
- In detection, each token's seed depends only on input tokens, so several
  tokens can be scored per clock.

### How it talks to the host

We use the course command-response pattern: one 64-bit AXI4-Stream in for
commands and data, one 64-bit AXI4-Stream out for responses, and AXI-Lite for
status only (halted, error code, tx_id). On PYNQ an AXI DMA drives both streams.

| Command | Header | Data after header | Response |
|---|---|---|---|
| CONFIG (0) | K, V, k, nonce, scheme, θ, δ | none | none |
| SAMPLE (1) | free bit, step, k context ids | ceil(V/4) beats of 4 int16 logits, TLAST on the last | header, then token id and its winning value |
| DETECT (2) | T | ceil(T/2) beats of 2 words (bits 23:0 token id, bit 31 skip), TLAST on the last | header, then S (Q21.11), N and g |
| END (3) | none | none | none, the kernel returns |
| MASK (4), stretch | k context ids | none | header, then ceil(V/64) beats; bit b of beat w is token 64w + b |

## IP Architecture

```
SAMPLE  in -> controller -> seed_unit -> hash_lanes[P] -> rom_index[P] -> gumbel_rom --+
                  +-> logit_unpack (4 logits per beat) ---------------------------> add_argmax -> resp -> out
DETECT  in -> controller -> ctx_shift -> detect_lanes[P_d] -> score_acc -> resp -> out
```

### Sub-modules

1. controller: reads the command headers, checks for errors, keeps the
   configuration and writes the AXI-Lite status.
2. seed_unit: computes the keyed seed s and the unkeyed seed s_f once per SAMPLE
   with fmix64. It runs once per step, so it is not on the critical path.
3. hash_lanes (P copies): lane p handles token index t = cP + p. Each lane has
   two pipelined fmix32 units (for u and u') and a comparator for the green bit
   (u ≥ θ). Instead of multiplying the index, each lane adds P·0x9E3779B1 every
   cycle.
4. rom_index (P copies): picks u or u' depending on the scheme and the free bit,
   then uses a leading zero count and a shift to make the ROM address. No DSPs.
5. gumbel_rom: the 16,896 x 16-bit table, about 9 BRAM36. It is dual-port, so two
   lanes share one copy.
6. logit_unpack: splits each 64-bit beat into 4 logits and sets indices past V to
   −32768 so they can never win.
7. add_argmax: P additions of ℓ + δ·green + G (18 bits, δ only in GREEN mode), a
   P-input max tree, and a running maximum with its index across beats.
8. ctx_shift and detect_lanes (P_d copies): a shift register gives each lane its
   token and the k tokens before it. Each lane has an fmix64, an fmix32, a
   rom_index and the green comparator. Two lanes share one dual-port score ROM
   (Q5.11). Skipped tokens add 0.
9. score_acc: adder trees over the lanes and 32-bit accumulators for S, g and N.
10. resp: builds the response header and payload.

For the MASK stretch goal, a mask_packer packs green bits into 64-bit beats.

### Interfaces

| From -> To | Type | What it carries |
|---|---|---|
| PS -> controller | AXI4-Stream in, 64-bit | command headers with tx_id, then logits or token words, ending in TLAST |
| resp -> PS | AXI4-Stream out, 64-bit | response header (tx_id, word count), then the results |
| controller -> PS | AXI-Lite | halted, error code, tx_id |
| controller -> seed_unit | FIFO (hls::stream) | K, nonce, context ids, step and free bit, once per SAMPLE |
| controller -> other blocks | registers, fixed during a command | scheme, θ, δ, V, k, and K for detect_lanes |
| seed_unit -> hash_lanes | FIFO | s[31:0] and s_f[31:0] |
| seed_unit -> rom_index | FIFO | the free bit |
| hash_lanes -> rom_index | FIFO, P per cycle | u, u' and the green bit |
| rom_index -> gumbel_rom | BRAM read port | 15-bit address in, 16-bit G out |
| gumbel_rom -> add_argmax | FIFO, P per cycle | G and the green bit |
| controller -> logit_unpack -> add_argmax | FIFO | 64-bit beats, then P logits per cycle |
| controller -> ctx_shift -> detect_lanes | FIFO | token words, then (context, token, skip) per lane |
| detect_lanes -> score_acc | FIFO, P_d per cycle | score (Q5.11) and green bit |
| add_argmax, score_acc -> resp | FIFO | result words |

No shared memory is used and the IP has no AXI master. The top level is an HLS
dataflow region, so if the output stream stalls the lanes wait and no data is
lost.

### Design choices

- Both schemes share seed_unit and hash_lanes. GREEN only adds a second fmix32
  and a comparator per lane, not a second datapath.
- Each block matches one function in the golden model (step_seed/free_seed,
  uniforms, pf_index, sample, detect), so each can be tested on its own.
- Build order: GUMBEL SAMPLE at P = 1, then P = 4, then DETECT, then GREEN, then
  MASK. Each step leaves a working IP.
- The stream width W sets the parallelism: P = W/16 logits and P_d = W/32 tokens
  per beat. The number of DSPs is the upper limit.
- For G(u) we looked at the AMD Floating-Point log core and CORDIC. Both need two
  logarithms per lane, which costs more DSPs and latency than one ROM read.

### Expected performance

- SAMPLE is limited by the input stream: 4 logits per beat, so P = 4 and
  ceil(50,257/4) = 12,565 cycles, about 126 µs per token at 100 MHz.
- DETECT handles 2 tokens per beat (P_d = 2) at II = 1.
- MASK (stretch) is limited by compute. Using all 8 fmix32 units on the keyed
  seed it takes ceil(50,257/8) = 6,283 cycles (about 63 µs) and sends out 6.3 KB
  instead of reading in 100 KB of logits.
- Rough resources on the xc7z020 (to be checked in synthesis): 110 to 150 of 220
  DSP48E1 and 27 of 140 BRAM36.
- Verification: bit-exact C simulation and co-simulation against the golden
  model, a χ² test that the samples follow softmax(ℓ) (or softmax(ℓ + δ·green)
  in GREEN), a check that p-values on normal text are uniform, and a NumPy
  software baseline for comparison.

### Existing IP

AMD AXI DMA for the two streams, and the AXI4-Stream Data FIFO if we need extra
buffering. We do not know of an AMD core for the keyed hash.

## References

- S. Aaronson, "My AI Safety Lecture for UT Effective Altruism," Shtetl-Optimized, 2022.
- R. Kuditipudi et al., "Robust Distortion-free Watermarks for Language Models," TMLR 2024, arXiv:2307.15593.
- J. Kirchenbauer et al., "A Watermark for Large Language Models," ICML 2023, arXiv:2301.10226.
- P. Fernandez et al., "Three Bricks to Consolidate Watermarks for Large Language Models," WIFS 2023, arXiv:2308.00113.
- J. Piet et al., "Mark My Words: Analyzing and Evaluating Language Model Watermarks," arXiv:2312.00273.
- A. Appleby, MurmurHash3 (public domain), fmix32 and fmix64.
