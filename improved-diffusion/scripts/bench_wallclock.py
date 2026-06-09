"""Wallclock benchmark: Diffusion+LCVR vs AR+CD on identical task.

For each (domain, backbone) we time the cost of generating
N_CANDS candidates for ONE structural target on the same GPU,
fairly batched, with warm-up runs excluded.

Outputs JSON + Markdown summary. Run once per domain (sonnet, ci).

Usage:
  python scripts/bench_wallclock.py --domain sonnet \
      --diff_ckpt diffusion_models/.../ema_0.9999_100000.pt \
      --ar_ckpt   ar_baseline/char_gpt2_sonnet_v2/ar_ckpt_ep60.pt \
      --target_json control_gen/target_son10.json \
      --out doc/wallclock_sonnet.json
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def time_ar_cd(ar_ckpt, target_rec, n_cands, seq_len, domain, device, n_warmup=2, n_repeat=3):
    """Time AR+CD for n_cands candidates × n_repeat runs."""
    from transformers import GPT2Config, GPT2LMHeadModel
    sys.path.insert(0, str(ROOT / "scripts"))
    if domain == "sonnet":
        from ar_sonnet_cd_sample import sample_constrained_batched
        def _sample(model, vocab, tgt, T, n_cands, seed):
            return sample_constrained_batched(model, vocab, tgt, T, n_cands,
                                              device=device, temperature=1.0,
                                              seed=seed, use_cd=True)
    elif domain == "ci":
        from ar_ci_cd_sample import sample_constrained_batched as _ci_batched
        def _sample(model, vocab, tgt, T, n_cands, seed):
            return _ci_batched(model, vocab, tgt["words_"], T, n_cands,
                               device=device, temperature=1.0, top_k=0,
                               seed=seed, use_cd=True)
    else:
        raise ValueError(domain)

    ck = torch.load(ar_ckpt, map_location=device, weights_only=False)
    vocab = ck["vocab"]
    cfg = GPT2Config(**ck["config"])
    model = GPT2LMHeadModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # warm-up
    for _ in range(n_warmup):
        _ = _sample(model, vocab, target_rec, seq_len, 4, 0)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for r in range(n_repeat):
        t0 = time.perf_counter()
        _ = _sample(model, vocab, target_rec, seq_len, n_cands, 42 + r)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  AR+CD run {r+1}/{n_repeat}: {elapsed:.3f}s for {n_cands} cands "
              f"({1000*elapsed/n_cands:.1f} ms/cand)", flush=True)

    return {
        "method": "AR+CD",
        "n_cands": n_cands,
        "n_repeat": n_repeat,
        "times_s": times,
        "median_s": sorted(times)[len(times)//2],
        "per_cand_ms": [1000*t/n_cands for t in times],
        "median_per_cand_ms": 1000 * sorted(times)[len(times)//2] / n_cands,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def time_diffusion_lcvr(diff_ckpt, target_rec, n_cands, seq_len_T, domain, device,
                        n_warmup=1, n_repeat=3,
                        respacing_override=None, use_ddim=False):
    """Time Diffusion+LCVR for n_cands candidates × n_repeat runs.

    If respacing_override is provided (e.g. "50" or "ddim50") it overrides
    the training-time timestep_respacing.  If use_ddim is True the sampler
    is ddim_sample_loop instead of p_sample_loop.
    """
    import argparse as _argparse
    from functools import partial
    from improved_diffusion.script_util import (
        create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict,
    )
    from improved_diffusion.rounding import load_models
    from improved_diffusion.test_util import get_weights, denoised_fn_round
    from improved_diffusion import dist_util

    diff_ckpt = Path(diff_ckpt)
    model_dir = diff_ckpt.parent
    with open(model_dir / "training_args.json") as f:
        training_args = json.load(f)
    defaults = model_and_diffusion_defaults()
    merged = dict(defaults); merged.update(training_args)
    cls = _argparse.Namespace(**merged)
    cls.batch_size = n_cands
    cls.sigma_small = True
    cls.clamp = "clamp"
    cls.top_p = -1.0
    if respacing_override is not None:
        cls.timestep_respacing = respacing_override
    if not hasattr(cls, "clip_denoised"): cls.clip_denoised = False
    if not hasattr(cls, "model_name_or_path"): cls.model_name_or_path = ""
    if not hasattr(cls, "experiment"): cls.experiment = "random"
    if cls.experiment == "random1": cls.experiment = "random"

    dist_util.setup_dist()
    print(f"[diff] loading {diff_ckpt}", flush=True)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cls, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(str(diff_ckpt), map_location="cpu"))
    model.to(device); model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diff] {n_params/1e6:.2f}M params, diffusion_steps="
          f"{diffusion.num_timesteps}", flush=True)

    model2, _tok = load_models(cls.modality, cls.experiment, cls.model_name_or_path,
                               cls.in_channel, str(model_dir))
    if cls.training_mode.startswith("e2e"):
        model2.weight = torch.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model3 = get_weights(model2, cls).to(device)
    denoise_fn = partial(denoised_fn_round, cls, model3)

    T = seq_len_T  # token dim
    C = cls.in_channel
    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    # warm-up (small batch)
    for _ in range(n_warmup):
        _ = sample_fn(model, (4, T, C), clip_denoised=cls.clip_denoised,
                      denoised_fn=denoise_fn, model_kwargs={}, top_p=cls.top_p)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for r in range(n_repeat):
        t0 = time.perf_counter()
        latents = sample_fn(model, (n_cands, T, C), clip_denoised=cls.clip_denoised,
                            denoised_fn=denoise_fn, model_kwargs={}, top_p=cls.top_p)
        # LCVR rounding is just a single forward + masked_argmax (negligible vs sampling)
        with torch.no_grad():
            logits = model.get_logits(latents)
            _ = logits.argmax(dim=-1)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Diff+LCVR run {r+1}/{n_repeat}: {elapsed:.3f}s for {n_cands} cands "
              f"({1000*elapsed/n_cands:.1f} ms/cand)", flush=True)
    return {
        "method": "Diffusion+LCVR",
        "n_cands": n_cands,
        "n_repeat": n_repeat,
        "times_s": times,
        "median_s": sorted(times)[len(times)//2],
        "per_cand_ms": [1000*t/n_cands for t in times],
        "median_per_cand_ms": 1000 * sorted(times)[len(times)//2] / n_cands,
        "n_params": n_params,
        "diffusion_steps": int(diffusion.num_timesteps),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["sonnet", "ci"])
    p.add_argument("--diff_ckpt", required=True)
    p.add_argument("--ar_ckpt", required=True)
    p.add_argument("--target_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_cands", type=int, default=50)
    p.add_argument("--ar_seq_len", type=int, default=200)
    p.add_argument("--n_repeat", type=int, default=3)
    p.add_argument("--target_idx", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--diff_respacing", default=None,
                   help="override timestep_respacing for diffusion (e.g. '50', 'ddim50')")
    p.add_argument("--use_ddim", action="store_true",
                   help="use ddim_sample_loop instead of p_sample_loop")
    args = p.parse_args()

    print(f"[bench] domain={args.domain}, n_cands={args.n_cands}, "
          f"n_repeat={args.n_repeat}, device={args.device}")

    # Load target
    targets = []
    with open(args.target_json) as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    tgt = targets[args.target_idx]
    print(f"[target] idx={args.target_idx}: "
          f"n_tokens={tgt.get('n_tokens', 'n/a')}")

    # Time AR+CD first (lighter)
    print("\n[AR+CD] starting...")
    ar_stats = time_ar_cd(args.ar_ckpt, tgt, args.n_cands, args.ar_seq_len,
                           args.domain, args.device, n_warmup=2, n_repeat=args.n_repeat)

    # Free AR model before loading diffusion
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # Diffusion sequence length = image_size^2
    diff_ckpt = Path(args.diff_ckpt)
    with open(diff_ckpt.parent / "training_args.json") as f:
        ta = json.load(f)
    T = int(ta["image_size"]) ** 2

    print(f"\n[Diff+LCVR] starting, T={T}, respacing={args.diff_respacing}, "
          f"use_ddim={args.use_ddim}...")
    diff_stats = time_diffusion_lcvr(args.diff_ckpt, tgt, args.n_cands, T,
                                      args.domain, args.device,
                                      n_warmup=1, n_repeat=args.n_repeat,
                                      respacing_override=args.diff_respacing,
                                      use_ddim=args.use_ddim)
    diff_stats["respacing"] = args.diff_respacing
    diff_stats["use_ddim"] = args.use_ddim

    speedup = diff_stats["median_per_cand_ms"] / ar_stats["median_per_cand_ms"]
    print(f"\n=== summary ({args.domain}) ===")
    print(f"  AR+CD       : {ar_stats['median_per_cand_ms']:.1f} ms/cand "
          f"({ar_stats['n_params']/1e6:.2f}M params, seq_len={args.ar_seq_len})")
    print(f"  Diff+LCVR   : {diff_stats['median_per_cand_ms']:.1f} ms/cand "
          f"({diff_stats['n_params']/1e6:.2f}M params, "
          f"T={T}, denoise_steps={diff_stats['diffusion_steps']})")
    if speedup >= 1:
        print(f"  AR is {speedup:.2f}x SLOWER than Diffusion per candidate.")
    else:
        print(f"  AR is {1/speedup:.2f}x FASTER than Diffusion per candidate.")

    out = {
        "domain": args.domain,
        "n_cands": args.n_cands,
        "n_repeat": args.n_repeat,
        "target_idx": args.target_idx,
        "device": args.device,
        "ar_cd": ar_stats,
        "diff_lcvr": diff_stats,
        "ar_vs_diff_speedup": speedup,
        "ar_seq_len": args.ar_seq_len,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
