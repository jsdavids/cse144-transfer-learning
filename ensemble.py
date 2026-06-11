"""Softmax-probability ensemble over several trained backbones.

Loads each model's best-val checkpoint, runs it over the test set to get
per-image class probabilities, averages those probabilities across models
(optionally weighted by each model's validation accuracy), and writes a single
``submission_ensemble.csv``.

Why average PROBABILITIES (not logits or hard votes):
* Probabilities are comparably scaled across architectures; raw logits are not.
* Averaging probabilities is the standard, well-behaved soft ensemble and
  degrades gracefully when one member is uncertain.
* Only include members that individually clear the baseline — a weak member
  drags the average down. Architecturally-diverse members (CNN + transformer)
  help more than near-duplicates.

Usage
-----
    # Ensemble every model that has a saved *_finetuned.pth:
    python ensemble.py

    # Ensemble a specific subset:
    python ensemble.py convnext_small swin_s resnet152

    # Weight each member by its stored validation accuracy:
    python ensemble.py --weight-by-val-acc

Each named model must already have <model>_finetuned.pth (run run_model.py first).
"""

import argparse
import os

import numpy as np

import transfer_lib as tl


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", choices=sorted(tl.MODEL_REGISTRY) + [],
                    help="models to ensemble (default: every one with a saved checkpoint)")
    ap.add_argument("--weight-by-val-acc", action="store_true",
                    help="weight each member's probabilities by its stored val accuracy")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation (original + h-flip) for every member")
    ap.add_argument("--tta-five-crop", action="store_true",
                    help="TTA with 5-crop per view for every member; implies --tta")
    ap.add_argument("--out", default="submission_ensemble.csv")
    return ap.parse_args()


def available_models(requested):
    """Resolve the model list: explicit request, or auto-discover checkpoints."""
    if requested:
        names = requested
    else:
        names = [m for m in sorted(tl.MODEL_REGISTRY)
                 if os.path.exists(tl.Config(model_name=m).ckpt_path)]
    resolved = []
    for m in names:
        cfg = tl.Config(model_name=m)
        if not os.path.exists(cfg.ckpt_path):
            print(f"  SKIP {m}: no checkpoint at {cfg.ckpt_path}")
            continue
        resolved.append(m)
    return resolved


def main():
    args = parse_args()
    names = available_models(args.models)
    if not names:
        raise SystemExit("No trained checkpoints found. Run run_model.py for at "
                         "least one model first.")
    print(f"Ensembling: {names}")

    tl.set_seed()
    sum_probs = None
    ref_ids = None
    scored_ids = None
    total_weight = 0.0
    members = []

    use_tta = args.tta or args.tta_five_crop
    for name in names:
        cfg, model, ckpt = tl.load_model_from_checkpoint(
            name, use_tta=use_tta, tta_five_crop=args.tta_five_crop)
        val_acc = float(ckpt.get("val_acc", 1.0))
        _, eval_tfms = tl.build_transforms(cfg)

        ids, probs, scored = tl.predict_probs(
            cfg, model, eval_tfms, predict_all=True, return_probs=True)

        if ref_ids is None:
            ref_ids, scored_ids = ids, scored
        elif ids != ref_ids:
            raise RuntimeError(
                f"{name} produced a different/ordered id list than the first "
                f"model; cannot align probabilities.")

        weight = val_acc if args.weight_by_val_acc else 1.0
        sum_probs = probs * weight if sum_probs is None else sum_probs + probs * weight
        total_weight += weight
        members.append((name, val_acc, weight))

        # Free GPU memory before the next (possibly large) backbone loads.
        del model
        if tl.USE_CUDA:
            tl.torch.cuda.empty_cache()

    avg_probs = sum_probs / total_weight
    preds = avg_probs.argmax(axis=1)
    rows = [(img_id, int(pred)) for img_id, pred in zip(ref_ids, preds)]

    print("\nEnsemble members:")
    for name, va, w in members:
        print(f"  {name:16s} val_acc={va:.3f} weight={w:.3f}")
    tl.write_submission(args.out, rows, scored_ids)


if __name__ == "__main__":
    main()
