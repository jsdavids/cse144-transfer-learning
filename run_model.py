"""Train + evaluate + infer a single backbone, using transfer_lib.

This is the thin per-model driver that replaces the old per-architecture
notebooks. All the real logic lives in transfer_lib.py; here we just pick a
model and (optionally) override hyperparameters.

Examples
--------
    # Train ResNet-152 (default 30 epochs, full fine-tune) end to end:
    python run_model.py resnet152

    # Train the new backbones:
    python run_model.py convnext_small
    python run_model.py swin_s
    python run_model.py resnet50
    python run_model.py efficientnet_b3 --batch-size 8     # 300px input, smaller batch

    # Skip training and just regenerate the submission from a saved checkpoint:
    python run_model.py convnext_small --infer-only

    # Shorter run / disable mixing for an ablation:
    python run_model.py swin_s --epochs 15 --no-mix

Available models: resnet152, resnet50, vit_b16, convnext_small, swin_s,
efficientnet_b3.
"""

import argparse

import transfer_lib as tl


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(tl.MODEL_REGISTRY),
                    help="which backbone to fine-tune")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fine-tune-mode", default="full",
                    choices=["full", "layer4", "last_block", "head"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override (default 16 on GPU / 8 on CPU; use 8 for efficientnet_b3 on 4GB)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42,
                    help="controls the val split + head init; vary it (e.g. 7, 13) to gauge "
                         "run-to-run noise. Non-42 seeds get _seed<N>-suffixed output files.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout prob before the classifier head (e.g. 0.3); 0 disables")
    ap.add_argument("--gradual-unfreeze", action="store_true",
                    help="start head-only, then thaw backbone blocks (head-end first) on a "
                         "schedule; targets the train/val gap under full fine-tuning")
    ap.add_argument("--unfreeze-every", type=int, default=2,
                    help="epochs between thawing successive backbone blocks (with --gradual-unfreeze)")
    ap.add_argument("--no-rotation", action="store_true", help="disable RandomRotation")
    ap.add_argument("--no-mix", action="store_true", help="disable MixUp/CutMix")
    ap.add_argument("--randaugment", action="store_true",
                    help="use RandAugment (replaces per-image rotation + ColorJitter); "
                         "directly targets data scarcity")
    ap.add_argument("--randaugment-magnitude", type=int, default=9,
                    help="RandAugment strength 0-30 (torchvision default 9; lower = milder)")
    ap.add_argument("--randaugment-num-ops", type=int, default=2,
                    help="RandAugment ops applied per image (default 2)")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation: average over original + h-flip views")
    ap.add_argument("--tta-five-crop", action="store_true",
                    help="TTA with 5-crop (4 corners + center) per view; implies --tta")
    ap.add_argument("--infer-only", action="store_true",
                    help="skip training; load the saved checkpoint and write the submission")
    ap.add_argument("--no-plot", action="store_true", help="skip saving the training-curve PNG")
    return ap.parse_args()


def main():
    args = parse_args()
    dropout = args.dropout
    # In infer-only mode the head shape must match the checkpoint, so adopt the
    # checkpoint's stored dropout unless the user explicitly overrode --dropout.
    if args.infer_only and args.dropout == 0.0:
        import torch
        peek = torch.load(tl.Config(model_name=args.model, seed=args.seed,
                                    gradual_unfreeze=args.gradual_unfreeze,
                                    use_randaugment=args.randaugment).ckpt_path,
                          map_location="cpu", weights_only=False)
        dropout = float(peek.get("dropout", 0.0))

    cfg = tl.Config(
        model_name=args.model,
        epochs=args.epochs,
        fine_tune_mode=args.fine_tune_mode,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        dropout=dropout,
        gradual_unfreeze=args.gradual_unfreeze,
        unfreeze_every=args.unfreeze_every,
        use_rotation=not args.no_rotation,
        use_mixup_cutmix=not args.no_mix,
        use_randaugment=args.randaugment,
        randaugment_magnitude=args.randaugment_magnitude,
        randaugment_num_ops=args.randaugment_num_ops,
        use_tta=args.tta or args.tta_five_crop,
        tta_five_crop=args.tta_five_crop,
    )

    print(f"Device: {tl.DEVICE} | model: {cfg.model_name} | "
          f"input {cfg.spec.img_size}px | batch {cfg.resolved_batch_size()} | "
          f"AMP {tl.USE_AMP}")
    tta_desc = ("5-crop+flip" if cfg.tta_five_crop else "flip") if cfg.use_tta else "off"
    gu_desc = f"every {cfg.unfreeze_every}ep" if cfg.gradual_unfreeze else "off"
    ra_desc = (f"m{cfg.randaugment_magnitude}/n{cfg.randaugment_num_ops}"
               if cfg.use_randaugment else "off")
    print(f"Seed: {cfg.seed} | dropout: {cfg.dropout} | TTA: {tta_desc} | "
          f"gradual-unfreeze: {gu_desc} | RandAugment: {ra_desc}")

    # build_loaders seeds from cfg.seed; no separate global set_seed needed here.
    class_names, train_loader, val_loader, eval_tfms = tl.build_loaders(cfg)
    model = tl.build_model(cfg)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Mode '{cfg.fine_tune_mode}': trainable {n_trainable:,}/{n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")

    if args.infer_only:
        tl.load_checkpoint(cfg, model)
    else:
        history, best = tl.train(cfg, model, train_loader, val_loader, class_names)
        if not args.no_plot:
            tl.plot_curves(cfg, history, show=False)
        # Reload the best-val checkpoint before inference (last epoch != best).
        tl.load_checkpoint(cfg, model)

    tl.run_inference(cfg, model, eval_tfms, predict_all=True)


if __name__ == "__main__":
    main()
