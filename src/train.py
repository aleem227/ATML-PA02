# # src/train.py
# import os, argparse, json, time
# import torch
# from src.utils.config import load_config, set_seed, device_from_cfg, ensure_dir
# from src.datasets.pacs import build_loaders, DOMAINS
# from src.models.backbones import build_backbone, LinearHead
# from src.methods.erm import ERM
# from src.methods.irm import IRM

# def build_method(name, backbone, head, cfg, device):
#     if name.lower() == "erm":
#         return ERM(backbone, head, cfg, device)
#     elif name.lower() == "irm":
#         return IRM(backbone, head, cfg, device)
#     else:
#         raise ValueError(f"Unknown method {name}")

# def main(args):
#     cfg = load_config(args.config)
#     set_seed(cfg["training"].get("seed", 42))
#     device = device_from_cfg(cfg)

#     data_root = cfg["data"]["root"]
#     sources = cfg["data"]["sources"]
#     target  = cfg["data"]["target"]
#     loaders = build_loaders(root=data_root, sources=sources, target=target,
#                             img_size=cfg["data"]["img_size"],
#                             batch_size=cfg["training"]["batch_size"],
#                             num_workers=cfg["training"]["num_workers"])

#     backbone, feat_dim = build_backbone(name=cfg["model"]["backbone"],
#                                         pretrained=cfg["model"]["pretrained"],
#                                         freeze=cfg["model"].get("freeze_backbone", False))
#     head = LinearHead(in_dim=feat_dim, num_classes=cfg["data"]["num_classes"])

#     method = build_method(cfg["method"]["name"], backbone, head, cfg, device)

#     epochs = cfg["training"]["epochs"]
#     outdir = cfg["training"].get("outdir", "runs")
#     ensure_dir(outdir)

#     best_tgt = -1.0
#     for epoch in range(1, epochs+1):
#         if hasattr(method, "set_epoch"): method.set_epoch(epoch)
#         irm_pen_sum, irm_pen_count, last_coef = 0.0, 0, 0.0


#         # train loop
#         method.backbone.train(); method.head.train()
#         for i, batch in enumerate(loaders["train"]["sources_merged"]):
#             stats = method.training_step(batch, i)
#             if "irm_penalty" in stats:
#                 irm_pen_sum += float(stats["irm_penalty"])
#                 irm_pen_count += 1
#                 last_coef = float(stats.get("coef", last_coef))
                

#             if i % 50 == 0:
#                 print(f"[ep {epoch:03d} it {i:04d}] loss={stats['loss']:.4f} erm={stats.get('erm',0):.4f} irm={stats.get('irm_penalty',0):.4f} c={stats.get('coef',0):.3f}")

#         # evaluate
#         val_res = method.evaluate_loader(loaders["val"]["sources_merged"], "val-sources")
#         tgt_res = method.evaluate_loader(loaders["test"]["target"], "target")
#         irm_epoch_avg = (irm_pen_sum / irm_pen_count) if irm_pen_count else float('nan')
#         print(f"[ep {epoch:03d}] val_acc={val_res['acc']:.3f}  target_acc={tgt_res['acc']:.3f}")

#         # checkpoint the best on target (for report only; no target used in training decisions)
#         if tgt_res["acc"] > best_tgt:
#             best_tgt = tgt_res["acc"]
#             torch.save(method.state_dict(), os.path.join(outdir, f"{cfg['method']['name']}_best.pt"))

#     # domain-wise report at the end
#     print("\nPer-domain accuracies:")
#     for dn, ld in loaders["test"]["per_domain"].items():
#         r = method.evaluate_loader(ld, dn)
#         print(f"  {dn:7s}: acc={r['acc']:.3f}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str, required=True)
#     args = ap.parse_args()
#     main(args)
# src/train.py

#########################################################
#######################################################
###########################################################
#############################################################



# import os, argparse, json, time, csv
# import torch
# from src.utils.config import load_config, set_seed, device_from_cfg, ensure_dir
# from src.datasets.pacs import build_loaders, DOMAINS
# from src.models.backbones import build_backbone, LinearHead
# from src.methods.erm import ERM
# from src.methods.irm import IRM
# from src.methods.groupdro import GroupDRO
# from src.methods.sam import SAMMethod


# def build_method(name, backbone, head, cfg, device):
#     name = name.lower()
#     if name == "erm":
#         return ERM(backbone, head, cfg, device)
#     if name == "irm":
#         return IRM(backbone, head, cfg, device)
#     if name == "groupdro":
#         return GroupDRO(backbone, head, cfg, device)
#     if name == "sam":
#         return SAMMethod(backbone, head, cfg, device)
#     raise ValueError(f"Unknown method {name}")

# def main(args):
#     cfg = load_config(args.config)

#     # -------- NEW: CLI overrides --------
#     if args.seed is not None:
#         cfg["training"]["seed"] = int(args.seed)
#     if args.lmbda is not None:
#         cfg.setdefault("method", {})
#         cfg["method"]["lambda"] = float(args.lmbda)
#         cfg["method"]["name"] = "irm"
#     if args.warmup is not None:
#         cfg.setdefault("method", {})
#         cfg["method"]["penalty_warmup"] = int(args.warmup)
#     # -----------------------------------

#     seed = cfg["training"].get("seed", 42)
#     set_seed(seed)
#     device = device_from_cfg(cfg)

#     data_root = cfg["data"]["root"]
#     sources = cfg["data"]["sources"]
#     target  = cfg["data"]["target"]
#     loaders = build_loaders(root=data_root, sources=sources, target=target,
#                             img_size=cfg["data"]["img_size"],
#                             batch_size=cfg["training"]["batch_size"],
#                             num_workers=cfg["training"]["num_workers"],
#                             seed=seed)

#     backbone, feat_dim = build_backbone(name=cfg["model"]["backbone"],
#                                         pretrained=cfg["model"]["pretrained"],
#                                         freeze=cfg["model"].get("freeze_backbone", False))
#     head = LinearHead(in_dim=feat_dim, num_classes=cfg["data"]["num_classes"])

#     method = build_method(cfg["method"]["name"], backbone, head, cfg, device)

#     epochs = int(os.environ.get("EPOCHS_OVERRIDE", cfg["training"]["epochs"]))

#     outdir = cfg["training"].get("outdir", "runs")
#     ensure_dir(outdir)

#     # -------- NEW: per-run file stems --------
#     run_tag = cfg["method"]["name"]
#     if cfg["method"]["name"].lower() == "irm":
#         lam = cfg["method"]["lambda"]
#         run_tag += f"_lam{lam:g}_seed{seed}"
#         if "penalty_warmup" in cfg["method"]:
#             run_tag += f"_wu{cfg['method']['penalty_warmup']}"
#     if args.name_suffix:
#         run_tag += f"_{args.name_suffix}"

#     ckpt_path = os.path.join(outdir, f"{run_tag}_best.pt")
#     csv_path  = os.path.join(outdir, f"{run_tag}_metrics.csv")
#     # -----------------------------------------

#     # -------- NEW: CSV header --------
#     write_header = not os.path.exists(csv_path)
#     csv_file = open(csv_path, "a", newline="")
#     csvw = csv.writer(csv_file)
#     if write_header:
#         csvw.writerow(["epoch","val_acc","target_acc","irm_penalty_avg","coef",
#                     "worst_source_acc","w_Art","w_Cartoon","w_Photo","sharpness"])

#     # ---------------------------------

#     best_tgt = -1.0
#     best_ep  = -1
#     for epoch in range(1, epochs+1):
#         if hasattr(method, "set_epoch"): method.set_epoch(epoch)

#         # accumulators for IRM penalty over epoch
#         irm_pen_sum, irm_pen_count, last_coef = 0.0, 0, 0.0

#         method.backbone.train(); method.head.train()
#         for i, batch in enumerate(loaders["train"]["sources_merged"]):
#             stats = method.training_step(batch, i)

#             if "irm_penalty" in stats:
#                 irm_pen_sum += float(stats["irm_penalty"])
#                 irm_pen_count += 1
#                 last_coef = float(stats.get("coef", last_coef))

#             if i % 50 == 0:
#                 print(f"[ep {epoch:03d} it {i:04d}] loss={stats['loss']:.4f} "
#                       f"erm={stats.get('erm',0):.4f} irm={stats.get('irm_penalty',0):.4f} c={stats.get('coef',0):.3f}")

#         # evaluate
#         val_res = method.evaluate_loader(loaders["val"]["sources_merged"], "val-sources")
#         tgt_res = method.evaluate_loader(loaders["test"]["target"], "target")

#         irm_epoch_avg = (irm_pen_sum / irm_pen_count) if irm_pen_count else float('nan')
#         print(f"[ep {epoch:03d}] val_acc={val_res['acc']:.3f}  target_acc={tgt_res['acc']:.3f}  "
#               f"irm_penalty_avg={irm_epoch_avg:.6f}  coef={last_coef:.3f}")

#         # log to CSV
#         csvw.writerow([epoch, f"{val_res['acc']:.6f}", f"{tgt_res['acc']:.6f}",
#                        f"{irm_epoch_avg:.6f}", f"{last_coef:.6f}"])
#         csv_file.flush()

#         # checkpoint the best on target
#         if tgt_res["acc"] > best_tgt:
#             best_tgt = tgt_res["acc"]; best_ep = epoch
#             torch.save(method.state_dict(), ckpt_path)

#     csv_file.close()

#     # domain-wise report at the end
#     print("\nPer-domain accuracies:")
#     final_domain = {}
#     for dn, ld in loaders["test"]["per_domain"].items():
#         r = method.evaluate_loader(ld, dn)
#         final_domain[dn] = r['acc']
#         print(f"  {dn:7s}: acc={r['acc']:.3f}")

#     # -------- NEW: save run_info.json for aggregator --------
#     run_info = {
#         "method": cfg["method"]["name"],
#         "lambda": cfg["method"].get("lambda", None),
#         "warmup": cfg["method"].get("penalty_warmup", None),
#         "seed": seed,
#         "best_target_acc": best_tgt,
#         "best_epoch": best_ep,
#         "checkpoint": ckpt_path,
#         "metrics_csv": csv_path,
#         "per_domain_final": final_domain,
#         "config": args.config,
#     }
#     with open(os.path.join(outdir, f"{run_tag}_run_info.json"), "w") as f:
#         json.dump(run_info, f, indent=2)
#     # --------------------------------------------

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str, required=True)
#     # NEW overrides:
#     ap.add_argument("--lambda", dest="lmbda", type=float, default=None)
#     ap.add_argument("--seed", type=int, default=None)
#     ap.add_argument("--warmup", type=int, default=None)
#     ap.add_argument("--name_suffix", type=str, default="")
#     args = ap.parse_args()
#     main(args)






#################################
################################
##################################
import os, argparse, json, time, csv
import torch
from src.utils.config import load_config, set_seed, device_from_cfg, ensure_dir
from src.datasets.pacs import build_loaders, DOMAINS
from src.models.backbones import build_backbone, LinearHead
from src.methods.erm import ERM
from src.methods.irm import IRM
from src.methods.groupdro import GroupDRO
from src.methods.sam import SAMMethod


# -------------------------------------------------
# Method Factory
# -------------------------------------------------
def build_method(name, backbone, head, cfg, device):
    name = name.lower()
    if name == "erm":
        return ERM(backbone, head, cfg, device)
    if name == "irm":
        return IRM(backbone, head, cfg, device)
    if name == "groupdro":
        return GroupDRO(backbone, head, cfg, device)
    if name == "sam":
        return SAMMethod(backbone, head, cfg, device)
    raise ValueError(f"Unknown method {name}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    cfg = load_config(args.config)

    # -------- CLI overrides --------
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    if args.lmbda is not None:
        cfg.setdefault("method", {})
        cfg["method"]["lambda"] = float(args.lmbda)
        cfg["method"]["name"] = "irm"
    if args.warmup is not None:
        cfg.setdefault("method", {})
        cfg["method"]["penalty_warmup"] = int(args.warmup)
    # --------------------------------

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    device = device_from_cfg(cfg)

    data_root = cfg["data"]["root"]
    sources = cfg["data"]["sources"]
    target  = cfg["data"]["target"]
    loaders = build_loaders(root=data_root, sources=sources, target=target,
                            img_size=cfg["data"]["img_size"],
                            batch_size=cfg["training"]["batch_size"],
                            num_workers=cfg["training"]["num_workers"],
                            seed=seed)

    backbone, feat_dim = build_backbone(name=cfg["model"]["backbone"],
                                        pretrained=cfg["model"]["pretrained"],
                                        freeze=cfg["model"].get("freeze_backbone", False))
    head = LinearHead(in_dim=feat_dim, num_classes=cfg["data"]["num_classes"])
    method = build_method(cfg["method"]["name"], backbone, head, cfg, device)

    epochs = int(os.environ.get("EPOCHS_OVERRIDE", cfg["training"]["epochs"]))
    outdir = cfg["training"].get("outdir", "runs")
    ensure_dir(outdir)

    # -------- Run naming --------
    run_tag = cfg["method"]["name"]
    if cfg["method"]["name"].lower() == "irm":
        lam = cfg["method"]["lambda"]
        run_tag += f"_lam{lam:g}_seed{seed}"
        if "penalty_warmup" in cfg["method"]:
            run_tag += f"_wu{cfg['method']['penalty_warmup']}"
    if args.name_suffix:
        run_tag += f"_{args.name_suffix}"

    ckpt_path = os.path.join(outdir, f"{run_tag}_best.pt")
    csv_path  = os.path.join(outdir, f"{run_tag}_metrics.csv")
    # -----------------------------------------

    # -------- CSV header --------
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csvw = csv.writer(csv_file)
    if write_header:
        csvw.writerow([
            "epoch", "val_acc", "target_acc", "irm_penalty_avg", "coef",
            "worst_source_acc", "w_Art", "w_Cartoon", "w_Photo", "sharpness"
        ])
    # -----------------------------

    best_tgt, best_ep = -1.0, -1
    for epoch in range(1, epochs + 1):
        if hasattr(method, "set_epoch"):
            method.set_epoch(epoch)

        irm_pen_sum, irm_pen_count, last_coef = 0.0, 0, 0.0
        sharp_sum, sharp_count = 0.0, 0

        method.backbone.train()
        method.head.train()

        for i, batch in enumerate(loaders["train"]["sources_merged"]):
            stats = method.training_step(batch, i)

            # IRM-specific
            if "irm_penalty" in stats:
                irm_pen_sum += float(stats["irm_penalty"])
                irm_pen_count += 1
                last_coef = float(stats.get("coef", last_coef))

            # SAM-specific
            if "sharpness" in stats:
                sharp_sum += float(stats["sharpness"])
                sharp_count += 1

            if i % 50 == 0:
                print(
                    f"[ep {epoch:03d} it {i:04d}] "
                    f"loss={stats['loss']:.4f} "
                    f"erm={stats.get('erm', 0):.4f} "
                    f"irm={stats.get('irm_penalty', 0):.4f} "
                    f"c={stats.get('coef', 0):.3f}"
                )

        # --------------- Evaluation ----------------
        val_res = method.evaluate_loader(loaders["val"]["sources_merged"], "val-sources")
        tgt_res = method.evaluate_loader(loaders["test"]["target"], "target")

        irm_epoch_avg = (irm_pen_sum / irm_pen_count) if irm_pen_count else float('nan')
        sharp_epoch_avg = (sharp_sum / sharp_count) if sharp_count else float('nan')

        # compute worst-source accuracy
        worst_src = 1.0
        for src_dn in cfg["data"]["sources"]:
            r_src = method.evaluate_loader(loaders["test"]["per_domain"][src_dn], f"src-{src_dn}")
            worst_src = min(worst_src, r_src["acc"])

        # fetch group weights (for GroupDRO)
        w_art = w_cartoon = w_photo = float("nan")
        if hasattr(method, "get_group_weights"):
            w = method.get_group_weights()
            w_art = w.get("Art", float("nan"))
            w_cartoon = w.get("Cartoon", float("nan"))
            w_photo = w.get("Photo", float("nan"))

        print(
            f"[ep {epoch:03d}] val_acc={val_res['acc']:.3f}  target_acc={tgt_res['acc']:.3f}  "
            f"irm_penalty_avg={irm_epoch_avg:.6f}  coef={last_coef:.3f}  "
            f"worst_src={worst_src:.3f}  sharp={sharp_epoch_avg:.6f}"
        )

        # --------------- Logging -------------------
        csvw.writerow([
            epoch, f"{val_res['acc']:.6f}", f"{tgt_res['acc']:.6f}",
            f"{irm_epoch_avg:.6f}", f"{last_coef:.6f}",
            f"{worst_src:.6f}", f"{w_art:.6f}", f"{w_cartoon:.6f}", f"{w_photo:.6f}",
            f"{sharp_epoch_avg:.6f}"
        ])
        csv_file.flush()

        # save best checkpoint
        if tgt_res["acc"] > best_tgt:
            best_tgt, best_ep = tgt_res["acc"], epoch
            torch.save(method.state_dict(), ckpt_path)

    csv_file.close()

    # --------------- Final domain report ----------------
    print("\nPer-domain accuracies:")
    final_domain = {}
    for dn, ld in loaders["test"]["per_domain"].items():
        r = method.evaluate_loader(ld, dn)
        final_domain[dn] = r["acc"]
        print(f"  {dn:7s}: acc={r['acc']:.3f}")

    # --------------- Save metadata ----------------------
    run_info = {
        "method": cfg["method"]["name"],
        "lambda": cfg["method"].get("lambda", None),
        "warmup": cfg["method"].get("penalty_warmup", None),
        "seed": seed,
        "best_target_acc": best_tgt,
        "best_epoch": best_ep,
        "checkpoint": ckpt_path,
        "metrics_csv": csv_path,
        "per_domain_final": final_domain,
        "config": args.config,
    }
    with open(os.path.join(outdir, f"{run_tag}_run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--lambda", dest="lmbda", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--name_suffix", type=str, default="")
    args = ap.parse_args()
    main(args)
