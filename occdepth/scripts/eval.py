import torch
import os
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer
from tqdm import tqdm

from occdepth.models.OccDepth import OccDepth
from occdepth.data.NYU.nyu_dm import NYUDataModule
from occdepth.data.semantic_kitti.kitti_dm import KittiDataModule
from occdepth.data.tartanair.tartanair_dm import TartanAirDataModule
from occdepth.data.sweeper.sweeper_dm import SweeperDataModule

config_path = os.getenv("DATA_CONFIG")


def to_cuda(datas):
    assert isinstance(datas, list)
    for i, data in enumerate(datas):
        datas[i] = data.cuda()


@hydra.main(config_name=config_path)
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    load_strict = config.get("load_strict", False)
    full_scene_size = tuple(config.full_scene_size)

    if config.dataset == "kitti":
        config.batch_size_per_gpu = 1
        data_module = KittiDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            pattern_id=config.pattern_id,
            multi_view_mode=config.multi_view_mode,
            use_stereo_depth_gt=config.use_stereo_depth_gt,
            use_lidar_depth_gt=config.use_lidar_depth_gt,
            data_stereo_depth_root=config.data_stereo_depth_root,
            data_lidar_depth_root=config.data_lidar_depth_root,
        )

    elif config.dataset == "NYU":
        config.batch_size_per_gpu = 1
        data_module = NYUDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            pattern_id=config.pattern_id,
            use_depth_gt=config.use_depth_gt,
        )
    elif config.dataset == "tartanair":
        data_module = TartanAirDataModule(
            config=config,
        )
    elif config.dataset == "sweeper":
        data_module = SweeperDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            frustum_size=config.frustum_size,
            project_scale=config.project_scale,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu),
            pattern_id=config.pattern_id,
            multi_view_mode=config.multi_view_mode,
            use_stereo_depth_gt=config.use_stereo_depth_gt,
            use_lidar_depth_gt=config.use_lidar_depth_gt,
            data_stereo_depth_root=config.data_stereo_depth_root,
            occluded_cls=config.occluded_cls if "occluded_cls" in config else False,
            use_strong_img_aug=config.get("use_strong_img_aug", False),
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    model_path = config.get("model_ckpt") or os.path.join(
        get_original_cwd(), "trained_models", "use_igev_rr.ckpt"
    )
    
    print(f"Loading checkpoint: {model_path}")

    print(
        "##### Max CUDA memory before load model: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )
    model = OccDepth.load_from_checkpoint(
        model_path,
        full_scene_size=full_scene_size,
        config=config,
        strict=load_strict,
    )
    model.cuda()
    model.eval()
    print(
        "##### Max CUDA memory after load model: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )

    data_module.setup()
    val_dataloader = data_module.val_dataloader()

    if config.dataset == "sweeper":
        # Manual eval loop for sweeper: compute metrics + save predictions
        model_tag = os.path.basename(model_path).replace(".ckpt", "")
        output_root = os.path.join(get_original_cwd(), "output", "sweeper", model_tag)
        print(f"##### Saving predictions to {output_root}")

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                batch["img"] = batch["img"].cuda()
                to_cuda(batch["T_velo_2_cam"])
                to_cuda(batch["cam_k"])
                to_cuda(batch["ida_mats"])

                pred = model(batch)
                ssc_logit = pred["ssc_logit"]  # (B, n_classes, X, Y, Z)
                y_pred = torch.softmax(ssc_logit, dim=1).detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                # Update metrics
                y_true = batch["target"].detach().cpu().numpy()
                model.test_metrics.add_batch(y_pred, y_true)

                # Save per-sample predictions (occ result + disparity)
                actual_bs = y_pred.shape[0]
                for i in range(actual_bs):
                    seq = batch["sequence"][i]
                    fid = batch["frame_id"][i]
                    write_path = os.path.join(output_root, seq)
                    os.makedirs(write_path, exist_ok=True)

                    # Occ prediction
                    occ_path = os.path.join(write_path, f"{fid}.npy")
                    np.save(occ_path, y_pred[i].astype(np.uint16))
                    os.chmod(occ_path, 0o666)

        # Print and save metrics
        classes = model.class_names
        stats = model.test_metrics.get_stats()
        print("test======")
        print(
            "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
            )
        )
        print("class IoU: {}, ".format(classes))
        print(
            " ".join(["{:.4f}, "] * len(classes)).format(
                *(stats["iou_ssc"] * 100).tolist()
            )
        )
        print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
        model.test_metrics.reset()

        # Save final results to file (sweeper)
        result_path = os.path.join(
            get_original_cwd(), "trained_models",
            os.path.basename(model_path).replace(".ckpt", "_eval.txt"),
        )
        with open(result_path, "w") as f:
            f.write(f"Precision={stats['precision']*100:.4f}, Recall={stats['recall']*100:.4f}, "
                    f"IoU={stats['iou']*100:.4f}\n")
            f.write(f"mIoU={stats['iou_ssc_mean']*100:.4f}\n")
            f.write("class IoU:\n")
            for cls_name, cls_iou in zip(model.class_names, (stats["iou_ssc"] * 100).tolist()):
                f.write(f"  {cls_name}: {cls_iou:.4f}\n")
        os.chmod(result_path, 0o666)
        # Fix output directory permissions so files are editable outside Docker
        for root_dir, dirs, files in os.walk(output_root):
            for d in dirs:
                os.chmod(os.path.join(root_dir, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root_dir, f), 0o666)
        print(f"##### Results saved to {result_path}")
        return  # skip the trainer + final file-save below
    else:
        trainer = Trainer(
            sync_batchnorm=True, deterministic=True, gpus=config.n_gpus, accelerator="ddp"
        )
        trainer.test(model, test_dataloaders=val_dataloader)

    print(
        "##### Max CUDA memory during all evaluation process: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )

    # Save final results to file
    result_path = os.path.join(
        get_original_cwd(), "trained_models",
        os.path.basename(model_path).replace(".ckpt", "_eval.txt"),
    )
    with open(result_path, "w") as f:
        stats = model.test_metrics.get_stats()
        f.write(f"Precision={stats['precision']*100:.4f}, Recall={stats['recall']*100:.4f}, "
                f"IoU={stats['iou']*100:.4f}\n")
        f.write(f"mIoU={stats['iou_ssc_mean']*100:.4f}\n")
        f.write("class IoU:\n")
        for cls_name, cls_iou in zip(model.class_names, (stats["iou_ssc"] * 100).tolist()):
            f.write(f"  {cls_name}: {cls_iou:.4f}\n")
    print(f"##### Results saved to {result_path}")


if __name__ == "__main__":
    main()
