import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torchmetrics.classification import BinaryJaccardIndex

from model import PortrainNetMobileNetV2, PortrainNetMobileNetV3
from util import FocalLoss, ConsistencyLoss
from data.data_aug import Anti_Normalize_Img
from data.datasets import Human


class PortraitNetModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams.update(args)
        self.save_hyperparameters(logger=False)
        if args.model == "mobilenetv2":
            self.model = PortrainNetMobileNetV2(args)
        elif args.model == "mobilenetv3":
            self.model = PortrainNetMobileNetV3(args)
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        self.focal_loss = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        self.consistency_loss = ConsistencyLoss(T=args.T)

        self.val_acc = BinaryJaccardIndex(threshold=0.5)
        self.test_acc = BinaryJaccardIndex(threshold=0.5)

        self.log_image_num = 2

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_acc.reset()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ori, input_aug, boundary, mask = batch
        mask_logits, boundary_logits = self.forward(input_aug)

        loss_mask = self.cross_entropy(mask_logits, mask)
        if self.hparams.use_boundary_loss:
            loss_boundary = (
                self.focal_loss(boundary_logits, boundary)
                * self.hparams.boundary_weight
            )
        else:
            loss_boundary = 0.0

        if self.hparams.use_consistency_loss:
            mask_ori_logits, _ = self.forward(input_ori)
            loss_mask_ori = self.cross_entropy(mask_ori_logits, mask)
            loss_consistency = (
                self.consistency_loss(mask_logits, mask_ori_logits.detach())
                * self.hparams.consistency_weight
            )
        else:
            loss_mask_ori = 0.0
            loss_consistency = 0.0

        loss = loss_mask + loss_mask_ori + loss_boundary + loss_consistency

        # log losses
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss_mask", loss_mask, on_step=False, on_epoch=True)
        self.log("train/loss_mask_ori", loss_mask_ori, on_step=False, on_epoch=True)
        self.log("train/loss_boundary", loss_boundary, on_step=False, on_epoch=True)
        self.log(
            "train/loss_consistency", loss_consistency, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ori, input_aug, boundary, mask = batch
        mask_logits, boundary_logits = self.forward(input_aug)

        loss_mask = self.cross_entropy(mask_logits, mask)
        if self.hparams.use_boundary_loss:
            loss_boundary = (
                self.focal_loss(boundary_logits, boundary)
                * self.hparams.boundary_weight
            )
        else:
            loss_boundary = 0.0

        mask_ori_logits, boundary_ori_logits = self.forward(input_ori)
        loss_mask_ori = self.cross_entropy(mask_ori_logits, mask)
        loss_consistency = (
            self.consistency_loss(mask_logits, mask_ori_logits.detach())
            * self.hparams.consistency_weight
        )

        loss = loss_mask + loss_mask_ori + loss_boundary + loss_consistency

        # log losses
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_mask", loss_mask, on_step=False, on_epoch=True)
        self.log("val/loss_mask_ori", loss_mask_ori, on_step=False, on_epoch=True)
        self.log("val/loss_boundary", loss_boundary, on_step=False, on_epoch=True)
        self.log("val/loss_consistency", loss_consistency, on_step=False, on_epoch=True)
        # log IoU accuracy
        mask_ori = torch.softmax(mask_ori_logits, dim=1)[:, 1, :, :]
        self.val_acc(mask_ori, mask)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        # log several images
        self.log_images(
            "val",
            input_ori,
            input_aug,
            mask,
            mask_logits,
            boundary,
            boundary_logits,
            mask_ori_logits,
            boundary_ori_logits,
            batch_idx,
        )

        return loss

    def test_step(self, batch, batch_idx):
        input_ori, input_aug, boundary, mask = batch
        mask_logits, boundary_logits = self.forward(input_aug)

        loss_mask = self.cross_entropy(mask_logits, mask)
        if self.hparams.use_boundary_loss:
            loss_boundary = (
                self.focal_loss(boundary_logits, boundary)
                * self.hparams.boundary_weight
            )
        else:
            loss_boundary = 0.0

        if self.hparams.use_consistency_loss:
            mask_ori_logits, _ = self.forward(input_ori)
            loss_mask_ori = self.cross_entropy(mask_ori_logits, mask)
            loss_consistency = (
                self.consistency_loss(mask_logits, mask_ori_logits.detach())
                * self.hparams.consistency_weight
            )
        else:
            loss_mask_ori = 0.0
            loss_consistency = 0.0

        loss = loss_mask + loss_mask_ori + loss_boundary + loss_consistency

        # log losses
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_mask", loss_mask, on_step=False, on_epoch=True)
        self.log("test/loss_mask_ori", loss_mask_ori, on_step=False, on_epoch=True)
        self.log("test/loss_boundary", loss_boundary, on_step=False, on_epoch=True)
        self.log(
            "test/loss_consistency", loss_consistency, on_step=False, on_epoch=True
        )
        # log IoU accuracy
        mask_ori = torch.softmax(mask_ori_logits, dim=1)[:, 1, :, :]
        self.test_acc(mask_ori, mask)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )

        return [optimizer], [self.get_lr_scheduler(optimizer)]

    def get_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch):
        return 0.95 ** (epoch // 20)

    def log_images(
        self,
        prefix,
        input_ori,
        input_aug,
        mask,
        mask_logits,
        boundary,
        boundary_logits,
        mask_ori_logits,
        boundary_ori_logits,
        batch_idx,
    ):
        # log some images
        if isinstance(self.logger, TensorBoardLogger):
            if self.hparams.use_boundary_loss:
                self.logger.experiment.add_images(
                    f"{prefix}/boundary_ori_{batch_idx}",
                    self.logits2image(boundary_ori_logits),
                    self.current_epoch,
                )
                self.logger.experiment.add_images(
                    f"{prefix}/boundary_aug_{batch_idx}",
                    self.logits2image(boundary_logits),
                    self.current_epoch,
                )
            self.logger.experiment.add_images(
                f"{prefix}/mask_ori_{batch_idx}",
                self.logits2image(mask_ori_logits),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                f"{prefix}/input_ori_{batch_idx}",
                self.tensor2image(input_ori),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                f"{prefix}/input_aug_{batch_idx}",
                self.tensor2image(input_aug),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                f"{prefix}/true_mask_{batch_idx}",
                self.label2image(mask),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                f"{prefix}/mask_aug_{batch_idx}",
                self.logits2image(mask_logits),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                f"{prefix}/true_boundary_{batch_idx}",
                self.label2image(boundary),
                self.current_epoch,
            )
        elif isinstance(self.logger, WandbLogger):
            import wandb
            from PIL import Image as PILImage

            def concat_images(images):
                if len(images.shape) != 4:
                    raise ValueError("image shape should be (N, C, H, W)")
                image_list = np.split(images, images.shape[0], axis=0)
                image_list = [np.squeeze(image, axis=0) for image in image_list]
                concatenated_image = np.concatenate(image_list, axis=-1).transpose((1, 2, 0))
                image = wandb.Image(concatenated_image)
                return image

            if self.hparams.use_boundary_loss:
                self.logger.experiment.log(
                    {
                        f"{prefix}/boundary_ori_{batch_idx}": concat_images(
                            self.logits2image(boundary_ori_logits)
                        )
                    }
                )
                self.logger.experiment.log(
                    {
                        f"{prefix}/boundary_aug_{batch_idx}": concat_images(
                            self.logits2image(boundary_logits)
                        ),
                    }
                )
            self.logger.experiment.log(
                {
                    f"{prefix}/mask_ori_{batch_idx}": concat_images(
                        self.logits2image(mask_ori_logits)
                    )
                }
            )
            self.logger.experiment.log(
                {
                    f"{prefix}/input_ori_{batch_idx}": concat_images(
                        self.tensor2image(input_ori)
                    ),
                }
            )
            self.logger.experiment.log(
                {
                    f"{prefix}/input_aug_{batch_idx}": concat_images(
                        self.tensor2image(input_aug)
                    ),
                }
            )
            self.logger.experiment.log(
                {
                    f"{prefix}/true_mask_{batch_idx}": concat_images(
                        self.label2image(mask)
                    ),
                }
            )
            self.logger.experiment.log(
                {
                    f"{prefix}/mask_aug_{batch_idx}": concat_images(
                        self.logits2image(mask_logits)
                    ),
                }
            )
            self.logger.experiment.log(
                {
                    f"{prefix}/true_boundary_{batch_idx}": concat_images(
                        self.label2image(boundary)
                    ),
                }
            )

    def tensor2image(self, tensor):
        image = np.uint8(
            (
                Anti_Normalize_Img(
                    np.transpose(
                        tensor[: self.log_image_num].cpu().numpy(), (0, 2, 3, 1)
                    ),
                    scale=1,
                    mean=[103.94, 116.78, 123.68],
                    val=[0.017, 0.017, 0.017],
                )
            )
        )
        image = np.transpose(image, (0, 3, 1, 2))[:, ::-1, :, :]
        return image

    def label2image(self, mask):
        mask = np.uint8(mask[: self.log_image_num].cpu().numpy())
        mask[mask == 255] = 0
        mask = mask * 255
        return mask[:, None, ...]

    def logits2image(self, logits):
        mask = torch.softmax(logits[: self.log_image_num], dim=1)[:, 1, :, :]
        mask = mask.detach().cpu().numpy()
        mask = mask * 255
        return mask[:, None, ...]


class PortraitDataModule(L.LightningDataModule):
    def __init__(self, args):
        super(PortraitDataModule, self).__init__()
        self.exp_args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_args = self.exp_args.copy()
        train_args.istrain = True
        self.trainset = Human(train_args)

        test_args = self.exp_args.copy()
        test_args.istrain = False
        self.testset = Human(test_args)

    def train_dataloader(self):
        return data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.testset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.testset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
