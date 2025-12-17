# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import time

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

from physicsnemo.distributed import DistributedManager
from physicsnemo import Module
from physicsnemo.core import ModelMetaData
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)

# TODO: update with base DiffusionUNet once refactor is complete
from physicsnemo.models.diffusion_unets import SongUNetPosEmbd
from physicsnemo.diffusion.multi_diffusion import RandomPatching2D

# TODO: replace with updated APIs once refactor is complete
from utils import EDMPreconditioner, EDMLoss, DiffusionAdapter


# Compilation settings
torch._dynamo.reset()
torch._dynamo.config.cache_size_limit = 264
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False
torch._logging.set_logs(recompiles=True, graph_breaks=True)


def main():
    # Configuration
    # TODO-NG: need to update with actual parameters
    load_checkpoint_flag = False
    checkpoint_dir = "./checkpoints"
    checkpoint_freq = 10
    log_freq = 10
    max_epochs = 100
    use_apex_flag = False

    # Initialize distributed environment
    DistributedManager.initialize()
    dist = DistributedManager()

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)

    # Initialize Weights & Biases
    checkpoint_path = get_checkpoint_dir(checkpoint_dir, "diffusion_sda")
    if load_checkpoint_flag:
        metadata = {"wandb_id": None}
        load_checkpoint(checkpoint_dir, metadata_dict=metadata)
        wandb_id: str = metadata["wandb_id"]
        resume: str = "must"
        rank_zero_logger.info(f"Resuming wandb run with ID: {wandb_id}")
    else:
        wandb_id, resume = None, None
    initialize_wandb(
        project="DiffusionSDA-Training",
        entity="PhysicsNeMo",
        mode="disabled",
        results_dir="./wandb",
        wandb_id=wandb_id,
        resume=resume,
        save_code=True,
        name=f"train-{timestamp}",
        init_timeout=600,
    )

    # Create model
    # TODO-NG: update with actual resolution of the global domain and number of
    # variables (channels). Then we need to tune some other parameters as well
    # based on these values.
    img_resolution = [128, 256]
    img_channels = 4
    num_grid_channels = 100
    model_backbone = SongUNetPosEmbd(
        img_resolution=img_resolution,
        in_channels=img_channels + num_grid_channels,
        out_channels=img_channels,
        N_grid_channels=num_grid_channels,
        gridtype="learnable",
        model_channels=128,
        channel_mult=[1, 2, 2, 2, 2],
        attn_resolutions=[28],
        use_apex_gn=use_apex_flag,
    ).to(dist.device)
    model = EDMPreconditioner(
        model=DiffusionAdapter(model_backbone),
        sigma_data=1.0,
    )
    rank_zero_logger.info(f"Training model with {model.num_parameters()} parameters.")

    # Setup DDP for multi-GPU training
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # Compile model
    model = torch.compile(model)
    rank_zero_logger.info("Model compiled.")

    # TODO-NG: Create training and validation dataloaders with InfiniteSampler.
    # Requirements:
    # - The data has to be z-score normalized (mean=0, std=1)
    # - Use InfiniteSampler for infinite iteration
    # train_loader = ...
    # val_loader = ...
    # train_iter = iter(train_loader)
    # val_iter = iter(val_loader)
    train_iter = None  # Placeholder
    val_iter = None  # Placeholder

    # Create loss function with multi-diffusion support
    patching = RandomPatching2D(
        img_shape=img_resolution,
        # TODO-NG: update with actual patch size, must be either a power of 2
        # or a multiple of 16
        patch_shape=(16, 16),
        patch_num=4,
    )
    loss_fn = EDMLoss(
        model=model,
        P_mean=0.0,
        P_std=1.0,
        sigma_data=1.0,
        patching=patching,
    )

    # Initialize optimizer.
    if use_apex_flag:
        FusedAdam = getattr(importlib.import_module("apex.optimizers"), "FusedAdam")
        optimizer = FusedAdam(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )

    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-6,
    )

    # Load checkpoint if requested
    loaded_epoch, total_samples_trained = 0, 0
    if dist.world_size > 1:
        torch.distributed.barrier()
    if load_checkpoint_flag:
        metadata = {"total_samples_trained": total_samples_trained}
        loaded_epoch = load_checkpoint(
            checkpoint_path,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
            metadata_dict=metadata,
        )
        total_samples_trained = metadata["total_samples_trained"]
        rank_zero_logger.info(f"Resumed from epoch {loaded_epoch}")

    # Training loop (batch-based with InfiniteSampler)
    rank_zero_logger.info("Training started...")
    model.train()

    # Running average for loss
    loss_running_mean = 0.0
    n_loss_running_mean = 1

    cur_iter = total_samples_trained
    max_iter = 100000  # TODO-NG: set based on training requirements
    tick_start_time = time.time()

    while cur_iter < max_iter:
        # Get next batch from infinite sampler
        x = next(train_iter)
        x = x.to(dist.device)

        # Forward pass
        loss = loss_fn(x).mean()

        # Backward pass
        optimizer.zero_grad(**({} if use_apex_flag else {"set_to_none": True}))
        loss.backward()
        optimizer.step()

        # Update running mean of loss
        loss_val = loss.item()
        loss_running_mean += (loss_val - loss_running_mean) / n_loss_running_mean
        n_loss_running_mean += 1
        cur_iter += 1

        # Periodic logging
        if cur_iter % log_freq == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            if dist.rank == 0:
                wandb.log(
                    {
                        "iter": cur_iter,
                        "loss": loss_val,
                        "loss_running_mean": loss_running_mean,
                        "lr": current_lr,
                    }
                )
            elapsed = time.time() - tick_start_time
            msg = f"iter: {cur_iter}, loss: {loss_running_mean:.3e}, "
            msg += f"lr: {current_lr:.2e}, time: {elapsed:.1f}s"
            rank_zero_logger.info(msg)

            # Reset running mean after logging
            loss_running_mean = 0.0
            n_loss_running_mean = 1
            tick_start_time = time.time()

        # Periodic validation
        if cur_iter % (log_freq * 10) == 0:
            val_loss = validation_step(model, val_iter, loss_fn, dist)
            if dist.rank == 0:
                wandb.log({"val_loss": val_loss, "iter": cur_iter})
            rank_zero_logger.info(f"iter: {cur_iter}, val_loss: {val_loss:.3e}")
            model.train()

        # Periodic checkpoint
        if cur_iter % checkpoint_freq == 0:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    checkpoint_path,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=cur_iter,
                    metadata={
                        "wandb_id": wandb.run.id,
                        "total_samples_trained": cur_iter,
                    },
                )
                rank_zero_logger.info(f"Saved checkpoint at iter {cur_iter}")

    # Cleanup.
    wandb.finish()
    rank_zero_logger.info("Training completed!")


@torch.no_grad()
def validation_step(model, val_iter, loss_fn, dist, num_steps=10):
    """Compute validation loss using running average."""
    model.eval()

    # Running average for validation loss
    loss_running_mean = 0.0
    n_loss_running_mean = 1

    for _ in range(num_steps):
        x = next(val_iter)
        x = x.to(dist.device)

        loss = loss_fn(x).mean()
        loss_val = loss.item()

        # Update running mean
        loss_running_mean += (loss_val - loss_running_mean) / n_loss_running_mean
        n_loss_running_mean += 1

    return loss_running_mean


if __name__ == "__main__":
    main()
