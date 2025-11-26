import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from .gan import Discriminator, FDGANGenerator
from .losses import FDGANLoss, denoising_score_matching_loss
from .score_net import ScoreNet
from .utils import (
    DehazingDataset,
    get_lf_hf,
    prepare_discriminator_input,
    weights_init_normal,
)

if __name__ == "__main__":
    # CONFIG and Hyperparameters
    CONFIG = {
        "data_dir": "/home/est_posgrado_ramsses.delossantos/FDGAN/data/training_subset",
        "batch_size": 6,
        "num_epochs": 100,
        "lr": 2e-6,
        "img_size": (256, 320),
        "num_workers": 16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "./checkpoints/SSM_clipped",
        "resume": True,
        "resume_checkpoint": None,
    }

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # Image Transformations
    img_transform = transforms.Compose(
        [
            transforms.Resize(CONFIG["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


    def save_checkpoint(
        epoch: int,
        generator: nn.Module,
        discriminator: nn.Module,
        score_net: nn.Module,
        opt_G: optim.Optimizer,
        opt_D: optim.Optimizer,
        opt_S: optim.Optimizer,
        save_dir: str,
    ) -> str:
        """Function to save a checkpoint.

        This function saves the state dictionaries of the generator, discriminator,
        score network, and their respective optimizers, along with the current epoch
        and configuration settings.

        Parameters
        ----------
        epoch : int
            Epoch number.
        generator : nn.Module
            Generator model.
        discriminator : nn.Module
            Discriminator model.
        score_net : nn.Module
            Score network model.
        opt_G : optim.Optimizer
            Optimizer for the generator.
        opt_D : optim.Optimizer
            Optimizer for the discriminator.
        opt_S : optim.Optimizer
            Optimizer for the score network.
        save_dir : str
            Directory to save the checkpoint.

        Returns
        -------
        str
            Path to the saved checkpoint file.
        """
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")

        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "score_net_state_dict": score_net.state_dict(),
            "optimizer_G_state_dict": opt_G.state_dict(),
            "optimizer_D_state_dict": opt_D.state_dict(),
            "optimizer_S_state_dict": opt_S.state_dict(),
            "config": CONFIG,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        gen_only_path = os.path.join(save_dir, f"gen_epoch_{epoch}.pth")
        torch.save(generator.state_dict(), gen_only_path)

        return checkpoint_path


    def load_checkpoint(
        checkpoint_path: str,
        generator: nn.Module,
        discriminator: nn.Module,
        score_net: nn.Module,
        opt_G: optim.Optimizer,
        opt_D: optim.Optimizer,
        opt_S: optim.Optimizer,
        device: torch.device,
    ) -> int:
        """Function to load a checkpoint.

        This function loads the state dictionaries of the generator, discriminator,
        score network, and their respective optimizers from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        generator : nn.Module
            Generator model.
        discriminator : nn.Module
            Discriminator model.
        score_net : nn.Module
            Score network model.
        opt_G : optim.Optimizer
            Optimizer for the generator.
        opt_D : optim.Optimizer
            Optimizer for the discriminator.
        opt_S : optim.Optimizer
            Optimizer for the score network.
        device : torch.device
            Device to map the loaded tensors.

        Returns
        -------
        int
            Next epoch to train from.
        """
        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model states
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        score_net.load_state_dict(checkpoint["score_net_state_dict"])

        # Load optimizer states
        opt_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        opt_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        opt_S.load_state_dict(checkpoint["optimizer_S_state_dict"])

        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}")

        return epoch + 1


    def find_latest_checkpoint(save_dir: str) -> str | None:
        """Function to find the latest checkpoint in a directory.

        This function scans the specified directory for checkpoint files
        and returns the path to the most recent one based on epoch number.

        Parameters
        ----------
        save_dir : str
            Directory to search for checkpoint files.

        Returns
        -------
        str or None
            Path to the latest checkpoint file or ``None`` if no checkpoints are found.
        """
        checkpoints = [
            f
            for f in os.listdir(save_dir)
            if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
        ]

        if not checkpoints:
            return None

        # Extract epoch numbers and find the maximum
        epochs = []
        for ckpt in checkpoints:
            try:
                epoch_num = int(ckpt.replace("checkpoint_epoch_", "").replace(".pth", ""))
                epochs.append((epoch_num, ckpt))
            except ValueError:
                continue

        if not epochs:
            return None

        latest_epoch, latest_ckpt = max(epochs, key=lambda x: x[0])
        return os.path.join(save_dir, latest_ckpt)


    def main():
        dataset = DehazingDataset(CONFIG["data_dir"], transform=img_transform)
        print("Cargando dataset con", len(dataset), "imágenes.")
        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
        )

        generator = FDGANGenerator(output_same_size=True).to(CONFIG["device"])

        # Discriminator configuration
        disc_config = [
            {
                "conv": {
                    "in_channels": 9,
                    "out_channels": 64,
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 1,
                    "activation": "leakyrelu",
                    "activation_kwargs": {"negative_slope": 0.2},
                    "use_batch_norm": False,
                }
            },
            {
                "conv": {
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 1,
                    "activation": "leakyrelu",
                    "activation_kwargs": {"negative_slope": 0.2},
                    "use_batch_norm": True,
                }
            },
            {
                "conv": {
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 1,
                    "activation": "leakyrelu",
                    "activation_kwargs": {"negative_slope": 0.2},
                    "use_batch_norm": True,
                }
            },
            {
                "conv": {
                    "in_channels": 256,
                    "out_channels": 512,
                    "kernel_size": 4,
                    "stride": 1,
                    "padding": 1,
                    "activation": "leakyrelu",
                    "activation_kwargs": {"negative_slope": 0.2},
                    "use_batch_norm": True,
                }
            },
            {
                "conv": {
                    "in_channels": 512,
                    "out_channels": 1,
                    "kernel_size": 4,
                    "stride": 1,
                    "padding": 1,
                    "activation": "linear",
                    "use_batch_norm": False,
                }
            },
        ]

        img_height = (
            CONFIG["img_size"][0]
            if isinstance(CONFIG["img_size"], tuple)
            else CONFIG["img_size"]
        )
        img_width = (
            CONFIG["img_size"][1]
            if isinstance(CONFIG["img_size"], tuple)
            else CONFIG["img_size"]
        )
        discriminator = Discriminator(
            img_shape=(9, img_height, img_width), conv_layers_config=disc_config
        ).to(CONFIG["device"])

        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

        score_net = ScoreNet(channels=3).to(CONFIG["device"])
        score_net.apply(weights_init_normal)

        # Optimizers
        opt_G = optim.Adam(generator.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.999))
        opt_D = optim.Adam(discriminator.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.999))
        opt_S = optim.Adam(score_net.parameters(), lr=1e-5, betas=(0.5, 0.999))

        criterion = FDGANLoss().to(CONFIG["device"])

        lambda_score = 0.1

        start_epoch = 0
        # Auto resume
        if CONFIG["resume"]:
            checkpoint_path = CONFIG["resume_checkpoint"]

            if checkpoint_path is None:
                checkpoint_path = find_latest_checkpoint(CONFIG["save_dir"])

            if checkpoint_path:
                start_epoch = load_checkpoint(
                    checkpoint_path,
                    generator,
                    discriminator,
                    score_net,
                    opt_G,
                    opt_D,
                    opt_S,
                    CONFIG["device"],
                )
            else:
                print("No previous checkpoint was founded.\nTraining from scratch..")
        else:
            print("Training from scratch...")

        print(f"Starting training in {CONFIG['device']} with {len(dataset)} images...")
        print(f"Epochs: {start_epoch} - {CONFIG['num_epochs']}")

        for epoch in range(start_epoch, CONFIG["num_epochs"]):
            for i, (hazy, clear) in enumerate(dataloader):
                hazy = hazy.to(CONFIG["device"])
                clear = clear.to(CONFIG["device"])

                # ------- Denoising Score Matching training ------- #
                opt_S.zero_grad()
                loss_dsm = denoising_score_matching_loss(score_net, clear, sigma=0.1)
                loss_dsm.backward()
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
                opt_S.step()

                # ------- FDGAN Training ------- #

                # --- Discriminator training --- #
                opt_D.zero_grad()

                fake = generator(hazy)

                # We get low- and high-frequency components for real and fake images
                real_lf, real_hf = get_lf_hf(clear)
                real_in = prepare_discriminator_input(clear, real_lf, real_hf)

                # We need to detach the fake images to avoid gradients flowing to G
                fake_lf, fake_hf = get_lf_hf(fake.detach())
                fake_in = prepare_discriminator_input(fake.detach(), fake_lf, fake_hf)

                pred_real = discriminator(real_in)
                pred_fake = discriminator(fake_in)

                target_real = torch.ones_like(pred_real)

                loss_d_real = nn.BCEWithLogitsLoss()(pred_real, target_real)

                loss_d_fake = nn.BCEWithLogitsLoss()(pred_fake, torch.zeros_like(pred_fake))

                loss_D = (loss_d_real + loss_d_fake) * 0.5

                loss_D.backward()
                opt_D.step()

                # --- Generator training --- #
                opt_G.zero_grad()

                # We get low- and high-frequency components for fake images without detaching the tensor
                fake_lf_g, fake_hf_g = get_lf_hf(fake)
                fake_in_g = prepare_discriminator_input(fake, fake_lf_g, fake_hf_g)

                pred_fake_g = discriminator(fake_in_g)

                # We are using the FDGAN loss as the generator loss
                loss_G, metrics = criterion(fake, clear, pred_fake_g)

                # We need to freeze the score network parameters during the calculation of the denoising
                # score matching regularization loss
                for param in score_net.parameters():
                    param.requires_grad = False

                # We need to add some noise to the fake images before passing to the score network
                sigma_reg = 0.1
                noise_g = torch.randn_like(fake) * sigma_reg
                fake_perturbed = fake + noise_g

                # The network gives us the estimated noise
                predicted_noise = score_net(fake_perturbed)

                # We calculate the denoising score matching regularization loss as the MSE between
                # the predicted noise and a zero tensor (since we want to push the score
                # towards zero for samples from the model distribution)
                loss_score_reg = F.mse_loss(
                    predicted_noise, torch.zeros_like(predicted_noise)
                )

                # Unfreeze the score network parameters
                for param in score_net.parameters():
                    param.requires_grad = True

                # The total generator loss is given by the FDGAN loss and the denoising score
                # matching regularization term
                loss_G_total = loss_G + (lambda_score * loss_score_reg)

                loss_G_total.backward()
                opt_G.step()

                if i % 10 == 0:
                    print(
                        f"[E{epoch}][B{i}] Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | "
                        f"Loss DSM: {loss_dsm.item():.4f} | Score Reg: {loss_score_reg.item():.4f} | "
                        f"L1: {metrics['l1']:.3f} Adv: {metrics['adv']:.3f}",
                        flush=True,
                    )

            save_checkpoint(
                epoch,
                generator,
                discriminator,
                score_net,
                opt_G,
                opt_D,
                opt_S,
                CONFIG["save_dir"],
            )


    if __name__ == "__main__":
        main()
