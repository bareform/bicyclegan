from models import (
    Generator,
    Discriminator,
    Encoder
)

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import torchutils

from .hf_utils import (
    ImageDataset
)
from .gen_utils import(
    weights_init
)

def get_argparser():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["edges2shoes"],
        help="Dataset to train on. Must be one of: `edges2shoes`",
    )
    parser.add_argument(
        "--generator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the generator.",
    )
    parser.add_argument(
        "--num_residual_blocks",
        type=int,
        default=9,
        help="Number of residual blocks to use in the generator.",
    )
    parser.add_argument(
        "--discriminator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the discriminator.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads to use in data loading (default: 4)"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Whether to pin memory for data loading (default: True)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train for (default: 200)",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Height of the input images (default: 256)",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Width of the input images (default: 256)",
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=0.0002,
        help="Learning rate for the generator (default: 0.0002).",
    )
    parser.add_argument(
        "--generator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--generator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=0.0002,
        help="Learning rate for the discriminator (default: 0.0002).",
    )
    parser.add_argument(
        "--discriminator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--discriminator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=8,
        help="Dimension of the latent representation (default: 8).",
    )
    parser.add_argument(
        "--latent_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the latent loss (default: 0.5).",
    )
    parser.add_argument(
        "--kl_divergence_loss_weight",
        type=float,
        default=0.01,
        help="Weight for the KL divergence loss (default: 0.01).",
    )
    parser.add_argument(
        "--pixel_loss_weight",
        type=float,
        default=100.0,
        help="Pixel loss weight (default: 100.0).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to save checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=200,
        help="Interval (in epochs) at which to save model checkpoints (default: 200).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Directory to save intermediate output (default: ./output).",
    )
    parser.add_argument(
        "--save_output_interval",
        type=int,
        default=50,
        help="Number of epochs between saving intermediate output (default: 50).",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=9,
        help="Number of rows in output grid (default: 9).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return parser

def main():
    args = get_argparser().parse_args()
    pad_length = len(str(args.num_epochs))

    torchutils.set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Training on {args.dataset}")
    if args.dataset == "edges2shoes":
        dataset = load_dataset("luethan2025/edges2shoes", split="train")
        transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImageDataset(
            hf_dataset=dataset,
            transform=transform
        )
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            drop_last=True,
        )
        dataset = "edges2shoes"
        img_size = (3, args.image_height, args.image_width)

    G = Generator(
        img_size=img_size,
        in_channels=args.generator_in_channels,
        latent_dim=args.latent_dim,
    )
    D_pair = Discriminator(
        img_size=img_size,
        in_channels=args.discriminator_in_channels,
        is_conditional=True,
    )
    D_img = Discriminator(
        img_size=img_size,
        in_channels=args.discriminator_in_channels,
        is_conditional=False,
    )
    E = Encoder(
        latent_dim=args.latent_dim,
    )

    for model in (G, D_pair, D_img, E):
        model.apply(weights_init)

    G = G.to(device)
    D_pair = D_pair.to(device)
    D_img = D_img.to(device)
    E = E.to(device)

    G_optimizer = optim.Adam(
        G.parameters(),
        lr=args.generator_lr,
        betas=(args.generator_adam_beta1, args.generator_adam_beta2),
    )
    D_pair_optimizer = optim.Adam(
        D_pair.parameters(),
        lr=args.discriminator_lr,
        betas=(args.discriminator_adam_beta1, args.discriminator_adam_beta2),
    )
    D_img_optimizer = optim.Adam(
        D_img.parameters(),
        lr=args.discriminator_lr,
        betas=(args.discriminator_adam_beta1, args.discriminator_adam_beta2),
    )
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=args.generator_lr,
        betas=(args.generator_adam_beta1, args.generator_adam_beta2),
    )

    adversarial_criterion = nn.MSELoss()
    pixel_criterion = nn.L1Loss()
    latent_criterion = nn.L1Loss()

    dtype = torchutils.get_torch_dtype(args.mixed_precision)
    use_amp = (device.type == "cuda") and (dtype != torch.float32)
    use_scaler = (device.type == "cuda") and (dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    if not (os.path.exists(args.checkpoint_dir) and os.path.isdir(args.checkpoint_dir)):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.dataset, "epoch")
    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)
    if not (os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir)):
        os.makedirs(checkpoint_dir, exist_ok=True)

    test_images, _ = next(iter(dataloader))
    test_image_index = torch.randint(test_images.size(0), (1,))
    test_images = test_images[test_image_index].to(device).repeat(args.nrow, 1, 1, 1)
    test_latent = torch.randn(test_images.size(0), args.latent_dim, device=device)

    for epoch in range(args.num_epochs):
        running_G_loss = 0.0
        running_D_pair_loss = 0.0
        running_D_img_loss = 0.0

        G.train()
        D_pair.train()
        D_img.train()
        E.train()

        with tqdm(dataloader, desc="Training", unit="batch") as pbar:
            for realA, realB in pbar:
                realA = realA.to(device, non_blocking=args.pin_memory)
                realB = realB.to(device, non_blocking=args.pin_memory)

                batch_size = realA.size(0)
                D_pair_optimizer.zero_grad(set_to_none=True)
                D_img_optimizer.zero_grad(set_to_none=True)
                G_optimizer.zero_grad(set_to_none=True)
                E_optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=use_amp,
                ):
                    mu, logvar = E(realB)
                    std = torch.exp(0.5 * logvar)
                    z_encoded = mu + torch.randn_like(std) * std
                    fakeB_cvae = G(realA, z_encoded)

                    z_random = torch.randn(
                        batch_size,
                        args.latent_dim,
                        device=device,
                    )
                    fakeB_prior = G(realA, z_random)
                    mu_prior, _ = E(fakeB_prior)

                    fake_pair_cvae = torch.cat((realA, fakeB_cvae), dim=1)
                    fake_pair_prior = torch.cat((realA, fakeB_prior), dim=1)

                    D_pair_cvae_logits = D_pair(fake_pair_cvae)
                    D_pair_prior_logits = D_pair(fake_pair_prior)
                    D_img_cvae_logits = D_img(fakeB_cvae)
                    D_img_prior_logits = D_img(fakeB_prior)

                    adversarial_loss = 0.25 * (
                        adversarial_criterion(
                            D_pair_cvae_logits,
                            torch.ones_like(D_pair_cvae_logits),
                        )
                        + adversarial_criterion(
                            D_pair_prior_logits,
                            torch.ones_like(D_pair_prior_logits),
                        )
                        + adversarial_criterion(
                            D_img_cvae_logits,
                            torch.ones_like(D_img_cvae_logits),
                        )
                        + adversarial_criterion(
                            D_img_prior_logits,
                            torch.ones_like(D_img_prior_logits),
                        )
                    )
                    pixel_loss = pixel_criterion(fakeB_cvae, realB)
                    latent_loss = latent_criterion(mu_prior, z_random)
                    kl_mu = mu.float()
                    kl_logvar = logvar.float()
                    kl_divergence_loss = -0.5 * torch.mean(
                        torch.sum(
                            1 + kl_logvar - kl_mu.pow(2) - kl_logvar.exp(),
                            dim=1,
                        )
                    )
                    G_loss = (
                        adversarial_loss
                        + args.pixel_loss_weight * pixel_loss
                        + args.latent_loss_weight * latent_loss
                        + args.kl_divergence_loss_weight * kl_divergence_loss
                    )

                scaler.scale(G_loss).backward()
                scaler.step(G_optimizer)
                scaler.step(E_optimizer)

                D_pair_optimizer.zero_grad(set_to_none=True)
                D_img_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=use_amp,
                ):
                    real_pair_logits = D_pair(torch.cat((realA, realB), dim=1))
                    fake_pair_cvae_logits = D_pair(fake_pair_cvae.detach())
                    fake_pair_prior_logits = D_pair(fake_pair_prior.detach())
                    real_img_logits = D_img(realB)
                    fake_img_cvae_logits = D_img(fakeB_cvae.detach())
                    fake_img_prior_logits = D_img(fakeB_prior.detach())

                    D_pair_loss = (
                        adversarial_criterion(
                            real_pair_logits,
                            torch.ones_like(real_pair_logits),
                        )
                        + adversarial_criterion(
                            fake_pair_cvae_logits,
                            torch.zeros_like(fake_pair_cvae_logits),
                        )
                        + adversarial_criterion(
                            fake_pair_prior_logits,
                            torch.zeros_like(fake_pair_prior_logits),
                        )
                    ) / 3
                    D_img_loss = (
                        adversarial_criterion(
                            real_img_logits,
                            torch.ones_like(real_img_logits),
                        )
                        + adversarial_criterion(
                            fake_img_cvae_logits,
                            torch.zeros_like(fake_img_cvae_logits),
                        )
                        + adversarial_criterion(
                            fake_img_prior_logits,
                            torch.zeros_like(fake_img_prior_logits),
                        )
                    ) / 3
                    D_loss = D_pair_loss + D_img_loss

                scaler.scale(D_loss).backward()
                scaler.step(D_pair_optimizer)
                scaler.step(D_img_optimizer)
                scaler.update()

                running_G_loss += G_loss.item()
                running_D_pair_loss += D_pair_loss.item()
                running_D_img_loss += D_img_loss.item()

                pbar.set_postfix({
                    "G_loss": f"{G_loss.item():.3f}",
                    "D_pair_loss": f"{D_pair_loss.item():.3f}",
                    "D_img_loss": f"{D_img_loss.item():.3f}",
                })

        average_G_loss = running_G_loss / len(dataloader)
        average_D_pair_loss = running_D_pair_loss / len(dataloader)
        average_D_img_loss = running_D_img_loss / len(dataloader)
        print(f"Epoch: {epoch + 1}/{args.num_epochs}")
        print(
            f"G Loss: {average_G_loss:.5f}, "
            f"D_pair Loss: {average_D_pair_loss:.5f}, "
            f"D_img Loss: {average_D_img_loss:.5f}"
        )

        if (epoch + 1) % args.save_output_interval == 0:
            print("Saving generated images")
            G.eval()
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=use_amp,
            ):
                fake_images = G(test_images, test_latent)
            grid = torchvision.utils.make_grid(
                fake_images,
                nrow=args.nrow,
                normalize=True,
                value_range=(-1, 1),
            )
            torchvision.utils.save_image(
                grid,
                os.path.join(
                    output_dir,
                    f"{epoch + 1:0{pad_length}d}.png",
                ),
            )
            G.train()

        if (epoch + 1) % args.save_checkpoint_interval == 0:
            print("Saving model checkpoints")
            checkpoint = {
                "dataset": args.dataset,
                "G": G.state_dict(),
                "E": E.state_dict(),
                "D_pair": D_pair.state_dict(),
                "D_img": D_img.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "discriminator_in_channels": args.discriminator_in_channels,
                "latent_dim": args.latent_dim,
                "generator_lr": args.generator_lr,
                "discriminator_lr": args.discriminator_lr,
                "G_optimizer": G_optimizer.state_dict(),
                "E_optimizer": E_optimizer.state_dict(),
                "D_pair_optimizer": D_pair_optimizer.state_dict(),
                "D_img_optimizer": D_img_optimizer.state_dict(),
                "image_height": args.image_height,
                "image_width": args.image_width,
            }
            torch.save(
                checkpoint,
                os.path.join(
                    checkpoint_dir,
                    f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth",
                ),
            )
            generator = {
                "dataset": args.dataset,
                "G": G.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "img_size": img_size,
                "latent_dim": args.latent_dim,
            }
            torch.save(
                generator,
                os.path.join(
                    checkpoint_dir,
                    f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth",
                ),
            )

    print("Saving final model checkpoints")
    checkpoint = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "E": E.state_dict(),
        "D_pair": D_pair.state_dict(),
        "D_img": D_img.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "discriminator_in_channels": args.discriminator_in_channels,
        "latent_dim": args.latent_dim,
        "generator_lr": args.generator_lr,
        "discriminator_lr": args.discriminator_lr,
        "G_optimizer": G_optimizer.state_dict(),
        "E_optimizer": E_optimizer.state_dict(),
        "D_pair_optimizer": D_pair_optimizer.state_dict(),
        "D_img_optimizer": D_img_optimizer.state_dict(),
        "image_height": args.image_height,
        "image_width": args.image_width,
    }
    torch.save(
        checkpoint,
        os.path.join(
            checkpoint_dir,
            f"{args.dataset}_checkpoint_{args.num_epochs:0{pad_length}d}.pth",
        ),
    )
    generator = {
        "dataset": args.dataset,
        "G": G.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "img_size": img_size,
        "latent_dim": args.latent_dim,
    }
    torch.save(
        generator,
        os.path.join(
            checkpoint_dir,
            f"{args.dataset}_{args.num_epochs:0{pad_length}d}.pth",
        ),
    )

if __name__ == "__main__":
    main()
