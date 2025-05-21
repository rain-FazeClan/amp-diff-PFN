# train_gan_pt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from data_loader import get_gan_dataloader # GAN DataLoader only for real positive samples
from model.gan_pt import Generator, Discriminator
from utils import NUM_AMINO_ACIDS # Number of actual amino acid characters

def train_gan(data_filepath='data/preprocessed_data.npz', gen_save_path='models/weights/generator_pt.pth', disc_save_path='models/weights/discriminator_pt.pth', epochs=800, batch_size=128, latent_dim=100, log_interval=100):
    """
    训练GAN模型（PyTorch）。

    Args:
        data_filepath (str): 预处理数据文件路径。
        gen_save_path (str): 生成器权重保存路径。
        disc_save_path (str): 判别器权重保存路径。
        epochs (int): 训练轮数。
        batch_size (int): 批次大小。
        latent_dim (int): 生成器输入的噪声向量维度。
        log_interval (int): 打印训练信息的间隔。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    gan_loader, max_sequence_length = get_gan_dataloader(data_filepath, batch_size)

    if gan_loader is None:
        print("加载GAN训练数据失败，退出训练。")
        return

    # Number of amino acid channels for one-hot encoding
    num_amino_acids_one_hot = NUM_AMINO_ACIDS

    # Build models
    generator = Generator(latent_dim, max_sequence_length, num_amino_acids_one_hot).to(device)
    discriminator = Discriminator(max_sequence_length, num_amino_acids_one_hot).to(device)

    # Define optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)) # Common GAN practice: lr=0.0002, beta1=0.5

    # Define loss function (Binary Cross Entropy)
    # Use BCELoss for output probabilities between 0 and 1 from the discriminator
    criterion = nn.BCELoss()

    # Labels for training D and G
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)


    print("开始训练GAN（PyTorch）...")
    for epoch in range(epochs):
        start_time = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for i, real_sequences_one_hot in enumerate(gan_loader):
            real_sequences_one_hot = real_sequences_one_hot.to(device)
            current_batch_size = real_sequences_one_hot.size(0) # Handle last batch if smaller

            # Adjust labels size for the last batch
            if real_labels.size(0) != current_batch_size:
                 real_labels = torch.ones(current_batch_size, 1).to(device)
                 fake_labels = torch.zeros(current_batch_size, 1).to(device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Train with real data
            output_real = discriminator(real_sequences_one_hot)
            loss_real = criterion(output_real, real_labels)
            loss_real.backward() # Compute gradients for real data

            # Train with fake data
            noise = torch.randn(current_batch_size, latent_dim, device=device) # Generate noise
            fake_sequences_probs = generator(noise) # Generate fake sequences (probabilities)

            # Detach generator output from generator computation graph for discriminator training
            output_fake = discriminator(fake_sequences_probs.detach()) # Feed fake probs to D
            loss_fake = criterion(output_fake, fake_labels)
            loss_fake.backward() # Compute gradients for fake data

            # Update Discriminator weights
            loss_D = loss_real + loss_fake
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            # Generate fake sequences again (retain graph for generator update)
            noise = torch.randn(current_batch_size, latent_dim, device=device)
            fake_sequences_probs = generator(noise) # Generate fake sequences

            # Get discriminator output for fake sequences (Generator wants D to say REAL)
            output_fake_for_G = discriminator(fake_sequences_probs)
            loss_G = criterion(output_fake_for_G, real_labels) # Calculate G loss

            # Update Generator weights
            loss_G.backward()
            optimizer_G.step()

            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            num_batches += 1

            # Log progress
            if (i + 1) % log_interval == 0 or i == len(gan_loader) - 1:
                 print(f'  Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(gan_loader)}] | D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}')


        avg_d_loss = epoch_d_loss / num_batches if num_batches > 0 else 0
        avg_g_loss = epoch_g_loss / num_batches if num_batches > 0 else 0
        end_time = time.time()

        print(f'Epoch {epoch+1}/{epochs} completed | Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f} | Time: {end_time-start_time:.2f} sec')

        # Save models periodically
        if (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
             torch.save(generator.state_dict(), gen_save_path.replace('.pth', f'_epoch{epoch+1}.pth'))
             torch.save(discriminator.state_dict(), disc_save_path.replace('.pth', f'_epoch{epoch+1}.pth'))
             print(f"模型权重在 Epoch {epoch+1} 保存到 {gen_save_path.replace('.pth', f'_epoch{epoch+1}.pth')} 等")

    # Save final weights
    torch.save(generator.state_dict(), gen_save_path)
    torch.save(discriminator.state_dict(), disc_save_path)
    print("\nGAN训练完成。")
    print(f"最终生成器权重保存到: {gen_save_path}")
    print(f"最终判别器权重保存到: {disc_save_path}")


if __name__ == '__main__':
    os.makedirs('models/weights', exist_ok=True)
    train_gan()