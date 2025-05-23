import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from utils import vocab, MAX_LEN, NUM_DISC_CLASSES, PAD_TOKEN # Import shared utilities
from data_loader import create_gan_dataloader
from model import Generator, Discriminator, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM # Import model definitions and hyperparameters

# --- Configuration ---
BATCH_SIZE = 128
NUM_EPOCHS = 1000 # GANs often require many epochs, starting low for example
LR_G = 0.0001
LR_D = 0.0001
BETA1 = 0.5 # For Adam optimizer
D_TRAIN_RATIO = 1 # Train D this many times per G training step

# Gumbel-Softmax temperature
INITIAL_TEMP = 1.0
MIN_TEMP = 0.1
ANNEALING_STEPS = 2000 # Number of training steps over which to anneal temperature

# Model saving
MODELS_DIR = 'models'
GENERATOR_MODEL_FILE = 'generator_model.pth'
DISCRIMINATOR_MODEL_FILE = 'discriminator_model.pth'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for GAN training: {device}")

# Gumbel-Softmax temperature schedule function
def get_temp(step, initial_temp, min_temp, annealing_steps):
    # Ensure steps don't exceed annealing_steps to prevent negative temp
    step = min(step, annealing_steps)
    return max(min_temp, initial_temp * (1 - step / annealing_steps))

def train_gan(epochs, batch_size, lr_g, lr_d, beta1, d_train_ratio,
              initial_temp, min_temp, annealing_steps,
              generator_save_path, discriminator_save_path):

    # Data loading
    dataloader, _ = create_gan_dataloader(batch_size)

    # Models
    generator = Generator(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          latent_dim=LATENT_DIM,
                          max_len=MAX_LEN,
                          pad_idx=vocab.pad_idx).to(device)

    discriminator = Discriminator(vocab_size=vocab.vocab_size,
                                embedding_dim=EMBEDDING_DIM,
                                hidden_dim=HIDDEN_DIM,
                                num_classes=NUM_DISC_CLASSES,
                                pad_idx=vocab.pad_idx).to(device)

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

    print("Starting GAN training...")

    total_steps = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for i, (real_sequences, real_labels) in enumerate(dataloader):
            batch_size = real_sequences.size(0) # Actual batch size (last batch might be smaller)

            # --- Train Discriminator ---
            discriminator.zero_grad()

            # Real data
            real_sequences = real_sequences.to(device)
            # Map original labels (0, 1) to discriminator classes (0, 1)
            # Ensure real_labels batch matches real_sequences batch size
            real_d_labels = real_labels.to(device) # Already 0 for neg, 1 for pos
            d_output_real = discriminator(real_sequences)
            errD_real = criterion(d_output_real, real_d_labels)


            # Fake data
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            # Get Gumbel-Softmax generated sequence (continuous)
            current_temp = get_temp(total_steps, initial_temp, min_temp, annealing_steps)
            fake_sequences_soft = generator(noise, current_temp)

            # Detach fake data for D training
            d_output_fake = discriminator(fake_sequences_soft.detach())
            # Discriminator's target label for fake data is 2
            fake_d_labels = torch.full((batch_size,), 2, dtype=torch.long, device=device)
            errD_fake = criterion(d_output_fake, fake_d_labels)

            # Total discriminator loss
            errD = errD_real + errD_fake
            errD.backward()
            optimizer_d.step()

            # --- Train Generator ---
            # Train G more or less frequently than D depending on training progress
            if total_steps % d_train_ratio == 0:
                generator.zero_grad()

                # Generate fake data again (this time for G's graph)
                noise = torch.randn(batch_size, LATENT_DIM, device=device)
                fake_sequences_soft = generator(noise, current_temp)

                # Get Discriminator output for fake data
                d_output_fake_for_g = discriminator(fake_sequences_soft)

                # Generator's target label: Trick D into thinking fake is Real Positive (1)
                target_labels_for_g = torch.full((batch_size,), 1, dtype=torch.long, device=device)
                errG = criterion(d_output_fake_for_g, target_labels_for_g)

                errG.backward()
                optimizer_g.step()

            total_steps += 1

            # --- Logging ---
            if total_steps % 100 == 0:
                 with torch.no_grad():
                     # Evaluate Discriminator accuracy
                     real_preds = torch.argmax(d_output_real, dim=1)
                     # Need to handle potential smaller last batch when calculating metrics
                     real_acc = (real_preds == real_d_labels[:real_preds.size(0)]).float().mean().item()

                     fake_preds = torch.argmax(d_output_fake.detach(), dim=1)
                     fake_acc = (fake_preds == fake_d_labels[:fake_preds.size(0)]).float().mean().item()

                     # Check what D thinks of fake data when G trains
                     fake_d_preds_for_g = torch.argmax(d_output_fake_for_g.detach(), dim=1)
                     perc_fake_as_real_pos = (fake_d_preds_for_g == 1).float().mean().item()
                     perc_fake_as_real_neg = (fake_d_preds_for_g == 0).float().mean().item()
                     perc_fake_as_fake = (fake_d_preds_for_g == 2).float().mean().item()

                 print(f'Epoch [{epoch}/{epochs}], Step [{total_steps}], Temp: {current_temp:.4f}, '
                       f'Loss D: {errD.item():.4f}, Loss G: {errG.item():.4f}, '
                       f'D Acc Real: {real_acc:.4f}, D Acc Fake: {fake_acc:.4f}, '
                       f'G Output (perc by D): RP: {perc_fake_as_real_pos:.4f}, RN: {perc_fake_as_real_neg:.4f}, Fake: {perc_fake_as_fake:.4f}')

        epoch_end_time = time.time()
        print(f'Epoch {epoch} finished in {(epoch_end_time - epoch_start_time):.2f} seconds.')

    # Save models after training
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(generator.state_dict(), generator_save_path)
    torch.save(discriminator.state_dict(), discriminator_save_path)
    print(f"\nTraining finished. Models saved to {generator_save_path} and {discriminator_save_path}")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds.")

if __name__ == '__main__':
    generator_path = os.path.join(MODELS_DIR, GENERATOR_MODEL_FILE)
    discriminator_path = os.path.join(MODELS_DIR, DISCRIMINATOR_MODEL_FILE)

    train_gan(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
              lr_g=LR_G, lr_d=LR_D, beta1=BETA1,
              d_train_ratio=D_TRAIN_RATIO,
              initial_temp=INITIAL_TEMP, min_temp=MIN_TEMP, annealing_steps=ANNEALING_STEPS,
              generator_save_path=generator_path, discriminator_save_path=discriminator_path)