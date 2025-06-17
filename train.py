# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from utils import vocab, MAX_LEN, NUM_DISC_CLASSES # Import shared utilities
from data_loader import create_gan_dataloader
# Import updated model definitions and hyperparameters
from model import Generator, Discriminator, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM, DROPOUT_RATE, DISC_INPUT_NOISE_STDDEV

# --- Configuration (Updated Hyperparameters) ---
BATCH_SIZE = 128
NUM_EPOCHS = 2000 # Increased epochs as GANs need longer
LR_G = 0.0002 # Slightly different learning rate often helps
LR_D = 0.0002
BETA1_ADAM = 0.9 # Default Adam beta1 for sequence data
BETA2_ADAM = 0.999 # Default Adam beta2

# D_TRAIN_RATIO = 0.5 means D is trained less often than G.
# We will implement this by training G on every batch,
# and training D on batches where total_steps is even (0, 2, 4...).
D_TRAIN_SKIP = 2 # Train D every D_TRAIN_SKIP generator steps (total_steps) - set to 2 for 0.5 ratio

# Gumbel-Softmax temperature
INITIAL_TEMP = 1.0
MIN_TEMP = 0.1
ANNEALING_STEPS = 5000 # Increased annealing steps

# Gradient Clipping
GRAD_CLIP_NORM = 1.0 # Max norm for gradient clipping (adjust if needed)


# Model saving
MODELS_DIR = 'models'
GENERATOR_MODEL_FILE = 'generator_model.pth'
DISCRIMINATOR_MODEL_FILE = 'discriminator_model.pth'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for GAN training: {device}")

# Gumbel-Softmax temperature schedule function
def get_temp(step, initial_temp, min_temp, annealing_steps):
    step = min(step, annealing_steps)
    return max(min_temp, initial_temp * (1 - step / annealing_steps))

def train_gan(epochs, batch_size, lr_g, lr_d, beta1_adam, beta2_adam, d_train_skip,
              initial_temp, min_temp, annealing_steps, grad_clip_norm,
              generator_save_path, discriminator_save_path):

    # Data loading
    dataloader, _ = create_gan_dataloader(batch_size)

    # Models (Pass dropout_rate and input_noise_stddev)
    generator = Generator(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          latent_dim=LATENT_DIM,
                          max_len=MAX_LEN,
                          pad_idx=vocab.pad_idx,
                          dropout_rate=DROPOUT_RATE).to(device)

    discriminator = Discriminator(vocab_size=vocab.vocab_size,
                                embedding_dim=EMBEDDING_DIM,
                                hidden_dim=HIDDEN_DIM,
                                num_classes=NUM_DISC_CLASSES,
                                pad_idx=vocab.pad_idx,
                                dropout_rate=DROPOUT_RATE,
                                input_noise_stddev=DISC_INPUT_NOISE_STDDEV).to(device)

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()
    # Use default Adam betas except beta1 (using 0.9 now)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1_adam, beta2_adam))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1_adam, beta2_adam))

    print("Starting GAN training...")

    total_steps = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for i, (real_sequences, real_labels) in enumerate(dataloader):
            batch_size = real_sequences.size(0) # Actual batch size

            # --- Train Discriminator ---
            # Only train D if total_steps % D_TRAIN_SKIP == 0
            # We calculate losses every step but only update D periodically
            train_d_this_step = (total_steps % d_train_skip == 0)
            if train_d_this_step:
                discriminator.zero_grad()
                discriminator.train() # Ensure D is in training mode for noise/dropout

            # Real data
            real_sequences = real_sequences.to(device)
            real_d_labels = real_labels.to(device)
            d_output_real = discriminator(real_sequences)
            errD_real = criterion(d_output_real, real_d_labels)

            # Fake data
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            current_temp = get_temp(total_steps, initial_temp, min_temp, annealing_steps)
            # fake_sequences_soft = generator(noise, current_temp) # Generate once

            # When training D, detach fake sequences
            generator.eval() # Set generator to eval mode briefly to avoid G-side dropout issues when generating for D?
                             # Or keep G in train mode and let its internal dropout handle it. Keeping G in train might be better for consistency.
            generator.train() # Let's keep generator in train mode always during its training phase batch

            fake_sequences_soft = generator(noise, current_temp)


            # Fake data output for Discriminator training
            # Need to get this output regardless of whether D is trained this step for logging/errG calculation
            d_output_fake = discriminator(fake_sequences_soft.detach())
            fake_d_labels = torch.full((batch_size,), 2, dtype=torch.long, device=device)
            errD_fake = criterion(d_output_fake, fake_d_labels)

            # Total discriminator loss calculation
            errD = errD_real + errD_fake # errD is always calculated

            # Backpropagate and step D only if training D this step
            if train_d_this_step:
                 errD.backward()
                 # Apply gradient clipping to D
                 torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)
                 optimizer_d.step()


            # --- Train Generator ---
            # G is trained on every step
            generator.zero_grad()
            generator.train() # Ensure G is in training mode

            # Generate fake data again (this time for G's graph)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_sequences_soft = generator(noise, current_temp)

            # Get Discriminator output for fake data (connected to G's graph)
            discriminator.eval() # Set D to eval mode for G training step (turn off noise/dropout)
            d_output_fake_for_g = discriminator(fake_sequences_soft) # Connects back to G

            # Generator's target label: Real Positive (1)
            target_labels_for_g = torch.full((batch_size,), 1, dtype=torch.long, device=device)
            errG = criterion(d_output_fake_for_g, target_labels_for_g)

            # Backpropagate and step G
            errG.backward()
            # Apply gradient clipping to G
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_norm)
            optimizer_g.step()


            total_steps += 1

            # --- Logging ---
            if total_steps % 100 == 0:
                 # It's good practice to evaluate Discriminator on the most recent batch of fake data *before* G's update
                 # We already calculated d_output_fake using fake_sequences_soft.detach()
                 with torch.no_grad():
                     # Ensure D is back in train mode temporarily for evaluation on real data batches
                     discriminator.train() # Evaluate on the same real batch used for training D
                     d_output_real_eval = discriminator(real_sequences) # Re-run forward pass in train mode for consistency
                     real_preds = torch.argmax(d_output_real_eval, dim=1)
                     real_acc = (real_preds == real_d_labels[:real_preds.size(0)]).float().mean().item()

                     # Evaluate on fake batch used for D training (ensure D is in train mode)
                     # d_output_fake was calculated when D was potentially in train mode
                     fake_preds = torch.argmax(d_output_fake.detach(), dim=1)
                     fake_acc = (fake_preds == fake_d_labels[:fake_preds.size(0)]).float().mean().item()


                     # Check what D (in eval mode) thinks of fake data (from G's training step)
                     # d_output_fake_for_g was calculated when D was in eval mode
                     fake_d_preds_for_g = torch.argmax(d_output_fake_for_g.detach(), dim=1)
                     perc_fake_as_real_pos = (fake_d_preds_for_g == 1).float().mean().item()
                     perc_fake_as_real_neg = (fake_d_preds_for_g == 0).float().mean().item()
                     perc_fake_as_fake = (fake_d_preds_for_g == 2).float().mean().item()

                 print(f'Epoch [{epoch}/{epochs}], Step [{total_steps}], Temp: {current_temp:.4f}, '
                       f'Loss D: {errD.item():.4f}, Loss G: {errG.item():.4f}, '
                       f'D Acc Real: {real_acc:.4f}, D Acc Fake: {fake_acc:.4f}, '
                       f'G Output (perc by D - eval): RP: {perc_fake_as_real_pos:.4f}, RN: {perc_fake_as_real_neg:.4f}, Fake: {perc_fake_as_fake:.4f}')

            # --- Set modes for next step ---
            # Discriminator mode is set at the start of D training section
            # Generator mode is set at the start of G training section
            # For logging, temporarily set modes might be needed, but ensure correct modes for backward/step.

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
              lr_g=LR_G, lr_d=LR_D, beta1_adam=BETA1_ADAM, beta2_adam=BETA2_ADAM,
              d_train_skip=D_TRAIN_SKIP,
              initial_temp=INITIAL_TEMP, min_temp=MIN_TEMP, annealing_steps=ANNEALING_STEPS,
              grad_clip_norm=GRAD_CLIP_NORM,
              generator_save_path=generator_path, discriminator_save_path=discriminator_path)