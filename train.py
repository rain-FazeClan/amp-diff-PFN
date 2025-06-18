# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from utils import vocab, MAX_LEN, NUM_DISC_CLASSES # Import shared utilities
from data_loader import create_gan_dataloader
from model import (Generator, Discriminator, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM,
                   DROPOUT_RATE, DISC_INPUT_NOISE_STDDEV,
                   TRANSFORMER_NUM_HEADS, TRANSFORMER_NUM_LAYERS, TRANSFORMER_FFN_HIDDEN_DIM)


# --- Configuration ---
BATCH_SIZE = 128
NUM_EPOCHS = 2500 # Increased epochs
LR_G = 0.0001
LR_D = 0.0002 # Keep LR for Transformer Discriminator potentially a bit higher

# Discriminator Conditional Training threshold
D_LOSS_THRESHOLD_SKIP_TRAINING = 0.5 # Adjust based on Transformer D loss behavior

# Gumbel-Softmax temperature
INITIAL_TEMP = 1.0
MIN_TEMP = 0.1
ANNEALING_STEPS = 12000 # Increased annealing steps

# Gradient Clipping
GRAD_CLIP_NORM = 1.0

# Model saving
MODELS_DIR = 'models'
GENERATOR_MODEL_FILE = 'generator_model_transformer_d.pth' # New name to distinguish
DISCRIMINATOR_MODEL_FILE = 'discriminator_model_transformer_d.pth' # New name

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for GAN training: {device}")

# Gumbel-Softmax temperature schedule function
def get_temp(step, initial_temp, min_temp, annealing_steps):
    step = min(step, annealing_steps)
    return max(min_temp, initial_temp * (1 - step / annealing_steps))

def train_gan(epochs, batch_size, lr_g, lr_d, beta1_adam, beta2_adam,
              d_loss_threshold_skip, initial_temp, min_temp, annealing_steps,
              grad_clip_norm, generator_save_path, discriminator_save_path):

    dataloader, _ = create_gan_dataloader(batch_size)

    generator = Generator(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          latent_dim=LATENT_DIM,
                          max_len=MAX_LEN,
                          pad_idx=vocab.pad_idx,
                          dropout_rate=DROPOUT_RATE).to(device)

    # Instantiate Discriminator with Transformer parameters
    discriminator = Discriminator(vocab_size=vocab.vocab_size,
                                embedding_dim=EMBEDDING_DIM,
                                hidden_dim=HIDDEN_DIM, # Not used by D now, but keep in constructor for consistency if needed
                                num_classes=NUM_DISC_CLASSES,
                                pad_idx=vocab.pad_idx,
                                dropout_rate=DROPOUT_RATE, # Used by Transformer layers
                                input_noise_stddev=DISC_INPUT_NOISE_STDDEV,
                                transformer_num_heads=TRANSFORMER_NUM_HEADS,
                                transformer_num_layers=TRANSFORMER_NUM_LAYERS,
                                transformer_ffn_hidden_dim=TRANSFORMER_FFN_HIDDEN_DIM).to(device)


    criterion = nn.CrossEntropyLoss()
    # Use Adam with betas common for Transformer training (often 0.9, 0.98 or 0.9, 0.999)
    # Let's use the 0.9, 0.999 we used last time.
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))


    print("Starting GAN training (Transformer Discriminator)...")

    total_steps = 0
    start_time = time.time()

    # Store recent D loss to decide when to skip
    recent_d_loss = float('inf')

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for i, (real_sequences, real_labels) in enumerate(dataloader):
            batch_size = real_sequences.size(0)

            real_sequences = real_sequences.to(device)
            real_labels = real_labels.to(device)

            # --- Train Discriminator ---
            # Train D if its recent loss is NOT below the threshold OR it's an early step
            train_d_this_step = (total_steps < 500) or (recent_d_loss > d_loss_threshold_skip)

            if train_d_this_step:
                real_d_labels = real_labels.to(device)
                discriminator.zero_grad()
                discriminator.train()
                generator.eval() # Set generator to eval mode for D training

                # Real data
                real_sequences = real_sequences.to(device)
                real_d_labels = real_labels.to(device)
                d_output_real = discriminator(real_sequences)
                errD_real = criterion(d_output_real, real_d_labels)

                # Fake data
                noise = torch.randn(batch_size, LATENT_DIM, device=device)
                current_temp = get_temp(total_steps, initial_temp, min_temp, annealing_steps)
                fake_sequences_soft = generator(noise, current_temp)

                # Fake data output for Discriminator training
                d_output_fake = discriminator(fake_sequences_soft.detach()) # Detach
                fake_d_labels = torch.full((batch_size,), 2, dtype=torch.long, device=device)
                errD_fake = criterion(d_output_fake, fake_d_labels)

                # Total discriminator loss
                errD = errD_real + errD_fake

                # Backpropagate and step D
                errD.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)
                optimizer_d.step()

                # Update recent D loss
                recent_d_loss = errD.item()


            # --- Train Generator ---
            generator.zero_grad()
            generator.train()
            # Keep Discriminator in train mode for G's backward pass to work with Transformer
            # Note: Unlike cuDNN RNN, Transformer's backward is generally fine in eval mode too.
            # But keeping D in train here allows its dropout/noise to potentially influence G's gradients (less common).
            # Let's explicitly set D to train mode before the forward pass for G's loss calculation,
            # similar to how we handle Generator mode.
            discriminator.train() # Ensure D is in train mode


            # Generate fake data
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            current_temp = get_temp(total_steps, initial_temp, min_temp, annealing_steps)
            fake_sequences_soft = generator(noise, current_temp)

            # Get Discriminator output for fake data (connected to G's graph)
            d_output_fake_for_g = discriminator(fake_sequences_soft) # D is in train mode

            # Generator's target label: Real Positive (1)
            target_labels_for_g = torch.full((batch_size,), 1, dtype=torch.long, device=device)
            errG = criterion(d_output_fake_for_g, target_labels_for_g)

            # Backpropagate and step G
            errG.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_norm)
            optimizer_g.step()


            total_steps += 1

            # --- Logging ---
            log_interval = 100
            if total_steps % log_interval == 0:
                 with torch.no_grad():
                     discriminator.eval()
                     generator.eval()

                     # Evaluate on the real batch used for training D
                     d_output_real_eval = discriminator(real_sequences)
                     real_preds = torch.argmax(d_output_real_eval, dim=1)
                     real_acc = (real_preds == real_d_labels[:real_preds.size(0)]).float().mean().item()

                     # Evaluate on fake batch used for D training
                     # Pass fake_sequences_soft (detached) through D again in eval mode
                     d_output_fake_eval = discriminator(fake_sequences_soft.detach())
                     fake_preds = torch.argmax(d_output_fake_eval, dim=1)
                     fake_d_labels = torch.full((batch_size,), 2, dtype=torch.long, device=device) # Need labels for fake acc eval
                     fake_acc = (fake_preds == fake_d_labels[:fake_preds.size(0)]).float().mean().item()

                     # Check what D (in eval mode) thinks of fake data
                     d_output_fake_for_g_eval = discriminator(fake_sequences_soft.detach())
                     fake_d_preds_for_g_eval = torch.argmax(d_output_fake_for_g_eval, dim=1)
                     perc_fake_as_real_pos = (fake_d_preds_for_g_eval == 1).float().mean().item()
                     perc_fake_as_real_neg = (fake_d_preds_for_g_eval == 0).float().mean().item()
                     perc_fake_as_fake = (fake_d_preds_for_g_eval == 2).float().mean().item()

                 d_loss_log = errD.item() if train_d_this_step or total_steps < 500 else float('nan')
                 g_loss_log = errG.item()

                 print(f'Epoch [{epoch}/{epochs}], Step [{total_steps}], Temp: {current_temp:.4f}, '
                       f'Loss D: {d_loss_log:.4f}, Loss G: {g_loss_log:.4f}, '
                       f'D Acc Real (eval): {real_acc:.4f}, D Acc Fake (eval): {fake_acc:.4f}, ' # Log Fake Acc
                       f'G Output (perc by D - eval): RP: {perc_fake_as_real_pos:.4f}, RN: {perc_fake_as_real_neg:.4f}, Fake: {perc_fake_as_fake:.4f}')


            # --- Set modes for next step ---
            # Modes are handled within the train_d_this_step block and at the start of G training block.
            # After logging (within no_grad), D is back to its state before the with block.
            # We need D in train mode for the next G step if D was skipped this step.
            if not train_d_this_step:
                discriminator.train()


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
              lr_g=LR_G, lr_d=LR_D, beta1_adam=0.9, beta2_adam=0.999, # Explicitly pass betas
              d_loss_threshold_skip=D_LOSS_THRESHOLD_SKIP_TRAINING,
              initial_temp=INITIAL_TEMP, min_temp=MIN_TEMP, annealing_steps=ANNEALING_STEPS,
              grad_clip_norm=GRAD_CLIP_NORM,
              generator_save_path=generator_path, discriminator_save_path=discriminator_path)