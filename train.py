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
NUM_EPOCHS = 2000
LR_G = 0.0001 # Decreased Generator learning rate
LR_D = 0.0002 # Keep Discriminator learning rate potentially higher initially
BETA1_ADAM = 0.9
BETA2_ADAM = 0.999

# Discriminator Conditional Training: Skip D training if its loss is below this threshold
# Adjust this based on observed D loss behavior. Start with a value that's low but achievable.
D_LOSS_THRESHOLD_SKIP_TRAINING = 0.5 # Example threshold, needs tuning based on training

# Gumbel-Softmax temperature
INITIAL_TEMP = 1.0
MIN_TEMP = 0.1
ANNEALING_STEPS = 10000 # Increased annealing steps


# Gradient Clipping
GRAD_CLIP_NORM = 1.0

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

    discriminator = Discriminator(vocab_size=vocab.vocab_size,
                                embedding_dim=EMBEDDING_DIM,
                                hidden_dim=HIDDEN_DIM,
                                num_classes=NUM_DISC_CLASSES,
                                pad_idx=vocab.pad_idx,
                                dropout_rate=DROPOUT_RATE,
                                input_noise_stddev=DISC_INPUT_NOISE_STDDEV).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1_adam, beta2_adam))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1_adam, beta2_adam))

    print("Starting GAN training...")

    total_steps = 0
    start_time = time.time()

    # Store recent D loss to decide when to skip
    recent_d_loss = float('inf') # Initialize high


    for epoch in range(epochs):
        epoch_start_time = time.time()
        for i, (real_sequences, real_labels) in enumerate(dataloader):
            batch_size = real_sequences.size(0)

            # --- Train Discriminator ---
            # Train D if its recent loss is NOT below the threshold OR it's an early step
            # Always train D in the first few steps to give it a start
            train_d_this_step = (total_steps < 500) or (recent_d_loss > d_loss_threshold_skip)
            # print(f"Step {total_steps}, D Loss: {recent_d_loss:.4f}, Train D: {train_d_this_step}") # Debugging D training skip


            if train_d_this_step:
                discriminator.zero_grad()
                discriminator.train() # Ensure D is in training mode
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
                d_output_fake = discriminator(fake_sequences_soft.detach()) # Detach for D training
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
            generator.train() # Ensure G is in training mode
            discriminator.train() # Keep Discriminator in train mode for G's backward pass

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
            # Log more frequently in early stages?
            log_interval = 100
            if total_steps % log_interval == 0:
                 # It's good practice to evaluate Discriminator on the most recent batch of fake data *before* G's update
                 # We already calculated d_output_fake using fake_sequences_soft.detach()
                 with torch.no_grad():
                     # Temporarily set D and G to eval mode for consistent evaluation metrics
                     discriminator.eval()
                     generator.eval()


                     # Evaluate on the real batch used for training D
                     d_output_real_eval = discriminator(real_sequences)
                     real_preds = torch.argmax(d_output_real_eval, dim=1)
                     real_acc = (real_preds == real_d_labels[:real_preds.size(0)]).float().mean().item()

                     # Evaluate on fake batch used for D training
                     d_output_fake_eval = discriminator(fake_sequences_soft.detach())
                     fake_preds = torch.argmax(d_output_fake_eval, dim=1)
                     fake_acc = (fake_preds == fake_d_labels[:fake_preds.size(0)]).float().mean().item()


                     # Check what D (in eval mode) thinks of fake data (from G's training step)
                     d_output_fake_for_g_eval = discriminator(fake_sequences_soft.detach()) # Re-pass in eval mode
                     fake_d_preds_for_g_eval = torch.argmax(d_output_fake_for_g_eval, dim=1)
                     perc_fake_as_real_pos = (fake_d_preds_for_g_eval == 1).float().mean().item()
                     perc_fake_as_real_neg = (fake_d_preds_for_g_eval == 0).float().mean().item()
                     perc_fake_as_fake = (fake_d_preds_for_g_eval == 2).float().mean().item()

                 # Log current loss values (errD and errG were calculated before no_grad)
                 # If D was skipped this step, errD would be from a previous step or conceptually invalid for this step.
                 # Let's only log D loss when D was actually trained this step for clarity.
                 d_loss_log = errD.item() if train_d_this_step or total_steps < 500 else float('nan') # Log if trained or very early
                 g_loss_log = errG.item()


                 print(f'Epoch [{epoch}/{epochs}], Step [{total_steps}], Temp: {current_temp:.4f}, '
                       f'Loss D: {d_loss_log:.4f}, Loss G: {g_loss_log:.4f}, '
                       f'D Acc Real (eval): {real_acc:.4f}, D Acc Fake (eval): {fake_acc:.4f}, '
                       f'G Output (perc by D - eval): RP: {perc_fake_as_real_pos:.4f}, RN: {perc_fake_as_real_neg:.4f}, Fake: {perc_fake_as_fake:.4f}')


            # --- Set modes for next step ---
            # Modes are explicitly set at the start of D and G training blocks.
            # After the logging block (within no_grad), modes revert.
            # We need D in train mode for the next G step if it's not the D step.
            if not train_d_this_step:
                discriminator.train() # Explicitly ensure D is train for the upcoming G step

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
              d_loss_threshold_skip=D_LOSS_THRESHOLD_SKIP_TRAINING,
              initial_temp=INITIAL_TEMP, min_temp=MIN_TEMP, annealing_steps=ANNEALING_STEPS,
              grad_clip_norm=GRAD_CLIP_NORM,
              generator_save_path=generator_path, discriminator_save_path=discriminator_path)