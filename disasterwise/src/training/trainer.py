import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

class Trainer:
    """
    Main training loop for disaster detection and prediction models.
    """
    def __init__(self, config, model, train_loader, val_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logging.getLogger('disasterwise')
        
        self.learning_rate = config['model']['parameters']['learning_rate']
        self.epochs = config['model']['parameters']['epochs']
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss() # Placeholder: Adjust based on model type

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        if not self.val_loader:
            return None
            
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f'Validation loss for epoch {epoch+1}: {avg_loss}')
        return avg_loss

    def save_checkpoint(self, epoch, val_loss):
        """
        Saves a training checkpoint including model weights, optimizer state, 
        and the current epoch.
        """
        checkpoint_dir = 'saved_models/checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved at: {checkpoint_path}")

    def save_model(self, filename='final_model.pt'):
        """
        Saves the final model weights after training is complete.
        """
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_path = os.path.join(model_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Final model saved at: {model_path}")

    def run(self):
        self.logger.info(f'Starting training for {self.epochs} epochs on {self.device}...')
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Save checkpoint every epoch
            if val_loss is not None:
                self.save_checkpoint(epoch, val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(filename='best_model.pt')
                    self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            
        self.save_model(filename='latest_model.pt')
        self.logger.info('Training complete!')
