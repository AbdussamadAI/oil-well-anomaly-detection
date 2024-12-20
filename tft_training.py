import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
import torch
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.001,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.2,
            hidden_continuous_size=16,
            loss=SMAPE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
            lstm_layers=2,
        )
        self.dataset = dataset
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.training_step_outputs.append(loss)
        
        return {"loss": loss, "predictions": y_hat, "targets": y}
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("train_epoch_loss", avg_loss, prog_bar=True)
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.validation_step_outputs.append(loss)
        
        return {"val_loss": loss, "predictions": y_hat, "targets": y}
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_epoch_loss", avg_loss, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2, 
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class TFTModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = QuantileLoss()
    
    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        loss = self.loss(y_pred, batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        loss = self.loss(y_pred, batch)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def prepare_data_for_tft(csv_path, time_idx_start=0):
    """
    Prepare data for TFT model
    """
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Drop problematic columns
    columns_to_drop = ['P-JUS-CKGL']  # This column seems to always be problematic
    df = df.drop(columns_to_drop, axis=1)
    
    # Fill missing values
    print("Filling missing values...")
    for col in df.columns:
        if col not in ['binary_class', 'hour', 'day', 'month']:
            df[col] = df[col].fillna(df[col].mean())
            df[col] = df[col].replace([np.inf, -np.inf], df[col].mean())
    
    # Create time index (required by TFT)
    df = df.sort_values(['month', 'day', 'hour'])
    df['time_idx'] = range(len(df))
    
    # Create group_id (required by TFT)
    df['group'] = 'group_1'  # Single group for now
    
    # Convert target to float
    df['binary_class'] = df['binary_class'].astype(float)
    
    # Identify static, time-varying known, and time-varying unknown variables
    static_categoricals = []
    static_reals = []
    time_varying_known_categoricals = []
    time_varying_known_reals = ['time_idx', 'hour', 'day', 'month']
    time_varying_unknown_categoricals = []
    time_varying_unknown_reals = [col for col in df.columns 
                                if col not in ['time_idx', 'group', 'binary_class', 'hour', 'day', 'month']]
    
    print(f"Features being used: {time_varying_unknown_reals}")
    
    # Create TimeSeriesDataSet
    max_encoder_length = 24  # Look back period
    max_prediction_length = 1  # Predict next step
    
    training_cutoff = df['time_idx'].max() - max_prediction_length
    
    dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="binary_class",
        group_ids=['group'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation set
    validation = TimeSeriesDataSet.from_dataset(
        dataset, df, min_prediction_idx=training_cutoff + 1
    )
    
    # Create data loaders
    batch_size = 128
    train_dataloader = dataset.to_dataloader(train=True, batch_size=batch_size)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)
    
    return dataset, train_dataloader, val_dataloader

def train_tft_model(csv_path):
    """
    Train TFT model on the given CSV file
    """
    print(f"\nTraining TFT model on {csv_path}")
    try:
        # Prepare data
        dataset, train_dataloader, val_dataloader = prepare_data_for_tft(csv_path)
        
        # Initialize model
        model = TFTLightningWrapper(dataset)
        
        # Configure trainer with callbacks
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode='min'
        )
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename=f'tft-{{epoch:02d}}-{{val_loss:.2f}}',
            save_top_k=1,
            mode='min',
        )
        
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator='cpu',
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=1,
            val_check_interval=0.25  # Validate every 25% of training epoch
        )
        
        print("\nStarting model training...")
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        # Save the model
        model_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_path = f'models/tft_{model_name}.pth'
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    except Exception as e:
        print(f"Error training model for {csv_path}: {str(e)}")

def main():
    # List of balanced datasets
    datasets = [
        ('Cleaned_data/balanced_drawn_data_cleaned_processed.csv',),
        ('Cleaned_data/balanced_simulated_data_cleaned_processed.csv',),
        ('Cleaned_data/balanced_well_data_cleaned_processed.csv',)
    ]
    
    # Train TFT models on each dataset
    for dataset_path in datasets:
        try:
            model = train_tft_model(dataset_path[0])
            print(f"Successfully trained model for {dataset_path[0]}")
        except Exception as e:
            print(f"Error training model for {dataset_path[0]}: {str(e)}")

if __name__ == "__main__":
    main()
