import torch
import lightning as L
import torch.nn.functional as F


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.view(inputs.size(0), -1)
        encoded = self.encoder(inputs)
        outputs = self.decoder(encoded)
        loss = F.mse_loss(outputs, inputs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
