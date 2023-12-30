import lightning as L

from Encoder import Encoder
from Decoder import Decoder
from LitAutoEncoder import LitAutoEncoder
from Dataset import *


autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = L.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
