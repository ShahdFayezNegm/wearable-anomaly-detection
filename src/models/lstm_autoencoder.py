from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multivariate time-series anomaly detection.

    Input shape:
        (batch_size, sequence_length, n_features)

    Output shape:
        (batch_size, sequence_length, n_features)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be greater than zero."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be greater than zero."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # Compress final hidden representation
        self.encoder_to_latent = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.latent_to_decoder = nn.Linear(
            latent_dim,
            hidden_dim,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            input_dim,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a sequence into a latent representation.
        """

        _, (hidden, _) = self.encoder(x)

        # Last encoder layer
        last_hidden = hidden[-1]

        latent = self.encoder_to_latent(
            last_hidden
        )

        return latent

    def decode(
        self,
        latent: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Decode latent representation back into
        the original sequence.
        """

        decoder_initial = self.latent_to_decoder(
            latent
        )

        # Repeat latent representation across time
        decoder_input = decoder_initial.unsqueeze(1).repeat(
            1,
            sequence_length,
            1,
        )

        decoder_output, _ = self.decoder(
            decoder_input
        )

        reconstruction = self.output_layer(
            decoder_output
        )

        return reconstruction

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, sequence, features)."
            )

        latent = self.encode(x)

        reconstruction = self.decode(
            latent,
            sequence_length=x.size(1),
        )

        return reconstruction
