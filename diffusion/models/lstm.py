import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    """
    LSTM-based policy for direct action sequence prediction.
    Input: Condition (State)
    Output: Action Sequence
    """
    def __init__(
        self,
        action_dim,
        obs_dim,
        horizon=16,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0,
    ):
        super(LSTMPolicy, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Condition projection to initial hidden state
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)

        # LSTM cell for sequence generation
        self.lstm = nn.LSTM(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Initial action token (learned)
        self.start_token = nn.Parameter(torch.randn(1, 1, action_dim) * 0.01)

    def forward(self, cond):
        """
        cond: (batch, obs_dim)
        Returns: (batch, action_dim, horizon)
        """
        batch_size = cond.shape[0]

        # Project condition to hidden state
        h0 = self.cond_proj(cond)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        # Start with learned token
        input_seq = self.start_token.repeat(batch_size, 1, 1)

        outputs = []
        for t in range(self.horizon):
            lstm_out, (h0, c0) = self.lstm(input_seq, (h0, c0))
            action_t = self.output_proj(lstm_out[:, -1, :])
            outputs.append(action_t)
            # Feed output back as next input
            input_seq = action_t.unsqueeze(1)

        # Stack outputs: (B, horizon, action_dim) -> (B, action_dim, horizon)
        output = torch.stack(outputs, dim=1).transpose(1, 2)
        return output
