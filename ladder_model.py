import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs"""
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class GlobalContextEncoder(nn.Module):
    """Transformer encoder for maintaining global context"""
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(GlobalContextEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        return output


class LocalReasoningDecoder(nn.Module):
    """Transformer decoder for step-by-step local reasoning"""
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(LocalReasoningDecoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        return output


class LADDERModel(nn.Module):
    """
    LADDER: Language Agent with Dual-Attention for Enhanced Reasoning
    
    Architecture:
    1. Global Context Encoder: Processes input question/context
    2. Local Reasoning Decoder: Generates reasoning steps
    3. Cross-Attention: Bridges global and local representations
    """
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, max_seq_len=256):
        super(LADDERModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Global Context Encoder (for input question/context)
        self.global_encoder = GlobalContextEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Local Reasoning Decoder (for reasoning steps)
        self.local_decoder = LocalReasoningDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Cross-attention alignment score (for interpretability)
        self.alignment_score = nn.Linear(d_model, 1)
        
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass
        
        Args:
            src: Input question/context tokens [batch_size, src_seq_len]
            tgt: Target reasoning tokens [batch_size, tgt_seq_len]
            src_mask: Padding mask for source
            tgt_mask: Padding mask for target
            
        Returns:
            logits: Output logits [batch_size, tgt_seq_len, vocab_size]
            alignment: Cross-attention alignment scores
        """
        # Embed inputs
        src_embed = self.embedding(src) * math.sqrt(self.d_model)
        tgt_embed = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Global Context Encoding
        global_context = self.global_encoder(src_embed, src_mask)
        
        # Generate causal mask for decoder
        tgt_seq_len = tgt.size(1)
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        # Local Reasoning Decoding with Cross-Attention to Global Context
        local_reasoning = self.local_decoder(
            tgt_embed, 
            global_context, 
            tgt_mask=causal_mask,
            memory_mask=src_mask
        )
        
        # Output projection
        logits = self.output_projection(local_reasoning)
        
        # Compute alignment score (for interpretability)
        alignment = torch.sigmoid(self.alignment_score(local_reasoning))
        
        return logits, alignment

    def inference(self, src, max_gen_len=128, temperature=1.0):
        """
        Generate reasoning steps given input context
        
        Args:
            src: Input tokens [batch_size, src_seq_len]
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            generated_ids: Generated token ids
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encode global context
            src_embed = self.embedding(src) * math.sqrt(self.d_model)
            global_context = self.global_encoder(src_embed)
            
            # Start with BOS token (assuming token_id=1)
            generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            for _ in range(max_gen_len):
                # Embed current sequence
                tgt_embed = self.embedding(generated) * math.sqrt(self.d_model)
                
                # Generate causal mask
                tgt_len = generated.size(1)
                causal_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
                
                # Decode
                output = self.local_decoder(tgt_embed, global_context, tgt_mask=causal_mask)
                
                # Project to vocabulary
                logits = self.output_projection(output[:, -1, :])
                
                # Sample next token
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token (assuming token_id=2)
                if (next_token == 2).all():
                    break
            
            return generated


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model architecture
    print("=" * 60)
    print("LADDER Model Architecture Test")
    print("=" * 60)
    
    vocab_size = 10000
    d_model = 256
    batch_size = 4
    src_len = 64
    tgt_len = 32
    
    model = LADDERModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"\nModel Parameters: {count_parameters(model):,}")
    print(f"Model Size: ~{count_parameters(model) * 4 / 1024 / 1024:.2f} MB (float32)\n")
    
    # Test forward pass
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    logits, alignment = model(src, tgt)
    
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Alignment scores shape: {alignment.shape}")
    print("\n✓ Model architecture test passed!")
