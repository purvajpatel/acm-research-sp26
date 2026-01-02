import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        positional_encoding_matrix = torch.zeros(max_len, d_model)
        position_indices = torch.arange(0, max_len, dtype=torch.float)
        position_indices = position_indices.unsqueeze(1)
        
        dimension_indices = torch.arange(0, d_model, 2)
        dimension_indices_float = dimension_indices.float()
        log_base_value = math.log(10000.0)
        division_term_value = -log_base_value / d_model
        multiplication_factor = dimension_indices_float * division_term_value
        div_term = torch.exp(multiplication_factor)
        
        position_times_div_term = position_indices * div_term
        sin_values = torch.sin(position_times_div_term)
        cos_values = torch.cos(position_times_div_term)
        
        positional_encoding_matrix[:, 0::2] = sin_values
        positional_encoding_matrix[:, 1::2] = cos_values
        
        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(0)
        self.register_buffer('pe', positional_encoding_matrix)

    def forward(self, x):
        input_sequence_length = x.size(1)
        positional_encoding_slice = self.pe[:, :input_sequence_length, :]
        output_with_position = x + positional_encoding_slice
        return output_with_position


class GlobalContextEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(GlobalContextEncoder, self).__init__()
        self.positional_encoder = PositionalEncoding(d_model)
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder_stack = nn.TransformerEncoder(
            transformer_encoder_layer, 
            num_layers=num_layers
        )
        self.model_dimension = d_model

    def forward(self, src, src_mask=None):
        source_with_positions = self.positional_encoder(src)
        encoder_output = self.transformer_encoder_stack(
            source_with_positions, 
            src_key_padding_mask=src_mask
        )
        return encoder_output


class LocalReasoningDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(LocalReasoningDecoder, self).__init__()
        self.positional_encoder = PositionalEncoding(d_model)
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder_stack = nn.TransformerDecoder(
            transformer_decoder_layer, 
            num_layers=num_layers
        )
        self.model_dimension = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        target_with_positions = self.positional_encoder(tgt)
        decoder_output = self.transformer_decoder_stack(
            target_with_positions, 
            memory, 
            tgt_mask=tgt_mask, 
            memory_key_padding_mask=memory_mask
        )
        return decoder_output


class LADDERModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, max_seq_len=256):
        super(LADDERModel, self).__init__()
        
        self.model_dimension = d_model
        self.maximum_sequence_length = max_seq_len
        
        self.token_embedding_layer = nn.Embedding(vocab_size, d_model)
        self.positional_encoding_layer = PositionalEncoding(d_model, max_len=max_seq_len)
        
        self.global_context_encoder_module = GlobalContextEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.local_reasoning_decoder_module = LocalReasoningDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.vocabulary_output_projection = nn.Linear(d_model, vocab_size)
        
        self.cross_attention_alignment_scorer = nn.Linear(d_model, 1)
        
        self._init_parameters()

    def _init_parameters(self):
        for parameter in self.parameters():
            parameter_dimension = parameter.dim()
            if parameter_dimension > 1:
                nn.init.xavier_uniform_(parameter)

    def generate_square_subsequent_mask(self, sz):
        ones_matrix = torch.ones(sz, sz)
        upper_triangular_mask = torch.triu(ones_matrix, diagonal=1)
        boolean_mask = upper_triangular_mask.bool()
        return boolean_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        source_embedded = self.token_embedding_layer(src)
        embedding_scale_factor = math.sqrt(self.model_dimension)
        source_embedded_scaled = source_embedded * embedding_scale_factor
        
        target_embedded = self.token_embedding_layer(tgt)
        target_embedded_scaled = target_embedded * embedding_scale_factor
        
        encoded_global_context = self.global_context_encoder_module(
            source_embedded_scaled, 
            src_mask
        )
        
        target_sequence_length = tgt.size(1)
        causal_attention_mask = self.generate_square_subsequent_mask(target_sequence_length)
        device_of_target = tgt.device
        causal_attention_mask = causal_attention_mask.to(device_of_target)
        
        decoded_local_reasoning = self.local_reasoning_decoder_module(
            target_embedded_scaled, 
            encoded_global_context, 
            tgt_mask=causal_attention_mask,
            memory_mask=src_mask
        )
        
        output_logits = self.vocabulary_output_projection(decoded_local_reasoning)
        
        alignment_scores_raw = self.cross_attention_alignment_scorer(decoded_local_reasoning)
        alignment_scores_normalized = torch.sigmoid(alignment_scores_raw)
        
        return output_logits, alignment_scores_normalized

    def inference(self, src, max_gen_len=128, temperature=1.0):
        self.eval()
        with torch.no_grad():
            batch_size_value = src.size(0)
            device_of_source = src.device
            
            source_embedded = self.token_embedding_layer(src)
            embedding_scale_factor = math.sqrt(self.model_dimension)
            source_embedded_scaled = source_embedded * embedding_scale_factor
            
            encoded_global_context = self.global_context_encoder_module(source_embedded_scaled)
            
            beginning_of_sequence_token_id = 1
            generated_sequence = torch.ones(
                batch_size_value, 
                1, 
                dtype=torch.long, 
                device=device_of_source
            )
            
            for generation_step in range(max_gen_len):
                target_embedded = self.token_embedding_layer(generated_sequence)
                target_embedded_scaled = target_embedded * embedding_scale_factor
                
                current_target_length = generated_sequence.size(1)
                causal_attention_mask = self.generate_square_subsequent_mask(current_target_length)
                causal_attention_mask = causal_attention_mask.to(device_of_source)
                
                decoder_output = self.local_reasoning_decoder_module(
                    target_embedded_scaled, 
                    encoded_global_context, 
                    tgt_mask=causal_attention_mask
                )
                
                last_token_output = decoder_output[:, -1, :]
                next_token_logits = self.vocabulary_output_projection(last_token_output)
                
                scaled_logits = next_token_logits / temperature
                token_probabilities = F.softmax(scaled_logits, dim=-1)
                sampled_next_token = torch.multinomial(token_probabilities, num_samples=1)
                
                generated_sequence = torch.cat([generated_sequence, sampled_next_token], dim=1)
                
                end_of_sequence_token_id = 2
                all_tokens_are_eos = (sampled_next_token == end_of_sequence_token_id).all()
                if all_tokens_are_eos:
                    break
            
            return generated_sequence


def count_parameters(model):
    total_parameter_count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter_element_count = parameter.numel()
            total_parameter_count = total_parameter_count + parameter_element_count
    return total_parameter_count


if __name__ == "__main__":
    separator_line = "=" * 60
    print(separator_line)
    print("LADDER Model Architecture Test")
    print(separator_line)
    
    vocabulary_size = 10000
    model_dimension = 256
    batch_size_for_test = 4
    source_sequence_length = 64
    target_sequence_length = 32
    
    test_model = LADDERModel(
        vocab_size=vocabulary_size,
        d_model=model_dimension,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    total_params = count_parameters(test_model)
    print(f"\nModel Parameters: {total_params:,}")
    
    model_size_bytes = total_params * 4
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024
    print(f"Model Size: ~{model_size_mb:.2f} MB (float32)\n")
    
    source_tokens = torch.randint(0, vocabulary_size, (batch_size_for_test, source_sequence_length))
    target_tokens = torch.randint(0, vocabulary_size, (batch_size_for_test, target_sequence_length))
    
    output_logits, alignment_scores = test_model(source_tokens, target_tokens)
    
    print(f"Input shape: {source_tokens.shape}")
    print(f"Target shape: {target_tokens.shape}")
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Alignment scores shape: {alignment_scores.shape}")
    print("\n✓ Model architecture test passed!")
