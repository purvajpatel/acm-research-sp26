import torch
from typing import List, Optional
import re

class VariantGenerator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate_simpler_variants(self, problem: str, num_variants: int = 3, max_attempts: int = 3) -> List[str]:
        prompt = self._create_variant_generation_prompt(problem, num_variants)
        
        for attempt in range(max_attempts):
            try:
                variants = self._generate_and_parse(prompt, num_variants)
                if len(variants) >= num_variants:
                    return variants[:num_variants]
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"Warning: Failed to generate variants: {e}")
                    return self._generate_fallback_variants(problem, num_variants)
                continue
        
        return self._generate_fallback_variants(problem, num_variants)
    
    def _create_variant_generation_prompt(self, problem: str, num_variants: int) -> str:
        return f"""You are an expert at creating progressively simpler versions of integration problems.

Given the integration problem: {problem}

Generate exactly {num_variants} variants that are progressively simpler. Each variant should:
1. Be easier to solve than the previous one
2. Maintain the same general structure/pattern
3. Use simpler functions, coefficients, or substitution patterns
4. Be a valid integration problem

Format your response EXACTLY as:
Variant 1: [problem in LaTeX/math notation]
Variant 2: [problem in LaTeX/math notation]
Variant 3: [problem in LaTeX/math notation]

Example:
Original: ∫ (x^3 + 2x^2 + x) / (x^2 + 1) dx
Variant 1: ∫ (x^2 + x) / (x + 1) dx
Variant 2: ∫ x / (x + 1) dx
Variant 3: ∫ 1 / (x + 1) dx

Now generate variants for: {problem}

Variant 1:"""
    
    def _generate_and_parse(self, prompt: str, num_variants: int) -> List[str]:
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        max_len = 256 if self.device == 'cpu' else 512
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        
        with torch.no_grad():
            try:
                max_tokens = 128 if self.device == 'cpu' else 256
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                else:
                    raise
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        del inputs, outputs
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return self._parse_variants_from_response(response, num_variants)
    
    def _parse_variants_from_response(self, response: str, num_variants: int) -> List[str]:
        variants = []
        
        for i in range(1, num_variants + 1):
            pattern1 = rf'Variant\s+{i}\s*:\s*(.+?)(?=Variant\s+{i+1}\s*:|$)'
            match = re.search(pattern1, response, re.IGNORECASE | re.DOTALL)
            
            if not match:
                pattern2 = rf'{i}[\.\)]\s*(.+?)(?={i+1}[\.\)]\s*|$)'
                match = re.search(pattern2, response, re.IGNORECASE | re.DOTALL)
            
            if match:
                variant = match.group(1).strip()
                variant = ' '.join(variant.split())
                variant = re.sub(r'^(∫|integral|integrate)\s*', '', variant, flags=re.IGNORECASE)
                variant = re.sub(r'\s*dx\s*$', ' dx', variant)
                if variant and len(variant) > 3:
                    variants.append(variant)
        
        return variants
    
    def _generate_fallback_variants(self, problem: str, num_variants: int) -> List[str]:
        return [problem] * num_variants
    
    def _simple_simplify(self, problem: str, level: int) -> Optional[str]:
        return problem
