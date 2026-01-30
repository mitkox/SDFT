"""
Lightweight vLLM HTTP client for external vLLM servers.
This avoids the need to install vLLM locally when using an external server.
"""
import requests
from typing import Optional, Dict, Any, List
import torch


class VLLMClient:
    """HTTP client for external vLLM server that mimics TRL's VLLMClient interface."""
    
    def __init__(self, base_url: str, connection_timeout: int = 600, tokenizer=None):
        self.base_url = base_url.rstrip('/')
        self.timeout = connection_timeout
        self.session = requests.Session()
        self.tokenizer = tokenizer  # Store tokenizer for text->token conversion
        
    def init_communicator(self, device: Optional[torch.device] = None):
        """Initialize communicator (no-op for HTTP client)."""
        pass
    
    def update_named_param(self, name: str, param: torch.Tensor):
        """Update model parameters (no-op for external server)."""
        # External vLLM server doesn't support dynamic parameter updates via HTTP
        pass
    
    def reset_prefix_cache(self):
        """Reset prefix cache (no-op for HTTP client)."""
        pass
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for converting text to token IDs."""
        self.tokenizer = tokenizer
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        images: Optional[List] = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 256,
        truncate_prompt_tokens: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, List]:
        """
        Generate completions using the external vLLM server.
        Returns format compatible with TRL's distillation trainer.
        """
        model_name = kwargs.get("model", "glm-4.7")  # Adjust to match your vLLM server model name
        
        # Build API parameters
        api_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n,
            "logprobs": 1,  # Request logprobs for distillation
            "echo": False,  # Don't echo the prompt
        }
        
        if top_k > 0:
            api_params["top_k"] = top_k
        if repetition_penalty != 1.0:
            api_params["repetition_penalty"] = repetition_penalty
        
        # Override with generation_kwargs if provided
        if generation_kwargs:
            api_params.update(generation_kwargs)
        
        # Process prompts and collect results
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        
        for prompt in prompts:
            for _ in range(n):  # Generate n completions per prompt
                # Use chat completions API for vLLM
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n": 1,  # vLLM chat API handles n=1 per call
                }
                
                if top_k > 0:
                    payload["top_k"] = top_k
                if repetition_penalty != 1.0:
                    payload["repetition_penalty"] = repetition_penalty
                
                # Override with generation_kwargs if provided
                if generation_kwargs:
                    payload.update(generation_kwargs)
                
                try:
                    response = self.session.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        
                        # Extract the completion text from the message
                        completion_text = ""
                        if 'message' in choice and 'content' in choice['message']:
                            completion_text = choice['message']['content']
                        elif 'text' in choice:
                            completion_text = choice['text']
                        
                        if not completion_text:
                            print(f"Warning: No text found in response: {choice}")
                            all_prompt_ids.append([])
                            all_completion_ids.append([1])  # Add dummy token to avoid empty list
                            all_logprobs.append([])
                            continue
                        
                        # Tokenize the completion text if tokenizer is available
                        if self.tokenizer:
                            completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
                        else:
                            # Fallback: return dummy token IDs
                            print("Warning: No tokenizer set, using dummy token IDs")
                            completion_ids = [1] * min(len(completion_text.split()), max_tokens)
                        
                        all_prompt_ids.append([])  # Placeholder - prompt  already tokenized by trainer
                        all_completion_ids.append(completion_ids)
                        
                        # Extract logprobs if available
                        if 'logprobs' in choice and choice['logprobs']:
                            token_logprobs = choice['logprobs'].get('token_logprobs', [])
                            all_logprobs.append(token_logprobs if token_logprobs else [])
                        else:
                            all_logprobs.append([])
                    else:
                        print(f"Warning: No choices in response: {data}")
                        all_prompt_ids.append([])
                        all_completion_ids.append([1])  # Dummy token to avoid empty list
                        all_logprobs.append([])
                        
                except Exception as e:
                    print(f"Error generating completion: {e}")
                    print(f"URL: {self.base_url}/v1/chat/completions")
                    print(f"Payload: {payload}")
                    all_prompt_ids.append([])
                    all_completion_ids.append([])
                    all_logprobs.append([])
        
        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs
        }
