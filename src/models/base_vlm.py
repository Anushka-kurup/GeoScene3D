"""
Base VLM wrapper for LLaVA-NeXT-Video
Handles loading, encoding, and generation
"""

import torch
import torch.nn as nn
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from peft import LoraConfig, get_peft_model


class BaseLLaVA(nn.Module):
    """
    Wrapper around LLaVA-NeXT-Video with optional LoRA
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print(f"Loading LLaVA model: {config.name}")
        
        # Load pre-trained model
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            config.name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load processor
        self.processor = LlavaNextVideoProcessor.from_pretrained(config.name)
        
        # Freeze if specified
        if config.freeze:
            print("Freezing base VLM parameters")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Add LoRA if specified
        if config.lora.enabled:
            print(f"Adding LoRA with r={config.lora.r}, alpha={config.lora.alpha}")
            lora_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def encode(self, img_t1, img_t2, question):
        """
        Encode images and question to get hidden features
        
        Args:
            img_t1: [B, 3, H, W] or PIL Images
            img_t2: [B, 3, H, W] or PIL Images
            question: str or List[str]
        
        Returns:
            hidden_states: [B, N, hidden_dim]
        """
        # Process inputs
        if isinstance(question, str):
            question = [question]
        
        # Convert tensors to PIL if needed
        images = self._prepare_images(img_t1, img_t2)
        
        # Create prompt
        prompts = [f"USER: <image>\n<image>\n{q}\nASSISTANT:" for q in question]
        
        # Process through processor
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # Get hidden states without generation
        with torch.cuda.amp.autocast():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Return last hidden state
        return outputs.hidden_states[-1]
    
    def generate(self, inputs_embeds=None, input_ids=None, attention_mask=None, 
                 max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate text from embeddings or tokens
        
        Args:
            inputs_embeds: [B, N, hidden_dim] (optional)
            input_ids: [B, N] (optional)
            attention_mask: [B, N] (optional)
            max_new_tokens: int
            temperature: float
            top_p: float
        
        Returns:
            generated_text: List[str]
        """
        with torch.cuda.amp.autocast():
            if inputs_embeds is not None:
                # Generate from embeddings
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            else:
                # Generate from tokens
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
        
        # Decode
        generated_text = self.processor.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def forward(self, img_t1, img_t2, question, labels=None):
        """
        Full forward pass with loss computation
        
        Args:
            img_t1: [B, 3, H, W]
            img_t2: [B, 3, H, W]
            question: str or List[str]
            labels: [B, L] (optional)
        
        Returns:
            dict with 'loss' and 'logits'
        """
        # Prepare inputs
        if isinstance(question, str):
            question = [question]
        
        images = self._prepare_images(img_t1, img_t2)
        prompts = [f"USER: <image>\n<image>\n{q}\nASSISTANT:" for q in question]
        
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # Add labels if provided
        if labels is not None:
            inputs['labels'] = labels
        
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs)
        
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }
    
    def _prepare_images(self, img_t1, img_t2):
        """
        Convert tensors to list of PIL Images
        """
        from PIL import Image
        import torchvision.transforms as T
        
        to_pil = T.ToPILImage()
        
        images = []
        batch_size = img_t1.shape[0] if isinstance(img_t1, torch.Tensor) else 1
        
        for i in range(batch_size):
            if isinstance(img_t1, torch.Tensor):
                img1 = to_pil(img_t1[i].cpu())
                img2 = to_pil(img_t2[i].cpu())
            else:
                img1 = img_t1[i] if isinstance(img_t1, list) else img_t1
                img2 = img_t2[i] if isinstance(img_t2, list) else img_t2
            
            images.append([img1, img2])
        
        # Flatten for processor
        return [img for pair in images for img in pair]