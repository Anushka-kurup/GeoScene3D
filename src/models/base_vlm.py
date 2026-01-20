"""
Base VLM wrapper for Molmo2
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model


class BaseLLaVA(nn.Module):
    """Wrapper around Molmo2 with optional LoRA"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print(f"Loading Molmo2 model: {config.name}")
        
        # Load model in float16 directly on GPU
        print("Loading model on GPU...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Move to GPU immediately
        print("Moving model to CUDA...")
        self.model = self.model.to('cuda:0')
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            print("Enabling gradient checkpointing...")
            self.model.gradient_checkpointing_enable()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            config.name,
            trust_remote_code=True
        )
        
        print("Molmo2 loaded successfully")
        
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
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def encode(self, img_t1, img_t2, question):
        """Encode images and question to get hidden features"""
        if isinstance(question, str):
            question = [question]
        
        images = self._prepare_images(img_t1, img_t2)
        
        # Molmo2 format: multi-image messages
        messages = []
        for i, q in enumerate(question):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Compare these two images and answer: {q}"},
                    {"type": "image", "image": images[i*2]},
                    {"type": "image", "image": images[i*2+1]},
                ]
            })
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        
        # Get model device - use first parameter's device since model.device might not exist
        model_device = next(self.model.parameters()).device
        
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.amp.autocast('cuda'):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = outputs.hidden_states[-1]
        return hidden_states
    
    def generate(self, inputs_embeds=None, input_ids=None, attention_mask=None,
                 max_new_tokens=512, temperature=0.7, top_p=0.9):
        """Generate text from embeddings or tokens"""
        with torch.amp.autocast('cuda'):
            if inputs_embeds is not None:
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
        
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return generated_text
    
    def forward(self, img_t1, img_t2, question, labels=None):
        """Full forward pass with loss computation"""
        features = self.encode(img_t1, img_t2, question)
        
        # For compatibility with training loop
        return {
            'loss': None,
            'logits': features
        }
    
    def _prepare_images(self, img_t1, img_t2):
        """Convert tensors to list of PIL Images"""
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
            
            images.extend([img1, img2])
        
        return images
