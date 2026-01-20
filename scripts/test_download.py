from transformers import AutoProcessor, AutoModelForCausalLM

print("Downloading processor...")
processor = AutoProcessor.from_pretrained("lmms-lab/LLaVA-NeXT-Video-7B")
print("Processor downloaded successfully!")
print(f"Processor type: {type(processor)}")

print("\nDownloading model...")
model = AutoModelForCausalLM.from_pretrained("lmms-lab/LLaVA-NeXT-Video-7B")
print("Model downloaded successfully!")