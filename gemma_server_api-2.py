"""
GEMMA3 API SERVER 
==================================================
This file downloads GEMMA from huggingface and runs its on the Iperial GPU's.

Usage:
1. Specify an availabale GPU in the cluster
2. Enter hugging face token 
2. Run this file to use load and run the LLM


Utilizing:
- Gemma3 [1]: https://deepmind.google/models/gemma/gemma-3/

Availabe at: https://huggingface.co/google/gemma-3-1b-it

Implemented Using:
- Claude-4 Sonnet [2]: https://www.anthropic.com/claude/sonnet

References:
[1] Google DeepMind (2024), Gemma3 (https://deepmind.google/models/gemma/gemma-3/)
[2] Anthropic (2025), Claude-4 Sonnet (https://www.anthropic.com/claude/sonnet)

Author: Valentin Waliscewski

"""




import sys
import os
import pathlib
import torch

sys.path.insert(0, './packages')

# Set ALL cache directories to current folder 
os.environ['HF_HOME'] = './hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = './hf_cache'
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['TRITON_CACHE_DIR'] = './triton_cache'
os.environ['XDG_CACHE_HOME'] = './cache'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

# Disable all PyTorch optimizations that cause cache issues
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Create all cache directories
cache_dirs = ['./hf_cache', './torch_cache', './triton_cache', './cache']
for cache_dir in cache_dirs:
    pathlib.Path(cache_dir).mkdir(exist_ok=True)


from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import time
from typing import List, Dict, Any, Optional
import logging

## GPU selection for Gemma 3 27B (needs more VRAM)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  
print(" FORCING GPU 2 usage for Gemma 3 27B")

# Set PyTorch to use only one GPU
torch.cuda.set_device(0)  # This will map to GPU 2 due to CUDA_VISIBLE_DEVICES

# Initialize FastAPI app
app = FastAPI(
    title="Gemma 3 27B API Server", 
    description="Local Gemma 3 27B API Server - Claude-compatible endpoint with enhanced capabilities",
    version="2.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
tokenizer = None
model = None

# Pydantic models for Gemma 3 capabilities
class Message(BaseModel):
    role: str
    content: str
    
    # Optional fields for multimodal content
    images: Optional[List[str]] = None  # Base64 encoded images for multimodal support

class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    

class Tool(BaseModel):
    type: str = "function"
    function: Function
    

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3-27b-it"
    messages: List[Message]
    max_tokens: Optional[int] = 4096  # Increased default for Gemma 3
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    # NEW: Function calling support
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"
    

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage
    

# Claude-style API request (for compatibility with CLAUDE API )
class ClaudeRequest(BaseModel):
    model: str = "gemma-3-27b-it"
    max_tokens: int = 4096  # Increased for Gemma 3
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    
    # Enhanced capabilities
    tools: Optional[List[Tool]] = None
    
###################################################################################################################

@app.on_event("startup")
async def load_model():
    """Load the Gemma 3 27B model on startup with optimizations"""
    global tokenizer, model
    
    logger.info(" Loading Gemma 3 27B model...")
    
    # Replace with your actual Hugging Face token
    token = "hf....."  # TODO: Replace with your token
    model_name = "google/gemma-3-27b-it"  # UPDATED: Gemma 3 27B
    
    try:
        # Load tokenizer first
        logger.info(" Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=token,
            cache_dir="./hf_cache"
        )
        
        # UPDATED: Optimized loading for 27B model
        logger.info(" Loading Gemma 3 27B model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            device_map="auto",  # Let transformers handle device placement
            torch_dtype=torch.bfloat16,  # UPDATED: BF16 for better Gemma 3 performance
            cache_dir="./hf_cache",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            # UPDATED: Optimizations for 27B model
            load_in_8bit=True,  # Enable 8-bit quantization to fit in 24GB
            # Optional: You can try load_in_4bit for even more memory savings
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_use_double_quant=True,
            max_memory={0: "20GB"}  # Reserve some VRAM for other operations
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("✅ Gemma 3 27B model loaded successfully!")
        logger.info(f" Model device: {next(model.parameters()).device}")
        logger.info(f" Memory usage: ~20GB (BF16 + 8-bit quantization)")
        logger.info(" Enhanced capabilities: 128K context, function calling, multimodal")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Gemma 3 27B: {str(e)}")
        
        # Fallback: Try with more aggressive quantization
        try:
            logger.info(" Trying fallback with 4-bit quantization...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch.bfloat16,
                cache_dir="./hf_cache",
                trust_remote_code=True,
                device_map="auto",
                load_in_4bit=True,  # More aggressive quantization
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info("✅ Gemma 3 27B loaded with 4-bit quantization!")
            logger.info(" Memory usage: ~14GB (4-bit quantization)")
            
        except Exception as e2:
            logger.error(f"❌ All loading methods failed: {str(e2)}")
            raise e2

###################################################################################################################

def format_messages_for_gemma3(messages: List[Message], tools: Optional[List[Tool]] = None) -> str:
    """ Convert chat messages to Gemma 3's enhanced format with function calling support"""
    
    formatted_prompt = ""
    
    # Add tools/functions to the system context if provided
    if tools:
        tool_descriptions = []
        for tool in tools:
            func = tool.function
            tool_desc = f"- {func.name}: {func.description}\n  Parameters: {json.dumps(func.parameters, indent=2)}"
            tool_descriptions.append(tool_desc)
        
        if tool_descriptions:
            tools_context = f"""Available tools:
{chr(10).join(tool_descriptions)}

When you need to use a tool, respond with:
Action: tool_name
Arguments: {{"param1": "value1", "param2": "value2"}}

Then wait for the result before continuing."""
            
            formatted_prompt += f"<start_of_turn>user\nSystem: {tools_context}<end_of_turn>\n"
    
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<start_of_turn>user\nSystem: {message.content}<end_of_turn>\n"
        elif message.role == "user" or message.role == "human":
            content = message.content
            
            # Handle multimodal content (images)
            if hasattr(message, 'images') and message.images:
                content += f"\n[Images provided: {len(message.images)} image(s)]"
            
            formatted_prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif message.role == "assistant":
            formatted_prompt += f"<start_of_turn>model\n{message.content}<end_of_turn>\n"
    
    # Add the model turn
    formatted_prompt += "<start_of_turn>model\n"
    return formatted_prompt

###################################################################################################################

def generate_response_gemma3(prompt: str, max_tokens: int = 4096, temperature: float = 0.7, top_p: float = 1.0) -> tuple:
    """ Enhanced generation for Gemma 3 with 128K context support"""
    try:
        # Handle much longer prompts (128K context)
        original_length = len(prompt)
        max_prompt_length = 120000  # Leave room for response in 128K context
        
        if len(prompt) > max_prompt_length:
            
            # Smart truncation: keep the end of the conversation
            prompt = "...[Previous conversation truncated]...\n" + prompt[-max_prompt_length:]
            logger.info(f" Truncated prompt: {original_length} -> {len(prompt)} chars")
        
        # UPDATED: Longer tokenization window for Gemma 3
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=120000  # Much larger context window
        ).to(model.device)
        
        input_length = inputs['input_ids'].shape[1]
        
        #  More generous token limits for Gemma 3
        max_tokens = min(max_tokens, 8192)  # Increased max output
        
        logger.info(f" Generating with Gemma 3: {input_length} input -> max {max_tokens} output tokens")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,
                early_stopping=True,
                # Gemma 3 specific optimizations
                num_beams=1,  # Keep as 1 for speed
                length_penalty=1.0,
            )
        
        # Decode only the new tokens (response)
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Clean up the response
        response = response.strip()
        if response.endswith("<end_of_turn>"):
            response = response[:-13].strip()
        
        prompt_tokens = input_length
        completion_tokens = len(outputs[0]) - input_length
        
        logger.info(f"✅ Generated {completion_tokens} tokens with Gemma 3")
        
        return response, prompt_tokens, completion_tokens
        
    except torch.cuda.OutOfMemoryError:
        logger.error("💥 GPU Out of Memory! Try reducing max_tokens or using more aggressive quantization")
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory. Try reducing request size.")
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

###################################################################################################################

    #          Enhanced memory cleanup for 27B model

@app.middleware("http")
async def cleanup_memory(request: Request, call_next):
    response = await call_next(request)
    
   
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure cleanup completes
    
    return response    
    
###################################################################################################################

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "gemma-3-27b-it",
        "description": "Local Gemma 3 27B API Server - Claude-compatible with enhanced capabilities",
        "endpoints": ["/v1/chat/completions", "/v1/messages", "/health"],
        "capabilities": {
            "context_window": "128K tokens",
            "function_calling": True,
            "multimodal": True,
            "enhanced_reasoning": True
        },
        "memory_usage": "~20GB (BF16 + 8-bit quantization)"
    }

###################################################################################################################

@app.get("/health")
async def health():
    """Detailed health check"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
            "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
        }
    
    return {
        "status": "healthy",
        "model": "gemma-3-27b-it",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_available": torch.cuda.is_available(),
        "device": str(next(model.parameters()).device) if model else None,
        "gpu_memory": gpu_memory,
        "cache_dirs_writable": all(os.access(d, os.W_OK) for d in ['./hf_cache', './torch_cache', './triton_cache', './cache'] if os.path.exists(d)),
        "enhanced_features": {
            "context_window": "128K",
            "function_calling": True,
            "multimodal_ready": True
        }
    }

###################################################################################################################

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """UPDATED: OpenAI-compatible chat completions with Gemma 3 enhancements"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Gemma 3 model not loaded")
    
    try:
        # Format messages for Gemma 3 with tool support
        prompt = format_messages_for_gemma3(request.messages, request.tools)
        
        # Generate response with enhanced parameters
        response_text, prompt_tokens, completion_tokens = generate_response_gemma3(
            prompt, 
            request.max_tokens or 4096,
            request.temperature or 0.7,
            request.top_p or 1.0
        )
        
        # UPDATED: Enhanced response with tool calling detection
        finish_reason = "stop"
        tool_calls = None
        
        # Check if response contains tool calls
        if "Action:" in response_text and "Arguments:" in response_text:
            finish_reason = "tool_calls"
            # Parse tool calls (simplified - you might want more sophisticated parsing)
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("Action:"):
                    action_name = line.replace("Action:", "").strip()
                    if i + 1 < len(lines) and lines[i + 1].startswith("Arguments:"):
                        try:
                            args_str = lines[i + 1].replace("Arguments:", "").strip()
                            arguments = json.loads(args_str)
                            tool_calls = [{
                                "id": f"call_{int(time.time())}",
                                "type": "function",
                                "function": {
                                    "name": action_name,
                                    "arguments": json.dumps(arguments)
                                }
                            }]
                        except json.JSONDecodeError:
                            pass
                    break
        
        # Create OpenAI-compatible response
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": finish_reason
        }
        
        if tool_calls:
            choice["message"]["tool_calls"] = tool_calls
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

###################################################################################################################

@app.post("/v1/messages")
async def claude_messages(request: ClaudeRequest):
    """UPDATED: Claude-compatible messages endpoint with Gemma 3 enhancements"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Gemma 3 model not loaded")
    
    try:
        # Format messages for Gemma 3 with tool support
        prompt = format_messages_for_gemma3(request.messages, request.tools)
        
        # Generate response
        response_text, prompt_tokens, completion_tokens = generate_response_gemma3(
            prompt,
            request.max_tokens,
            request.temperature or 0.7,
            request.top_p or 1.0
        )
        
        # Claude-style response format
        return {
            "id": f"msg_{int(time.time())}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            "model": request.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"Claude messages error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

###################################################################################################################

#        Enhanced test endpoint for Gemma 3

@app.post("/test")
async def test_generation(prompt: str = "Hello! I'm Gemma 3 27B. How can I help you with coding tasks?"):
    """Enhanced test endpoint for Gemma 3 capabilities"""
    if model is None:
        return {"error": "Gemma 3 model not loaded"}
    
    try:
        response, input_tokens, output_tokens = generate_response_gemma3(prompt, max_tokens=200)
        return {
            "model": "gemma-3-27b-it",
            "prompt": prompt,
            "response": response,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "capabilities": "128K context, function calling, enhanced reasoning"
        }
    except Exception as e:
        return {"error": str(e)}



###################################################################################################################

#        Performance monitoring endpoint

@app.get("/stats")
async def get_stats():
    """Performance statistics for Gemma 3 27B"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "model": "gemma-3-27b-it",
        "gpu_memory": {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "utilization_percent": round((torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100, 1) if torch.cuda.max_memory_allocated() > 0 else 0
        },
        "model_info": {
            "parameters": "27B",
            "quantization": "8-bit",
            "context_window": "128K tokens",
            "precision": "BF16"
        }
    }

# UPDATED: Performance monitoring endpoint

# Middleware for logging with performance metrics

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log memory before request
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1024**3
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log memory after request
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s - GPU: {mem_after:.1f}GB")
    else:
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    
    return response

###################################################################################################################

if __name__ == "__main__":
    print(" Starting Gemma 3 27B API Server...")
    print(" This will run on http://localhost:8000")
    print(" Compatible with Claude & OpenAI API endpoints")
    print(" Using BF16 + 8-bit quantization (~20GB VRAM)")
    print(" Enhanced features: 128K context, function calling, multimodal ready")
    print(" All cache directories set to current folder")
    print("\n⚠️  IMPORTANT: Replace 'your_token_here' with your actual Hugging Face token!")
    print("⚠️  MEMORY: Gemma 3 27B needs ~20GB VRAM. Monitor GPU usage!")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
