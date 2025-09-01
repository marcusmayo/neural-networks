import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_trained_model():
    """Load the fine-tuned compliance model"""
    
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./outputs/compliance-tinyllama-lora/final"
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        device_map=None
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=200):
    """Generate response to compliance question"""
    
    prompt = f"Question: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response.strip()
    
    return answer

def test_compliance_questions():
    """Test the model on various compliance questions"""
    
    # Load the trained model
    model, tokenizer = load_trained_model()
    
    # Test questions covering different frameworks
    test_questions = [
        "Which SOC 2 control covers password requirements?",
        "What does ISO 27001 say about access reviews?", 
        "Do we need encryption at rest for HIPAA?",
        "What does SOC 2 require for access controls?",
        "Which ISO 27001 control addresses password management?",
        "What are HIPAA's requirements for audit logs?",
        "How do SOC 2 and ISO 27001 differ in encryption requirements?",
        "What backup requirements does ISO 27001 specify?",
        "Does HIPAA require multi-factor authentication?",
        "What is required for SOC 2 system monitoring?"
    ]
    
    print("\n" + "="*60)
    print("GRC COMPLIANCE LLM - TESTING RESULTS")
    print("="*60)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test Question {i} ---")
        print(f"Q: {question}")
        
        try:
            answer = generate_response(model, tokenizer, question)
            print(f"A: {answer}")
            
            results.append({
                "question": question,
                "answer": answer,
                "status": "success"
            })
            
        except Exception as e:
            print(f"A: [Error generating response: {e}]")
            results.append({
                "question": question,
                "answer": f"Error: {e}",
                "status": "error"
            })
    
    # Save test results
    with open("outputs/compliance_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"TESTING COMPLETED")
    print(f"Total Questions: {len(test_questions)}")
    print(f"Successful Responses: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Results saved to: outputs/compliance_test_results.json")
    print("="*60)

if __name__ == "__main__":
    test_compliance_questions()
