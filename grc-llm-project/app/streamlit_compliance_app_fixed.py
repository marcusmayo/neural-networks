import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import os

@st.cache_resource
def load_compliance_model():
    """Load the fine-tuned compliance model (cached for performance)"""
    
    # Determine the correct path based on environment
    if os.path.exists("/mount/src/machine-learning-portfolio"):
        # Running on Streamlit Cloud
        base_path = "/mount/src/machine-learning-portfolio/grc-llm-project"
    else:
        # Running locally
        base_path = "."
    
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = os.path.join(base_path, "outputs/compliance-tinyllama-lora/final")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Try to load LoRA adapter if it exists
        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            st.warning("LoRA adapter not found. Using base model only.")
            model = base_model
            
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_compliance_answer(model, tokenizer, question):
    """Generate answer to compliance question"""
    if model is None or tokenizer is None:
        return "Error: Model not loaded properly"
        
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    return answer

def main():
    st.set_page_config(
        page_title="GRC Compliance LLM",
        page_icon="🛡️",
        layout="centered"
    )
    
    # Header
    st.title("🛡️ GRC Compliance Chatbot")
    st.markdown("**SOC 2 • ISO 27001 • HIPAA Compliance Assistant**")
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar with information and examples
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot is trained on compliance frameworks:
        - **SOC 2** - Security controls
        - **ISO 27001** - Information security  
        - **HIPAA** - Healthcare data protection
        """)
        
        st.header("Example Questions")
        examples = [
            "Which SOC 2 control covers password requirements?",
            "What does ISO 27001 say about access reviews?",
            "Do we need encryption at rest for HIPAA?",
            "What backup requirements does ISO 27001 specify?",
            "Does HIPAA require multi-factor authentication?"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.current_question = example
                st.rerun()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Load model
    model, tokenizer = load_compliance_model()
    
    # Chat interface
    st.markdown("---")
    
    # Display conversation history
    if st.session_state.conversation_history:
        for i, (q, a, response_time) in enumerate(st.session_state.conversation_history):
            with st.container():
                st.markdown(f"**You:** {q}")
            with st.container():
                st.markdown(f"**Assistant:** {a}")
                st.caption(f"*Response time: {response_time:.1f}s*")
                st.markdown("---")
    
    # Current question input
    question = st.text_area(
        "Ask your compliance question:",
        value=st.session_state.get('current_question', ''),
        height=100,
        placeholder="e.g., Which SOC 2 control covers password management?",
        key="question_input"
    )
    
    # Submit button
    if st.button("📤 Send", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("Analyzing compliance requirements..."):
                start_time = time.time()
                answer = generate_compliance_answer(model, tokenizer, question)
                response_time = time.time() - start_time
                
                # Add to conversation history
                st.session_state.conversation_history.append((question, answer, response_time))
                
                # Clear current question
                st.session_state.current_question = ""
                
                st.rerun()
        else:
            st.warning("Please enter a compliance question")
    
    # Footer
    st.markdown("---")
    st.markdown("**GRC Compliance LLM** - Trained on SOC 2, ISO 27001, and HIPAA frameworks")

if __name__ == "__main__":
    main()
