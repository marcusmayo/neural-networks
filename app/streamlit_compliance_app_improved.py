import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

@st.cache_resource
def load_compliance_model():
    """Load the fine-tuned compliance model (cached for performance)"""
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./outputs/compliance-tinyllama-lora/final"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer

def generate_compliance_answer(model, tokenizer, question):
    """Generate answer to compliance question"""
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
        layout="centered"  # Changed to centered for chat-like experience
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
        
        st.header("Model Details")
        st.markdown("""
        - **Base Model:** TinyLlama 1.1B
        - **Fine-tuning:** LoRA adapters
        - **Training:** Compliance Q&A dataset
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
    
    # Chat-style interface
    st.markdown("---")
    
    # Display conversation history
    if st.session_state.conversation_history:
        for i, (q, a, response_time) in enumerate(st.session_state.conversation_history):
            # Question container
            with st.container():
                st.markdown(f"**You:** {q}")
            
            # Answer container  
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
                try:
                    # Load model
                    model, tokenizer = load_compliance_model()
                    
                    # Generate answer
                    start_time = time.time()
                    answer = generate_compliance_answer(model, tokenizer, question)
                    response_time = time.time() - start_time
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append((question, answer, response_time))
                    
                    # Clear current question
                    st.session_state.current_question = ""
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.warning("Please enter a compliance question")
    
    # Current answer display (for immediate feedback)
    if question and st.session_state.get('show_current_answer', False):
        st.markdown("### Current Answer")
        # This would show the answer immediately below the input
    
    # Footer
    st.markdown("---")
    st.markdown("**GRC Compliance LLM** - Trained on SOC 2, ISO 27001, and HIPAA frameworks")

if __name__ == "__main__":
    main()
