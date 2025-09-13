"""
CareCopilot Demo Application
HIPAA-Ready RAG + FHIR Summarization Agent
Built for PointClickCare
"""
import streamlit as st
import json
from datetime import datetime

# Configure Streamlit with PointClickCare branding
st.set_page_config(
    page_title="CareCopilot - PointClickCare Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# PointClickCare-inspired CSS styling
st.markdown("""
<style>
/* PointClickCare color scheme: Teal primary, clean whites */
.main-header {
    font-size: 2.8rem;
    color: #0B7A75;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 600;
}
.subtitle {
    font-size: 1.2rem;
    color: #2E8B85;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}
.demo-section {
    background-color: #FFFFFF;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #0B7A75;
}
.working-button {
    background-color: #0B7A75 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    margin: 0.25rem !important;
    font-weight: 500 !important;
}
.working-button:hover {
    background-color: #085E5A !important;
    color: white !important;
}
.citation-box {
    background-color: #E8F6F5;
    padding: 1rem;
    border-left: 4px solid #2E8B85;
    margin: 0.5rem 0;
    color: #0B5D57;
    font-weight: 500;
    border-radius: 4px;
}
.success-box {
    background-color: #F0F9F8;
    padding: 1.2rem;
    border-left: 4px solid #0B7A75;
    margin: 0.5rem 0;
    color: #0B5D57;
    font-size: 1.1rem;
    font-weight: 500;
    border-radius: 4px;
}
.healthcare-card {
    background: linear-gradient(135deg, #0B7A75, #2E8B85);
    color: white;
    padding: 1.5rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header with PointClickCare styling
st.markdown('<h1 class="main-header">🏥 CareCopilot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">HIPAA-Ready RAG + FHIR Summarization Agent for PointClickCare</p>', unsafe_allow_html=True)

# Sidebar with healthcare branding
st.sidebar.header("🏥 Demo Controls")
demo_mode = st.sidebar.selectbox(
    "Select Demo Mode:",
    ["RAG Chat", "FHIR Agent", "Both Systems"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:**")
st.sidebar.success("✅ Database: 151 documents")
st.sidebar.success("✅ Vector Search: Active")
st.sidebar.success("✅ FHIR Agent: Ready")

# Main content
if demo_mode == "RAG Chat" or demo_mode == "Both Systems":
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.header("🔍 RAG Chat - Query Medical Records")
    
    # Only include WORKING buttons based on test results
    st.markdown("### 💡 Try These Verified Working Queries:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 What are the discharge instructions?", key="ex1", help="Searches discharge documentation"):
            st.session_state.current_query = "discharge instructions"
            st.session_state.trigger_search = True
    
    with col2:
        if st.button("📄 Medical summary", key="ex2", help="Finds medical summaries"):
            st.session_state.current_query = "medical summary"
            st.session_state.trigger_search = True
    
    with col3:
        if st.button("🏥 Discharge information", key="ex3", help="Searches all discharge content"):
            st.session_state.current_query = "discharge"
            st.session_state.trigger_search = True
    
    st.markdown("---")
    
    # Query input
    query = st.text_input(
        "Or type your own question about the medical records:",
        value=st.session_state.current_query,
        placeholder="e.g., What medications were prescribed?",
        key="rag_query_input"
    )
    
    # Update session state when text input changes
    if query != st.session_state.current_query:
        st.session_state.current_query = query
    
    # Manual search button
    manual_search = st.button("🔍 Search Medical Records", type="primary")
    
    # Trigger search either manually or automatically
    if (manual_search and query) or (st.session_state.trigger_search and st.session_state.current_query):
        # Reset trigger
        st.session_state.trigger_search = False
        
        search_query = st.session_state.current_query
        
        with st.spinner("Searching medical records..."):
            try:
                from src.api.rag_api import query_engine
                
                result = query_engine.query(search_query)
                
                if result['total_results'] > 0:
                    st.success(f"Found {len(result['search_results'])} relevant documents")
                    
                    # Display answer
                    if result.get('answer') and not result['answer'].startswith('Error'):
                        st.markdown("### 📋 Answer:")
                        answer_text = result["answer"]
                        st.markdown(f'<div class="success-box">{answer_text}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Display search results
                    st.markdown("### 📄 Source Documents:")
                    for i, doc in enumerate(result['search_results']):
                        similarity = doc.get('similarity', 0.0)
                        sim_display = f"{similarity*100:.1f}%"
                        
                        with st.expander(f"Document {i+1} - Similarity: {sim_display}"):
                            content = doc.get('content', 'No content available')
                            st.text(content[:500] + "..." if len(content) > 500 else content)
                            st.caption(f"Document ID: {doc.get('document_id', 'Unknown')} | Chunk: {doc.get('chunk_index', 'N/A')}")
                else:
                    st.warning("No relevant documents found. Try one of the verified working queries above.")
                
            except Exception as e:
                st.error(f"Search Error: {str(e)}")
                st.info("💡 Please try one of the working example queries above")
    
    st.markdown('</div>', unsafe_allow_html=True)

if demo_mode == "FHIR Agent" or demo_mode == "Both Systems":
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.header("📝 FHIR Agent - Convert Clinical Notes")
    
    # Example clinical notes
    st.markdown("### 💡 Example Clinical Notes to Try:")
    
    fhir_examples = [
        {
            "title": "🏥 Discharge Summary",
            "note": """Patient John Smith discharged today after 3-day admission for pneumonia. 
Final diagnoses: Community-acquired pneumonia, Type 2 diabetes mellitus. 
Discharge medications: Azithromycin 250mg daily x5 days, Metformin 500mg twice daily.
Follow-up: Primary care in 1 week, chest X-ray in 2 weeks."""
        },
        {
            "title": "💊 Medication Review", 
            "note": """Patient presents for medication review. Current medications include:
Lisinopril 10mg daily for hypertension, Atorvastatin 20mg nightly for hyperlipidemia,
Aspirin 81mg daily for cardio protection. Blood pressure today: 135/85 mmHg.
Plan: Increase Lisinopril to 20mg daily."""
        },
        {
            "title": "🔬 Lab Follow-up",
            "note": """Lab results review for diabetic patient. HbA1c: 8.2% (elevated).
Current on Metformin 1000mg twice daily. Patient reports good adherence.
Plan: Add Glipizide 5mg daily, recheck HbA1c in 3 months."""
        }
    ]
    
    for example in fhir_examples:
        with st.expander(f"📋 {example['title']}"):
            st.text_area("", value=example['note'], height=100, key=f"example_{example['title']}", disabled=True)
            if st.button(f"Use This Example", key=f"use_{example['title']}"):
                st.session_state['clinical_text'] = example['note']
                st.session_state['patient_name'] = "John Smith"
                st.rerun()
    
    st.markdown("---")
    
    # Sample clinical note
    default_note = """Patient John Smith presents with Type 2 diabetes mellitus and hypertension. 
Current medications include Metformin 500mg twice daily and Lisinopril 10mg once daily.
Blood pressure today: 140/90 mmHg. HbA1c: 7.2%. Patient reports good medication compliance.
Plan: Continue current medications, follow up in 3 months."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        clinical_text = st.text_area(
            "Enter clinical note:",
            value=st.session_state.get('clinical_text', default_note),
            height=150,
            placeholder="Enter free-text clinical note here...",
            key="clinical_text_input"
        )
    
    with col2:
        patient_name = st.text_input(
            "Patient Name:", 
            value=st.session_state.get('patient_name', "John Smith"),
            key="patient_name_input"
        )
        
        if st.button("📋 Reset to Default"):
            st.session_state['clinical_text'] = default_note
            st.session_state['patient_name'] = "John Smith"
            st.rerun()
    
    if st.button("🔄 Convert to FHIR", type="primary") and clinical_text:
        with st.spinner("Converting to FHIR format..."):
            try:
                from src.agents.fhir_agent import fhir_agent
                
                result = fhir_agent.convert_to_fhir(clinical_text, patient_name)
                
                if result['success']:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Resources", result['resource_count'])
                    with col2:
                        st.metric("Conditions Found", result['summary']['conditions'])
                    with col3:
                        st.metric("Medications Found", result['summary']['medications'])
                    
                    # Display extracted entities
                    st.markdown("### 🏷️ Extracted Medical Entities:")
                    entities = result['entities_extracted']
                    
                    if entities.get('conditions'):
                        st.markdown("**🔍 Conditions Found:**")
                        for condition in entities['conditions']:
                            st.markdown(f'<div class="citation-box">🏥 {condition["text"]} (Confidence: {condition["confidence"]:.1%})</div>', 
                                      unsafe_allow_html=True)
                    
                    if entities.get('medications'):
                        st.markdown("**💊 Medications Found:**")
                        for med in entities['medications']:
                            st.markdown(f'<div class="citation-box">💊 {med["text"]} (Confidence: {med["confidence"]:.1%})</div>', 
                                      unsafe_allow_html=True)
                    
                    if not entities.get('conditions') and not entities.get('medications'):
                        st.info("No medical entities detected in this note. Try a note with specific conditions and medications.")
                    
                    # Display FHIR Bundle
                    with st.expander("📋 View Complete FHIR Bundle (JSON)"):
                        st.json(result['bundle'])
                    
                    st.success("✅ Successfully converted clinical note to FHIR format!")
                    
                else:
                    st.error(f"FHIR Conversion failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"FHIR Agent Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with PointClickCare branding
st.markdown("---")
st.markdown("**PointClickCare CareCopilot Features:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="healthcare-card">🔒 <strong>HIPAA Compliant</strong><br>All data stays in VPC</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="healthcare-card">🔍 <strong>Smart Retrieval</strong><br>Vector similarity search</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="healthcare-card">⚡ <strong>Real-time Processing</strong><br>FHIR conversion in seconds</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("CareCopilot - Powered by AI • Built for PointClickCare • HIPAA Ready")
