import os
import shutil
import json
from pathlib import Path

def portfolio_cleanup():
    """Clean project for GitHub portfolio - keep only working files"""
    
    print("Cleaning project for GitHub portfolio...")
    print("=" * 50)
    
    # Files to KEEP (core working components)
    keep_files = {
        # Core application files
        'app/streamlit_compliance_app_improved.py': 'Main web interface',
        
        # Training and model files  
        'src/train_qlora_fixed.py': 'Working training script',
        'src/test_compliance_model.py': 'Model testing',
        'src/deploy_sagemaker_correct_path.py': 'Working SageMaker deployment',
        
        # Data files
        'data/compliance_train.jsonl': 'Training dataset',
        'data/compliance_eval.jsonl': 'Evaluation dataset',
        'data/create_compliance_dataset.py': 'Dataset creation',
        
        # Model artifacts (keep the final trained model)
        'outputs/compliance-tinyllama-lora/': 'Trained model directory',
        'outputs/compliance_test_results.json': 'Test results',
        'outputs/sagemaker_working.json': 'SageMaker deployment info',
        
        # Configuration
        'project_config.json': 'Project configuration',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
        '.env': 'Environment variables'
    }
    
    # Files and patterns to DELETE
    delete_patterns = [
        # Failed deployment attempts
        'src/deploy_sagemaker_final.py',
        'src/deploy_sagemaker_final_fix.py', 
        'src/deploy_sagemaker_minimal.py',
        'src/deploy_sagemaker_sdk.py',
        'src/deploy_sagemaker_simple.py',
        'src/deploy_sagemaker_supported_version.py',
        'src/deploy_to_sagemaker.py',
        'src/deploy_to_sagemaker_fixed.py',
        
        # Redundant model files
        'outputs/compliance-final.tar.gz',
        'outputs/minimal-model.tar.gz',
        'outputs/model.tar.gz', 
        'outputs/readme-model.tar.gz',
        'outputs/sagemaker-model.tar.gz',
        'outputs/simple-model.tar.gz',
        'outputs/simple-sagemaker.tar.gz',
        
        # Failed deployment configs
        'outputs/readme_deployment.json',
        'outputs/sagemaker_deployment.json',
        'outputs/sagemaker_final.json',
        'outputs/sagemaker_minimal_deployment.json',
        'outputs/sagemaker_simple_deployment.json',
        'outputs/roles.json',
        
        # Experimental and test scripts
        'scripts/check_minimal_status.py',
        'scripts/check_simple_status.py', 
        'scripts/fix_sagemaker_permissions.py',
        'scripts/get_sagemaker_image.py',
        'scripts/run_simple_training.py',
        'scripts/run_training.py',
        
        # Original apps (keep only improved version)
        'app/streamlit_app.py',
        'app/streamlit_compliance_app.py',
        
        # Redundant training scripts
        'src/train_qlora.py',
        'src/train.py',
        
        # Redundant data files
        'data/compliance_training.json',
        'data/compliance_training.jsonl',
        'data/upload_to_s3.py',
        
        # Setup files (not needed for portfolio)
        'app/test_setup.py',
        'test_final_setup.py',
        'complete_setup_test.py',
        'create_bucket_no_cli.py',
        'create_grc_bucket.py',
        
        # Temporary files
        'logs/',
        '__pycache__/',
        '*.pyc',
        '*.tmp',
        '.DS_Store'
    ]
    
    removed_count = 0
    kept_count = 0
    
    # Delete unnecessary files
    for pattern in delete_patterns:
        if os.path.exists(pattern):
            try:
                if os.path.isdir(pattern):
                    shutil.rmtree(pattern)
                    print(f"✅ Removed directory: {pattern}")
                else:
                    os.remove(pattern)
                    print(f"✅ Removed file: {pattern}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Could not remove {pattern}: {e}")
    
    # Clean up empty directories
    empty_dirs = ['infra', 'pipelines']
    for dir_name in empty_dirs:
        if os.path.exists(dir_name) and not os.listdir(dir_name):
            try:
                os.rmdir(dir_name)
                print(f"✅ Removed empty directory: {dir_name}")
            except:
                pass
    
    # Count kept files
    for file_path in keep_files.keys():
        if os.path.exists(file_path):
            kept_count += 1
    
    print(f"\n" + "="*50)
    print(f"PORTFOLIO CLEANUP COMPLETE")
    print(f"="*50)
    print(f"Files removed: {removed_count}")
    print(f"Core files kept: {kept_count}")
    
    # Show final project structure
    print(f"\n📁 CLEAN PROJECT STRUCTURE:")
    show_clean_structure()

def show_clean_structure():
    """Display the cleaned project structure"""
    
    structure = """
    grc-llm-project/
    ├── README.md                    # Complete documentation
    ├── requirements.txt             # Dependencies  
    ├── project_config.json          # Project configuration
    ├── 
    ├── data/
    │   ├── compliance_train.jsonl   # Training data
    │   ├── compliance_eval.jsonl    # Evaluation data
    │   └── create_compliance_dataset.py
    ├── 
    ├── src/  
    │   ├── train_qlora_fixed.py     # Training script
    │   ├── test_compliance_model.py # Testing
    │   └── deploy_sagemaker_correct_path.py # Deployment
    ├── 
    ├── app/
    │   └── streamlit_compliance_app_improved.py # Web interface
    ├── 
    └── outputs/
        ├── compliance-tinyllama-lora/final/     # Trained model
        ├── compliance_test_results.json         # Results
        └── sagemaker_working.json              # Deployment info
    """
    
    print(structure)

if __name__ == "__main__":
    portfolio_cleanup()
