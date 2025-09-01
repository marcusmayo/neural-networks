#!/usr/bin/env python3
"""
GitHub Portfolio Cleanup Script
Removes unnecessary files and prepares project for portfolio upload
"""

import os
import shutil
import json
from pathlib import Path

def github_portfolio_cleanup():
    """Clean project for GitHub portfolio - keep only essential working files"""
    
    print("Preparing GRC Compliance LLM for GitHub Portfolio")
    print("=" * 60)
    
    # Core files to KEEP (essential portfolio components)
    essential_files = {
        # Main application files
        'app/streamlit_compliance_app_improved.py': 'Primary web interface',
        
        # Core training and deployment
        'src/train_qlora_fixed.py': 'Working fine-tuning script',
        'src/test_compliance_model.py': 'Model evaluation',
        'src/deploy_sagemaker_correct_path.py': 'SageMaker deployment',
        'src/interactive_compliance_chat.py': 'CLI interface',
        
        # Dataset and configuration
        'data/compliance_train.jsonl': 'Training dataset',
        'data/compliance_eval.jsonl': 'Evaluation dataset', 
        'data/create_compliance_dataset.py': 'Data creation script',
        'project_config.json': 'Configuration',
        'requirements.txt': 'Dependencies',
        
        # Documentation and results
        'README.md': 'Project documentation',
        'PORTFOLIO_SUMMARY.md': 'Portfolio summary',
        'outputs/compliance_test_results.json': 'Test results',
        'outputs/sagemaker_evidence.json': 'SageMaker proof',
        
        # Model artifacts (keep structure, exclude large binaries)
        'outputs/compliance-tinyllama-lora/final/': 'Model directory structure'
    }
    
    # Files and patterns to DELETE
    cleanup_targets = [
        # Failed/experimental deployment attempts
        'src/deploy_sagemaker_final.py',
        'src/deploy_sagemaker_final_fix.py',
        'src/deploy_sagemaker_minimal.py', 
        'src/deploy_sagemaker_sdk.py',
        'src/deploy_sagemaker_simple.py',
        'src/deploy_sagemaker_supported_version.py',
        'src/deploy_to_sagemaker.py',
        'src/deploy_to_sagemaker_fixed.py',
        'src/deploy_sagemaker_final_fix.py',
        
        # Redundant training scripts
        'src/train_qlora.py',
        'src/train.py',
        
        # Experimental scripts
        'scripts/capture_sagemaker_evidence.py',
        'scripts/check_minimal_status.py',
        'scripts/check_simple_status.py',
        'scripts/fix_sagemaker_permissions.py',
        'scripts/get_sagemaker_image.py',
        'scripts/run_training.py',
        'scripts/run_simple_training.py',
        'scripts/demo_commands.py',
        
        # Redundant app versions (keep only improved)
        'app/streamlit_app.py',
        'app/streamlit_compliance_app.py',
        'app/test_endpoint.py',
        
        # Setup and test files
        'complete_setup_test.py',
        'test_final_setup.py',
        'create_bucket_no_cli.py',
        'create_grc_bucket.py',
        'app/test_setup.py',
        
        # Redundant data files
        'data/compliance_training.json',
        'data/compliance_training.jsonl',
        'data/upload_to_s3.py',
        'data/test_dataset.py',
        
        # Failed deployment records
        'outputs/readme_deployment.json',
        'outputs/sagemaker_deployment.json',
        'outputs/sagemaker_final.json',
        'outputs/sagemaker_minimal_deployment.json',
        'outputs/sagemaker_simple_deployment.json',
        'outputs/roles.json',
        
        # Redundant model packages
        'outputs/compliance-final.tar.gz',
        'outputs/minimal-model.tar.gz',
        'outputs/model.tar.gz',
        'outputs/readme-model.tar.gz',
        'outputs/sagemaker-model.tar.gz',
        'outputs/simple-model.tar.gz',
        'outputs/simple-sagemaker.tar.gz',
        
        # Git download artifacts
        'git-2.39.0/',
        'v2.39.0.tar.gz',
        
        # Empty or unnecessary directories
        'infra/',
        'pipelines/',
        'logs/',
        
        # Cache and temporary files
        '__pycache__/',
        '*.pyc',
        '*.pyo',
        '*.tmp',
        '*.swp',
        '.DS_Store',
        '.cache/'
    ]
    
    removed_count = 0
    kept_count = 0
    
    # Remove unnecessary files
    print("Removing unnecessary files...")
    for target in cleanup_targets:
        if os.path.exists(target):
            try:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                    print(f"  Removed directory: {target}")
                else:
                    os.remove(target)
                    print(f"  Removed file: {target}")
                removed_count += 1
            except Exception as e:
                print(f"  Could not remove {target}: {e}")
    
    # Remove empty directories
    empty_dirs = []
    for root, dirs, files in os.walk('.'):
        if not dirs and not files:
            empty_dirs.append(root)
    
    for empty_dir in empty_dirs:
        try:
            os.rmdir(empty_dir)
            print(f"  Removed empty directory: {empty_dir}")
        except:
            pass
    
    # Count kept files
    for file_path in essential_files.keys():
        if file_path.endswith('/'):  # Directory
            if os.path.exists(file_path):
                kept_count += len([f for f in os.listdir(file_path) if not f.startswith('.')])
        else:  # File
            if os.path.exists(file_path):
                kept_count += 1
    
    print(f"\nCleanup Summary:")
    print(f"  Files/directories removed: {removed_count}")
    print(f"  Essential files kept: {kept_count}")
    
    return True

def create_gitignore():
    """Create comprehensive .gitignore for GitHub"""
    
    gitignore_content = """# Large model files (use download script instead)
outputs/compliance-tinyllama-lora/final/pytorch_model.bin
outputs/compliance-tinyllama-lora/final/model.safetensors
outputs/compliance-tinyllama-lora/final/pytorch_model-*.bin

# Python cache and bytecode
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
env/
venv/
ENV/
env.bak/
venv.bak/

# Environment variables and secrets
.env
.env.local
.env.production
*.key
*.pem

# Logs and databases
*.log
logs/
*.sqlite3
*.db

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~

# Operating system files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary files
*.tmp
*.temp
.cache/

# Model training outputs (keep results, not checkpoints)
outputs/compliance-tinyllama-lora/checkpoint-*/
outputs/*/runs/
outputs/*/logs/

# AWS and deployment
.aws/
terraform.tfstate
terraform.tfstate.backup
*.tfvars
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("Created comprehensive .gitignore")

def show_final_structure():
    """Display clean project structure for portfolio"""
    
    structure = """
Final Portfolio Structure:
├── README.md                           # Complete documentation
├── PORTFOLIO_SUMMARY.md                # Project summary
├── requirements.txt                    # Python dependencies
├── project_config.json                 # Configuration
├── download_model.py                   # Model setup script
├── .gitignore                          # Git exclusions
│
├── data/                               # Dataset
│   ├── compliance_train.jsonl         # Training data
│   ├── compliance_eval.jsonl          # Evaluation data
│   └── create_compliance_dataset.py   # Data creation
│
├── src/                                # Core source code
│   ├── train_qlora_fixed.py           # Fine-tuning script
│   ├── test_compliance_model.py       # Model testing
│   ├── deploy_sagemaker_correct_path.py # SageMaker deployment
│   └── interactive_compliance_chat.py # CLI interface
│
├── app/                                # Web application
│   └── streamlit_compliance_app_improved.py # Main interface
│
└── outputs/                            # Results and evidence
    ├── compliance-tinyllama-lora/final/ # Model artifacts*
    ├── compliance_test_results.json    # Evaluation results
    └── sagemaker_evidence.json        # Deployment proof

*Large model files excluded from git, use download_model.py
"""
    
    print(structure)

def create_download_script():
    """Create script for users to download and setup the model"""
    
    download_script = '''#!/usr/bin/env python3
"""
GRC Compliance LLM - Model Download Script
Run this script to download the base model and setup the project
"""

import os
import sys
from pathlib import Path

def setup_model():
    """Download and setup the GRC Compliance LLM model"""
    
    print("Setting up GRC Compliance LLM...")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("Installing required packages...")
        os.system("pip install transformers torch peft accelerate")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    
    # Create output directory
    output_dir = Path("outputs/compliance-tinyllama-lora/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download base model
    print("Downloading TinyLlama base model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Save locally
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        
        print(f"Model saved to: {output_dir}")
        print("Setup complete! You can now run:")
        print("  streamlit run app/streamlit_compliance_app_improved.py")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please ensure you have internet connection and try again.")

if __name__ == "__main__":
    setup_model()
'''
    
    with open('download_model.py', 'w') as f:
        f.write(download_script)
    
    os.chmod('download_model.py', 0o755)  # Make executable
    print("Created download_model.py for GitHub users")

def main():
    """Main cleanup process"""
    
    # Run cleanup
    success = github_portfolio_cleanup()
    
    if success:
        # Create supporting files
        create_gitignore()
        create_download_script()
        
        # Show final structure
        show_final_structure()
        
        print("\n" + "=" * 60)
        print("GITHUB PORTFOLIO CLEANUP COMPLETE")
        print("=" * 60)
        print("Your project is now ready for GitHub upload!")
        print("Next steps:")
        print("1. git init")
        print("2. git remote add origin https://github.com/marcusmayo/machine-learning-portfolio.git")
        print("3. git add .")
        print("4. git commit -m 'Add GRC Compliance LLM project'")
        print("5. git push -u origin main")

if __name__ == "__main__":
    main()
