import os
import shutil
import json
from pathlib import Path

def cleanup_project():
    """Clean up unnecessary and temporary files"""
    
    print("🧹 Cleaning up GRC LLM project...")
    
    # Files and directories to remove
    cleanup_targets = [
        # Temporary files
        "__pycache__",
        "*.pyc", 
        "*.pyo",
        ".DS_Store",
        
        # Training logs and checkpoints (keep final model)
        "outputs/compliance-tinyllama-lora/checkpoint-*",
        "logs/events.out.tfevents.*",
        "logs/trainer_state.json",
        
        # Temporary model files
        "*.tmp",
        "*.swp",
        ".cache",
        
        # Large model downloads cache (can be re-downloaded)
        "~/.cache/huggingface/transformers",
        
        # Redundant data files
        "data/compliance_training.json",  # Keep jsonl versions
    ]
    
    # Directories to clean recursively
    dirs_to_clean = [
        ".",
        "src",
        "app", 
        "data",
        "outputs",
        "logs",
        "scripts"
    ]
    
    removed_count = 0
    space_saved = 0
    
    for directory in dirs_to_clean:
        if not os.path.exists(directory):
            continue
            
        print(f"\n🔍 Cleaning {directory}/")
        
        for root, dirs, files in os.walk(directory):
            # Remove __pycache__ directories
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                try:
                    size = sum(os.path.getsize(os.path.join(pycache_path, f)) 
                              for f in os.listdir(pycache_path))
                    shutil.rmtree(pycache_path)
                    print(f"  ✅ Removed __pycache__ ({size/1024:.1f} KB)")
                    removed_count += 1
                    space_saved += size
                except Exception as e:
                    print(f"  ❌ Could not remove __pycache__: {e}")
            
            # Remove specific files
            for file in files:
                file_path = os.path.join(root, file)
                
                # Remove .pyc files
                if file.endswith(('.pyc', '.pyo')):
                    try:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        print(f"  ✅ Removed {file} ({size} bytes)")
                        removed_count += 1
                        space_saved += size
                    except Exception as e:
                        print(f"  ❌ Could not remove {file}: {e}")
                
                # Remove temporary files
                if file.endswith(('.tmp', '.swp')) or file == '.DS_Store':
                    try:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        print(f"  ✅ Removed {file} ({size} bytes)")
                        removed_count += 1
                        space_saved += size
                    except Exception as e:
                        print(f"  ❌ Could not remove {file}: {e}")
    
    # Clean up training checkpoints (keep final model)
    checkpoints_dir = "outputs/compliance-tinyllama-lora"
    if os.path.exists(checkpoints_dir):
        print(f"\n🔍 Cleaning training checkpoints in {checkpoints_dir}/")
        for item in os.listdir(checkpoints_dir):
            if item.startswith('checkpoint-') and item != 'final':
                checkpoint_path = os.path.join(checkpoints_dir, item)
                try:
                    if os.path.isdir(checkpoint_path):
                        size = sum(os.path.getsize(os.path.join(root, file))
                                  for root, dirs, files in os.walk(checkpoint_path)
                                  for file in files)
                        shutil.rmtree(checkpoint_path)
                        print(f"  ✅ Removed checkpoint {item} ({size/1024/1024:.1f} MB)")
                        removed_count += 1
                        space_saved += size
                except Exception as e:
                    print(f"  ❌ Could not remove {item}: {e}")
    
    # Remove redundant JSON file (keep JSONL)
    json_file = "data/compliance_training.json"
    if os.path.exists(json_file):
        try:
            size = os.path.getsize(json_file)
            os.remove(json_file)
            print(f"  ✅ Removed redundant JSON file ({size/1024:.1f} KB)")
            removed_count += 1
            space_saved += size
        except Exception as e:
            print(f"  ❌ Could not remove JSON file: {e}")
    
    # Show cleanup summary
    print(f"\n{'='*50}")
    print(f"🧹 CLEANUP COMPLETE")
    print(f"{'='*50}")
    print(f"Files removed: {removed_count}")
    print(f"Space saved: {space_saved/1024/1024:.2f} MB")
    
    # Show what's kept
    print(f"\n📁 IMPORTANT FILES KEPT:")
    important_files = [
        "data/compliance_train.jsonl",
        "data/compliance_eval.jsonl", 
        "outputs/compliance-tinyllama-lora/final/",
        "outputs/compliance_test_results.json",
        "project_config.json",
        "src/train_qlora_fixed.py",
        "app/streamlit_compliance_app_improved.py"
    ]
    
    for file in important_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❓ {file} (not found)")

def show_project_structure():
    """Show clean project structure"""
    
    print(f"\n📂 CLEAN PROJECT STRUCTURE:")
    print(f"{'='*40}")
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if not file.startswith('.') and not file.endswith(('.pyc', '.pyo')):
                print(f"{subindent}{file}")

if __name__ == "__main__":
    cleanup_project()
    show_project_structure()
