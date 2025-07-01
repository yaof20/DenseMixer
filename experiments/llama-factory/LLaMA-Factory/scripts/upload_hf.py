import argparse
import os
import sys
from huggingface_hub import create_repo, HfApi, login
from huggingface_hub.utils import HfHubHTTPError

def parse_args():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub with Weights & Biases logging")
    
    # Required positional arguments
    parser.add_argument("--local_folder", type=str, 
                        help="Path to local model folder")
    parser.add_argument("--repo_id", type=str, 
                        help="Repository ID on Hugging Face (username/repo-name)")
    parser.add_argument("--repo_type", type=str, choices=["model", "dataset", "space"],
                        help="Repository type ('model', 'dataset', or 'space')")
    # Optional token argument
    parser.add_argument("--token", type=str, 
                        help="Hugging Face token (optional if already logged in)")
    
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        # Handle authentication
        if args.token:
            print("🔐 Logging in with provided token...")
            login(token=args.token)
        elif os.getenv("HF_TOKEN"):
            print("🔐 Using token from HF_TOKEN environment variable...")
            login(token=os.getenv("HF_TOKEN"))
        else:
            print("🔐 Using existing login credentials...")
            # Will use cached token from previous huggingface-cli login
        


        # Try to create repository - it will fail if already exists
        try:
            create_repo(repo_id=args.repo_id, repo_type=args.repo_type)
            print(f"✓ Repository {args.repo_id} created successfully.")
        except HfHubHTTPError as e:
            if "409" in str(e):  # 409 means "Conflict" - repository already exists
                print(f"✓ Repository {args.repo_id} already exists. Proceeding to upload...")
            else:
                print(f"✗ ERROR creating repository: {e}")
                sys.exit(1)  # Exit with error code
        except Exception as e:
            print(f"✗ ERROR: Unexpected error creating repository: {e}")
            sys.exit(1)
        
        # Upload to Hugging Face
        print(f"📤 Uploading {args.local_folder} to {args.repo_id} as a {args.repo_type}...")
        
        api = HfApi()
        result = api.upload_folder(
            folder_path=args.local_folder,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
        )
        
        # Verify upload by checking if we can list files
        print("🔍 Verifying upload...")
        files = api.list_repo_files(args.repo_id, repo_type=args.repo_type)
        
        if len(files) > 0:
            print(f"✅ SUCCESS: Upload completed! Found {len(files)} files in repository {args.repo_id}")
            sys.exit(0)  # Explicit success exit
        else:
            print(f"✗ ERROR: Upload may have failed - no files found in repository {args.repo_id}")
            sys.exit(1)
            
    except HfHubHTTPError as e:
        print(f"✗ ERROR: Hugging Face API error during upload: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"✗ ERROR: Local folder not found: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"✗ ERROR: Permission denied: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: Unexpected error during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()