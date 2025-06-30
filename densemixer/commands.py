import sys
import os
import shutil
import site

def run():
    if len(sys.argv) < 2 or sys.argv[1] not in {"setup"}:
        print("Usage: densemixer setup")
        sys.exit(1)

    if sys.argv[1] == "setup":
        user_site = site.getusersitepackages()
        if not os.path.isdir(user_site):
            os.makedirs(user_site, exist_ok=True)

        uc_path = os.path.join(user_site, "usercustomize.py")
        import_line = "try:\n    if __import__('os').environ.get('DENSEMIXER_ENABLED') == '1':\n        import densemixer\nexcept ImportError:\n    pass\n"
        need_write = True
        if os.path.exists(uc_path):
            with open(uc_path, 'r') as f:
                if 'import densemixer' in f.read():
                    print("DenseMixer already set up in usercustomize.py.")
                    need_write = False
        if need_write:
            with open(uc_path, 'a') as f:
                f.write(import_line)
            print(f"DenseMixer auto-import added to {uc_path}")
        print("To activate DenseMixer auto-patching, set DENSEMIXER_ENABLED=1 in your environment.") 