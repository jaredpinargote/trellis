import subprocess
import sys
import shutil

def run_command(command, description):
    print(f"\n🔹 {description}...")
    try:
        # shell=True for windows compatibility with some commands, but list is safer
        # formatting command for display
        cmd_str = " ".join(command)
        print(f"   Running: {cmd_str}")
        
        result = subprocess.run(
            command,
            check=False, # We handle return code manually
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ PASSED: {description}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print(f"❌ FAILED: Command not found: {command[0]}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Error executing {description}: {e}")
        return False

def run_ci_suite():
    print("🚀 Starting Local CI/CD Suite")
    print("=============================")
    
    steps = []
    
    # 1. Type Checking
    steps.append(("Type Checking (mypy)", [sys.executable, "-m", "mypy", "app"]))
    
    # 2. Unit Tests
    steps.append(("Unit Tests (pytest)", [sys.executable, "-m", "pytest"]))
    
    # 3. Docker Build Check
    if shutil.which("docker"):
        # We perform a build check but don't save the image to save time/space
        # Actually standard build is fine, usually cached
        steps.append(("Docker Build Check", ["docker", "build", ".", "-t", "trellis-ci-check"]))
    else:
        print("\n⚠️ Docker not found, skipping Docker build check.")

    # Execute
    failed_steps = []
    for desc, cmd in steps:
        if not run_command(cmd, desc):
            failed_steps.append(desc)
            
    print("\n=============================")
    if not failed_steps:
        print("✅ ALL CHECKS PASSED. Ready for commit/deploy!")
        sys.exit(0)
    else:
        print(f"❌ CI FAILED. The following steps failed: {', '.join(failed_steps)}")
        sys.exit(1)

if __name__ == "__main__":
    run_ci_suite()
