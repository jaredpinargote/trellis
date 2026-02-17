import sys
import subprocess
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

# Force UTF-8 encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


# --- Configuration ---
SCRIPTS = [
    {
        "name": "Unit Tests & Type Checks",
        "command": [sys.executable, "dev_scripts/run_full_ci_suite.py"],
        "critical": True,
        "type": "Code Quality"
    },
    {
        "name": "System Stress & Latency",
        "command": [sys.executable, "dev_scripts/demo_stress.py"],
        "critical": False,
        "type": "Stability",
        "requires_server": True
    }
]

def check_server_health(url="http://localhost:8000/health"):
    try:
        import requests
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except:
        return False

def run_step(step):
    print(f"\n🔘 Running: {step['name']}...")
    
    if step.get("requires_server") and not check_server_health():
        print(f"⚠️ SKIPPING {step['name']}: API Server not detected at localhost:8000")
        return {
            "name": step["name"],
            "status": "SKIPPED",
            "output": "Server down",
            "type": step["type"]
        }

    start_time = time.time()
    try:
        # $env:PYTHONPATH="." compatibility for subprocess
        env = {"PYTHONPATH": "."}
        if sys.platform == "win32":
            # Start in new shell to inherit current env? 
            # Actually just passing shell=True might be easier for some, 
            # but let's stick to list and manual env merge if needed.
            # Ideally we inherit os.environ
            import os
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
        
        # Force child processes to output UTF-8 so they don't crash when printing emojis
        env["PYTHONIOENCODING"] = "utf-8"
        # Force unbuffered output so we see progress immediately
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            step["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,       # Read raw bytes to avoid codec blocking
            env=env,
            bufsize=0         # Unbuffered
        )

        output_lines = []
        while True:
            # simple readline on bytes still blocks until newline, 
            # but usually CI tools output lines. 
            # If progress bars are an issue, they usually flush \r.
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # Decode manually
                decoded_line = line.decode('utf-8', errors='replace')
                # Sanitize for Windows consoles
                clean_line = decoded_line.strip().encode('ascii', 'replace').decode('ascii')
                if clean_line:
                    print(f"   > {clean_line}")
                output_lines.append(decoded_line)

        returncode = process.poll()
        duration = time.time() - start_time
        
        status = "PASS" if returncode == 0 else "FAIL"
        
        return {
            "name": step["name"],
            "status": status,
            "output": "".join(output_lines),
            "duration": duration,
            "type": step["type"]
        }
    except Exception as e:
        return {
            "name": step["name"],
            "status": "ERROR",
            "output": str(e),
            "duration": time.time() - start_time,
            "type": step["type"]
        }

def print_summary(results):
    print("\n\n" + "="*80)
    print("TRELLIS ECOSYSTEM VERIFICATION REPORT")
    print("="*80)
    print(f"{'TYPE':<15} | {'COMPONENT':<30} | {'STATUS':<10} | {'DURATION':<10}")
    print("-" * 75)
    
    all_passed = True
    
    for r in results:
        status_icon = "[PASS]" if r["status"] == "PASS" else ("[SKIP]" if r["status"] == "SKIPPED" else "[FAIL]")
        if r["status"] == "FAIL":
            all_passed = False
            
        dur = f"{r.get('duration', 0):.2f}s"
        print(f"{r['type']:<15} | {r['name']:<30} | {status_icon:<10} | {dur:<10}")

    print("-" * 75)
    
    if all_passed:
        print("\n[SUCCESS] SYSTEM STATUS: GREEN. Ready for Production.")
        sys.exit(0)
    else:
        print("\n[WARN] SYSTEM STATUS: YELLOW/RED. Issues detected (see details above).")
        sys.exit(1)

def main():
    print("[START] One-Click Verification Ecosystem...")
    results = []
    
    # Run sequentially to avoid fighting for resources/ports/locks
    for step in SCRIPTS:
        res = run_step(step)
        results.append(res)
        
        # Print immediate feedback
        if res["status"] == "FAIL":
            print(f"[FAIL] {step['name']} FAILED")
            # Optional: print last few lines of output
            lines = res['output'].split('\n')
            print("   Error Snippet:")
            for line in lines[-5:]:
                 # Sanitize for Windows consoles (replace non-ascii with ?)
                 clean_line = line.encode('ascii', 'replace').decode('ascii')
                 print(f"   > {clean_line}")
        elif res["status"] == "PASS":
            print(f"[PASS] {step['name']} PASSED ({res['duration']:.2f}s)")
            
    print_summary(results)

if __name__ == "__main__":
    main()
