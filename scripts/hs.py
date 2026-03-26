#!/usr/bin/env python3
"""Hyperstack GPU cloud management for Claude Code workflow.

Commands:
  hs.py up [--flavor FLAVOR]       Create GPU VM and wait until ready
  hs.py sync <local_dir>           Rsync local dir to VM workspace
  hs.py run <command...>           Run command on VM (cwd = workspace)
  hs.py down [--pull REMOTE LOCAL] Pull results (optional) and destroy VM
  hs.py status                     Show current session info
  hs.py flavors                    List available GPU flavors with stock

Setup:
  1. uv pip install requests
  2. Fill in HYPERSTACK_API_KEY in ~/.config/hyperstack/config.env
  3. Optionally create .hyperstack.env in project root (see scripts/hyperstack.env.template)
  4. chmod +x scripts/hs.py

Config:
  Global (~/.config/hyperstack/config.env): HYPERSTACK_API_KEY
  Per-project (.hyperstack.env in project root): HYPERSTACK_FLAVOR, HYPERSTACK_SSH_USER
  Repo is auto-detected from `git remote get-url origin`.
"""

import os
import sys
import json
import shlex
import time
import socket
import subprocess
import argparse
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: uv pip install requests")
    sys.exit(1)

# --- Constants ---
API_BASE = "https://infrahub-api.nexgencloud.com/v1"
SESSION_FILE = Path.home() / ".hyperstack_session.json"
GLOBAL_CONFIG_FILE = Path.home() / ".config" / "hyperstack" / "config.env"
SSH_KEY_PATH = Path.home() / ".ssh" / "id_ed25519_hyperstack"

KEY_NAME = "claude-local-key"
SSH_USER = "ubuntu"
WORKSPACE = "/home/ubuntu/workspace"

VM_ACTIVE_TIMEOUT = 600  # 10 min
SSH_READY_TIMEOUT = 300  # 5 min


# --- Config ---

def _parse_env_file(path: Path) -> dict:
    config = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            config[k.strip()] = v.strip()
    return config


def _find_project_root() -> Path:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return Path('.')


def load_config() -> dict:
    """Load config: global file first, then per-project overrides, then env vars."""
    config = {}
    if GLOBAL_CONFIG_FILE.exists():
        config.update(_parse_env_file(GLOBAL_CONFIG_FILE))
    local_config = _find_project_root() / '.hyperstack.env'
    if local_config.exists():
        config.update(_parse_env_file(local_config))
    for key in ('HYPERSTACK_API_KEY', 'HYPERSTACK_FLAVOR', 'HYPERSTACK_SSH_USER'):
        if key in os.environ:
            config[key] = os.environ[key]
    return config


def require_api_key(config: dict) -> str:
    key = config.get('HYPERSTACK_API_KEY', '').strip()
    if not key or key == 'your_api_key_here':
        print(f"Error: HYPERSTACK_API_KEY not set in {GLOBAL_CONFIG_FILE} or environment.")
        sys.exit(1)
    return key


def detect_repo() -> str:
    """Auto-detect git remote URL from current directory."""
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Error: could not detect git remote origin. Run from a git repo with a remote origin set.")
        sys.exit(1)


# --- API helpers ---

class APIError(Exception):
    def __init__(self, status_code: int, method: str, path: str, text: str):
        self.status_code = status_code
        super().__init__(f"API error {status_code} [{method} {path}]: {text}")


def api_request(method: str, api_key: str, path: str, data=None) -> dict:
    url = f"{API_BASE}{path}"
    headers = {'api_key': api_key, 'Content-Type': 'application/json'}
    r = requests.request(method, url, headers=headers, json=data, timeout=30)
    if not r.ok:
        raise APIError(r.status_code, method, path, r.text)
    return r.json() if r.content else {}


def api_get(api_key: str, path: str) -> dict:
    return api_request('GET', api_key, path)


def api_post(api_key: str, path: str, data: dict) -> dict:
    return api_request('POST', api_key, path, data)


def api_delete(api_key: str, path: str) -> dict:
    return api_request('DELETE', api_key, path)


# --- Session ---

def load_session() -> dict | None:
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return None


def save_session(data: dict) -> None:
    SESSION_FILE.write_text(json.dumps(data, indent=2))


def clear_session() -> None:
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


def require_session() -> dict:
    session = load_session()
    if not session:
        print("No active session. Run 'hs.py up' first.")
        sys.exit(1)
    return session


# --- Setup helpers ---

def ensure_environment(api_key: str, region: str) -> str:
    """Use the default environment for the given region."""
    env_name = f"default-{region}"
    data = api_get(api_key, '/core/environments')
    names = [e['name'] for e in data.get('environments', [])]
    if env_name in names:
        print(f"  Environment '{env_name}': exists")
        return env_name
    print(f"  Creating environment '{env_name}' in {region}...")
    api_post(api_key, '/core/environments', {'name': env_name, 'region': region})
    print(f"  Environment '{env_name}': created")
    return env_name


def ensure_keypair(api_key: str, env_name: str) -> None:
    data = api_get(api_key, '/core/keypairs')
    keypairs = data.get('keypairs', [])
    existing = next((k for k in keypairs if k['name'] == KEY_NAME), None)

    if existing and SSH_KEY_PATH.exists():
        print(f"  Keypair '{KEY_NAME}': exists (local key present)")
        return

    if existing and not SSH_KEY_PATH.exists():
        print(f"  Keypair '{KEY_NAME}': stale (no local key), re-registering...")
        api_delete(api_key, f"/core/keypair/{existing['id']}")

    pub_path = Path(str(SSH_KEY_PATH) + '.pub')
    if not SSH_KEY_PATH.exists():
        print(f"  Generating SSH key at {SSH_KEY_PATH}...")
        SSH_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ['ssh-keygen', '-t', 'ed25519', '-f', str(SSH_KEY_PATH), '-N', '', '-C', 'hyperstack-claude'],
            check=True, capture_output=True,
        )

    pub_key = pub_path.read_text().strip()
    print(f"  Importing keypair '{KEY_NAME}' to '{env_name}'...")
    api_post(api_key, '/core/keypairs', {
        'name': KEY_NAME,
        'environment_name': env_name,
        'public_key': pub_key,
    })
    print(f"  Keypair '{KEY_NAME}': imported")


def get_flavors(api_key: str) -> list:
    """Return available GPU flavors sorted by (gpu_count, ram) ascending."""
    data = api_get(api_key, '/core/flavors')
    flavors = []
    for group in data.get('data', []):
        for f in group.get('flavors', []):
            if f.get('gpu_count', 0) > 0 and f.get('stock_available'):
                flavors.append(f)
    flavors.sort(key=lambda f: (f.get('gpu_count', 0), f.get('ram', 0)))
    return flavors


def print_flavors(flavors: list) -> None:
    print(f"\n{'#':<4} {'Flavor':<32} {'GPU':<22} {'CPU':<5} {'RAM(GB)':<9} {'Disk(GB)':<10} {'Region'}")
    print('-' * 100)
    for i, f in enumerate(flavors, 1):
        gpu = f"{f.get('gpu_count', '?')}x {f.get('gpu', 'GPU')}"
        print(f"{i:<4} {f['name']:<32} {gpu:<22} {f.get('cpu', '?'):<5} "
              f"{f.get('ram', '?'):<9} {f.get('disk', '?'):<10} {f.get('region_name', '')}")
    print()


def pick_flavor(api_key: str, config: dict) -> tuple[str, str]:
    """Return (flavor_name, region) for the chosen flavor."""
    flavors = get_flavors(api_key)
    if not flavors:
        print("No GPU flavors with available stock.")
        sys.exit(1)

    print_flavors(flavors)

    configured = config.get('HYPERSTACK_FLAVOR', '').strip()
    if configured:
        match = next((f for f in flavors if f['name'] == configured), None)
        if match:
            print(f"Using configured flavor: {configured}")
            return configured, match['region_name']
        print(f"Warning: configured flavor '{configured}' not in stock, choose another.")

    cheapest = flavors[0]
    choice = input(f"Enter number or flavor name [Enter = cheapest: {cheapest['name']}]: ").strip()
    if not choice:
        return cheapest['name'], cheapest['region_name']
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(flavors):
            return flavors[idx]['name'], flavors[idx]['region_name']
    match = next((f for f in flavors if f['name'] == choice), None)
    if match:
        return match['name'], match['region_name']
    print(f"Invalid choice, using cheapest: {cheapest['name']}")
    return cheapest['name'], cheapest['region_name']


def get_ubuntu_image(api_key: str, region: str) -> str:
    """Find Ubuntu 24.04 CUDA image, preferring the matching region."""
    data = api_get(api_key, '/core/images')
    all_images = []
    for group in data.get('images', []):
        for img in group.get('images', []):
            all_images.append(img)

    for img in all_images:
        name = img.get('name', '')
        if img.get('region_name') == region and '24.04' in name and 'cuda' in name.lower():
            return name
    for img in all_images:
        name = img.get('name', '')
        if '24.04' in name and 'cuda' in name.lower():
            return name
    for img in all_images:
        name = img.get('name', '')
        if '24.04' in name:
            return name
    print("Ubuntu 24.04 image not found. Available images:")
    for img in all_images:
        print(f"  {img.get('name', '')} [{img.get('region_name', '')}]")
    sys.exit(1)


def get_vm(api_key: str, vm_id: int) -> dict:
    data = api_get(api_key, f'/core/virtual-machines/{vm_id}')
    return data.get('instance', data)


def wait_vm_active(api_key: str, vm_id: int) -> dict:
    print("Waiting for VM to become active", end='', flush=True)
    deadline = time.time() + VM_ACTIVE_TIMEOUT
    while time.time() < deadline:
        vm = get_vm(api_key, vm_id)
        status = str(vm.get('status', ''))
        if status == 'ACTIVE':
            print(" ready.")
            return vm
        if 'ERROR' in status.upper():
            print(f"\nVM entered error state: {status}")
            sys.exit(1)
        print('.', end='', flush=True)
        time.sleep(15)
    print("\nTimeout: VM did not become active within 10 minutes.")
    sys.exit(1)


def enable_ssh_firewall(api_key: str, vm_id: int) -> None:
    try:
        api_post(api_key, f'/core/virtual-machines/{vm_id}/sg-rules', {
            'direction': 'ingress',
            'protocol': 'tcp',
            'port_range_min': 22,
            'port_range_max': 22,
            'ethertype': 'IPv4',
            'remote_ip_prefix': '0.0.0.0/0',
        })
        print("  SSH firewall rule: added")
    except APIError:
        print("  SSH firewall rule: may already exist, continuing")


def wait_ssh_ready(ip: str) -> None:
    print("Waiting for SSH to be reachable", end='', flush=True)
    deadline = time.time() + SSH_READY_TIMEOUT
    while time.time() < deadline:
        try:
            s = socket.create_connection((ip, 22), timeout=5)
            s.close()
            print(" up.")
            return
        except (socket.timeout, ConnectionRefusedError, OSError):
            print('.', end='', flush=True)
            time.sleep(5)
    print("\nTimeout: SSH not reachable after 5 minutes.")
    sys.exit(1)


def extract_ip(vm: dict) -> str | None:
    """Extract floating/public IP from VM response dict."""
    for key in ('floating_ip', 'public_ip'):
        val = vm.get(key)
        if val:
            return val
    for net_addrs in vm.get('addresses', {}).values():
        for addr in net_addrs:
            if addr.get('OS-EXT-IPS:type') == 'floating':
                return addr['addr']
    return None


def wait_floating_ip(api_key: str, vm_id: int, timeout: int = 120) -> str:
    print("Waiting for floating IP", end='', flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        vm = get_vm(api_key, vm_id)
        ip = extract_ip(vm)
        if ip:
            print(f" {ip}")
            return ip
        print('.', end='', flush=True)
        time.sleep(5)
    print("\nTimeout: floating IP not assigned.")
    sys.exit(1)


def ssh_opts(extra: list | None = None) -> list:
    opts = [
        '-i', str(SSH_KEY_PATH),
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=30',
        '-o', 'ServerAliveInterval=30',
    ]
    if extra:
        opts += extra
    return opts


# --- Commands ---

def cmd_up(args, config):
    session = load_session()
    if session:
        print(f"Session already active: VM {session['vm_id']} at {session['ip']}")
        print("Run 'hs.py down' first to destroy it.")
        sys.exit(1)

    api_key = require_api_key(config)
    repo = detect_repo()

    print("\n[1/6] Selecting flavor...")
    if args.flavor:
        flavors = get_flavors(api_key)
        match = next((f for f in flavors if f['name'] == args.flavor), None)
        if not match:
            print(f"Flavor '{args.flavor}' not found or out of stock.")
            sys.exit(1)
        flavor, region = args.flavor, match['region_name']
    else:
        flavor, region = pick_flavor(api_key, config)
    print(f"  Flavor: {flavor} (region: {region})")

    print("\n[2/6] Checking environment and keypair...")
    env_name = ensure_environment(api_key, region)
    ensure_keypair(api_key, env_name)

    print("\n[3/6] Finding Ubuntu 24.04 image...")
    image = get_ubuntu_image(api_key, region)
    print(f"  Image: {image}")

    vm_name = f"claude-gpu-{int(time.time())}"
    print(f"\n[4/6] Creating VM '{vm_name}' ...")
    result = api_post(api_key, '/core/virtual-machines', {
        'name': vm_name,
        'environment_name': env_name,
        'image_name': image,
        'flavor_name': flavor,
        'key_name': KEY_NAME,
        'assign_floating_ip': True,
        'enable_port_randomization': False,
        'count': 1,
    })

    vm_list = result.get('virtual_machines', result.get('instances', []))
    if vm_list:
        vm_id = vm_list[0].get('id')
    else:
        vm_id = result.get('virtual_machine', {}).get('id')
    if not vm_id:
        print(f"Could not extract VM id from API response:\n{json.dumps(result, indent=2)}")
        sys.exit(1)
    print(f"  VM id: {vm_id}")

    print("\n[5/6] Waiting for VM ready + SSH...")
    vm = wait_vm_active(api_key, vm_id)
    enable_ssh_firewall(api_key, vm_id)
    ip = extract_ip(vm) or wait_floating_ip(api_key, vm_id)
    print(f"  IP: {ip}")
    wait_ssh_ready(ip)

    save_session({'vm_id': vm_id, 'ip': ip, 'flavor': flavor,
                  'created_at': time.time(), 'repo': repo})

    print(f"\n[6/6] Cloning repo {repo} ...")
    subprocess.run(
        ['ssh'] + ssh_opts() + [
            f'{SSH_USER}@{ip}',
            f'git clone {shlex.quote(repo)} {WORKSPACE} 2>&1 || (cd {WORKSPACE} && git pull)',
        ],
        check=True,
    )
    print(f"\nVM ready.")
    print(f"  SSH: ssh {' '.join(ssh_opts())} {SSH_USER}@{ip}")
    print(f"  Workspace: {WORKSPACE}")
    print(f"  Session: {SESSION_FILE}")
    print("\n[BUDGET POLICY] Run 'hs.py down' immediately when the job finishes. Do not leave the VM idle.")


def cmd_sync(args, config):
    session = require_session()
    local = args.local_dir.rstrip('/') + '/'
    remote = f'{SSH_USER}@{session["ip"]}:{WORKSPACE}/'
    print(f"Syncing {local} -> {remote}")
    subprocess.run([
        'rsync', '-avz', '--progress',
        '--exclude=.git',
        '--exclude=__pycache__',
        '--exclude=*.pyc',
        '--exclude=.env',
        '--exclude=*.egg-info',
        '--exclude=.DS_Store',
        '-e', f'ssh {" ".join(ssh_opts())}',
        local, remote,
    ], check=True)


def cmd_run(args, config):
    session = require_session()
    ip = session['ip']
    command = shlex.join(args.command)
    print(f"[VM] {command}\n")
    result = subprocess.run(
        ['ssh'] + ssh_opts() + [f'{SSH_USER}@{ip}', f'cd {WORKSPACE} && {command}']
    )
    sys.exit(result.returncode)


def cmd_down(args, config):
    session = require_session()
    api_key = require_api_key(config)
    ip = session['ip']
    vm_id = session['vm_id']

    if args.pull:
        remote_path, local_path = args.pull
        print(f"Pulling {remote_path} -> {local_path} ...")
        subprocess.run([
            'rsync', '-avz', '--progress',
            '-e', f'ssh {" ".join(ssh_opts())}',
            f'{SSH_USER}@{ip}:{remote_path}/', local_path,
        ], check=True)

    print(f"Destroying VM {vm_id} ...")
    api_delete(api_key, f'/core/virtual-machines/{vm_id}')
    clear_session()
    elapsed = (time.time() - session['created_at']) / 3600
    print(f"VM destroyed. Session lasted {elapsed:.2f} hours.")


def cmd_status(args, config):
    session = load_session()
    if not session:
        print("No active session.")
        return
    elapsed = (time.time() - session['created_at']) / 60
    print(f"VM ID:     {session['vm_id']}")
    print(f"IP:        {session['ip']}")
    print(f"Flavor:    {session['flavor']}")
    print(f"Repo:      {session.get('repo', 'unknown')}")
    print(f"Running:   {elapsed:.0f} minutes")
    print(f"SSH:       ssh {' '.join(ssh_opts())} {SSH_USER}@{session['ip']}")
    print(f"Workspace: {WORKSPACE}")


def cmd_flavors(args, config):
    api_key = require_api_key(config)
    flavors = get_flavors(api_key)
    if not flavors:
        print("No GPU flavors with available stock.")
        return
    print_flavors(flavors)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Hyperstack GPU cloud for Claude Code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_up = sub.add_parser('up', help='Create GPU VM')
    p_up.add_argument('--flavor', help='Flavor name (skips interactive prompt)')

    p_sync = sub.add_parser('sync', help='Rsync local dir to VM workspace')
    p_sync.add_argument('local_dir', help='Local directory to sync')

    p_run = sub.add_parser('run', help='Run command on VM')
    p_run.add_argument('command', nargs='+', help='Command to run on VM')

    p_down = sub.add_parser('down', help='Pull results (optional) and destroy VM')
    p_down.add_argument('--pull', nargs=2, metavar=('REMOTE_PATH', 'LOCAL_PATH'),
                        help='rsync remote path to local before destroying')

    sub.add_parser('status', help='Show current session info')
    sub.add_parser('flavors', help='List available GPU flavors')

    args = parser.parse_args()
    config = load_config()

    global SSH_USER
    SSH_USER = config.get('HYPERSTACK_SSH_USER', SSH_USER)

    dispatch = {
        'up': cmd_up,
        'sync': cmd_sync,
        'run': cmd_run,
        'down': cmd_down,
        'status': cmd_status,
        'flavors': cmd_flavors,
    }
    try:
        dispatch[args.cmd](args, config)
    except APIError as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
