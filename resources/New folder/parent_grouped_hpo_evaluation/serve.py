"""Serve the pinned base plus the verified refit adapter, without a merged copy."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import shlex
import socket
import subprocess

from external_protocol import digest, model_aliases, read_json, verify_manifest, write_json


def server_command(manifest, port):
    aliases = model_aliases(manifest)
    return ['vllm', 'serve', manifest['model'], '--revision', manifest['revision'],
            '--tokenizer', manifest['adapter'], '--served-model-name', aliases['base'],
            '--enable-lora', '--lora-modules', aliases['grouped'] + '=' + manifest['adapter'],
            '--max-lora-rank', str(manifest['lora_rank']), '--max-loras', '1',
            '--dtype', 'bfloat16', '--tensor-parallel-size', '1',
            '--max-model-len', str(manifest['max_model_len']), '--max-num-seqs', '1',
            '--gpu-memory-utilization', '0.90', '--generation-config', 'vllm',
            '--host', '127.0.0.1', '--port', str(port)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest = verify_manifest(args.artifacts)
    command = server_command(manifest, args.port)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to one GPU before launching')
    # Abort before writing provenance if another process owns this port. Cohort
    # aliases additionally prevent accepting a different server after this check.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', args.port))
    aliases = model_aliases(manifest)
    serving = {'manifest_sha256': digest(manifest), 'command': command,
               'packages': {n: version(n) for n in ('vllm', 'torch', 'transformers', 'numpy')},
               'url': f'http://127.0.0.1:{args.port}/v1', 'adapter_alias': aliases['grouped'], 'base_alias': aliases['base']}
    path = args.artifacts / 'serving.json'
    if path.exists() and read_json(path) != serving:
        raise ValueError('Serving settings changed; prepare a new external-evaluation cohort')
    write_json(path, serving)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
