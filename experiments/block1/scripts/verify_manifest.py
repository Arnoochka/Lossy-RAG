#!/usr/bin/env python3
"""Verify the immutable Block 1 release files, from any current directory."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
entries=json.loads((ROOT/'file_manifest.json').read_text(encoding='utf-8'))['files']
failed=[]
for item in entries:
    relative=Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        failed.append('UNSAFE PATH '+str(relative));continue
    path=ROOT/relative
    if not path.is_file():failed.append('MISSING '+str(relative));continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    if path.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:failed.append('HASH/SIZE '+str(relative))
print(f'Manifest: {len(entries)} files, errors={len(failed)}')
for message in failed:print(message)
sys.exit(bool(failed))
