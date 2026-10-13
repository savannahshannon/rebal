#!/usr/bin/env python3
"""Verify ZIP and exact member byte identities against an external manifest.

Standard-library only. No extraction, network, model loading, numeric replay,
training or privacy-pattern scanning occurs. The supplied manifest is a reference,
not an authenticated authority. Use a fresh JSON receipt outside the ZIP.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import re
import stat
import struct
import zipfile

BLOCK_BYTES = 8 * 1024**2
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON manifest key')
        result[key] = value
    return result


def nonfinite(value):
    raise ValueError('Nonfinite JSON manifest constant')


def count(value, label, positive=False):
    require(type(value) is int and value >= (1 if positive else 0),
            label + ' must be a nonnegative integer, not a boolean or float')
    return value


def digest(value):
    require(isinstance(value, str) and SHA256.fullmatch(value), 'Invalid SHA-256 identity')
    return value


def safe_path(value):
    require(isinstance(value, str) and value and all(ord(c) >= 32 and ord(c) != 127 for c in value),
            'Unsafe member path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in value and ':' not in value
            and str(path) == value and str(path) not in ('', '.') and not value.endswith('/'),
            'Unsafe member path')
    return path


def manifest_record(body):
    require(isinstance(body, dict) and isinstance(body.get('files'), list) and len(body['files']) == 1,
            'Exactly one ZIP record is required in external manifest files')
    record = body['files'][0]
    require(isinstance(record, dict) and
            {'path', 'bytes', 'sha256', 'entries', 'max_member_bytes', 'max_uncompressed_bytes'} <= set(record),
            'Incomplete external ZIP record')
    path = safe_path(record['path'])
    require(len(path.parts) == 1 and path.suffix.lower() == '.zip', 'Manifest ZIP path must be a relative basename')
    count(record['bytes'], 'Archive bytes', positive=True); digest(record['sha256'])
    max_member = count(record['max_member_bytes'], 'Member byte ceiling')
    max_total = count(record['max_uncompressed_bytes'], 'Uncompressed byte ceiling')
    require(isinstance(record['entries'], list) and record['entries'], 'Nonempty exact member catalog required')
    entries = {}
    for entry in record['entries']:
        require(isinstance(entry, dict) and {'path', 'bytes', 'sha256'} <= set(entry), 'Incomplete member identity')
        name = str(safe_path(entry['path']))
        require(name not in entries, 'Duplicate manifest member path')
        size = count(entry['bytes'], 'Member bytes'); digest(entry['sha256'])
        require(size <= max_member, 'Manifest member exceeds declared member byte ceiling')
        entries[name] = dict(bytes=size, sha256=entry['sha256'])
    require(sum(entry['bytes'] for entry in entries.values()) <= max_total,
            'Manifest members exceed declared uncompressed byte ceiling')
    return record, entries


def stat_identity(current):
    return (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)


def snapshot(stream):
    return stat_identity(os.fstat(stream.fileno()))


def pathname_snapshot(path):
    current = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(current.st_mode), 'Input pathname must still refer to a non-symlink regular file')
    return stat_identity(current)


def hash_stream(stream, expected_bytes):
    result = hashlib.sha256(); size = 0
    for block in iter(lambda: stream.read(BLOCK_BYTES), b''):
        size += len(block)
        require(size <= expected_bytes, 'Stream exceeds its declared byte size')
        result.update(block)
    require(size == expected_bytes, 'Stream byte size differs from manifest')
    return result.hexdigest()


def local_flags(stream, info):
    stream.seek(info.header_offset)
    header = stream.read(30)
    require(len(header) == 30 and header[:4] == b'PK\x03\x04', 'Invalid ZIP local header')
    return struct.unpack_from('<H', header, 6)[0]


def verify(archive_path, manifest_path, output_path):
    archive_path = Path(archive_path); manifest_path = Path(manifest_path); output_path = Path(output_path)
    require(archive_path.is_file() and not archive_path.is_symlink(), 'Archive must be a regular file')
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'External manifest must be a regular file')
    require(not output_path.exists() and not output_path.is_symlink(), 'Fresh receipt only; refusing overwrite')
    require(output_path.suffix == '.json' and output_path.parent.is_dir(), 'Receipt requires an existing directory and JSON filename')
    require(output_path.resolve() not in {archive_path.resolve(), manifest_path.resolve()}, 'Receipt must be outside the ZIP and distinct from inputs')
    original_archive_path = pathname_snapshot(archive_path)
    original_manifest_path = pathname_snapshot(manifest_path)
    raw_manifest = manifest_path.read_bytes()
    body = json.loads(raw_manifest, object_pairs_hook=unique_object, parse_constant=nonfinite)
    record, expected = manifest_record(body)
    total_uncompressed = sum(entry['bytes'] for entry in expected.values())
    with archive_path.open('rb') as stream:
        original = snapshot(stream)
        require(original == original_archive_path, 'Archive pathname changed before identity verification')
        require(original[2] == record['bytes'], 'Archive size differs from manifest')
        archive_digest = hash_stream(stream, record['bytes'])
        require(archive_digest == record['sha256'], 'Whole ZIP SHA-256 differs from manifest')
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream, 'r', allowZip64=True) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                require(len(names) == len(set(names)), 'Duplicate ZIP member path')
                for info in infos:
                    safe_path(info.filename)
                    require(info.orig_filename == info.filename, 'Truncated or altered ZIP member path')
                    require(not info.is_dir(), 'ZIP directory members are outside this file catalog')
                    mode = (info.external_attr >> 16) & 0xffff
                    require(not stat.S_ISLNK(mode), 'ZIP symlink member is forbidden')
                    require(stat.S_IFMT(mode) in (0, stat.S_IFREG), 'ZIP member is not a regular file')
                    require(not (info.flag_bits & (1 | 0x40)), 'Encrypted ZIP member is forbidden')
                    require(not (local_flags(stream, info) & (1 | 0x40)), 'Encrypted ZIP local header is forbidden')
                    require(info.file_size <= record['max_member_bytes'], 'ZIP member exceeds declared byte ceiling')
                require(set(names) == set(expected) and len(infos) == len(expected),
                        'ZIP member inventory differs: missing or extra member')
                require(sum(info.file_size for info in infos) <= record['max_uncompressed_bytes'],
                        'ZIP members exceed declared uncompressed byte ceiling')
                for info in infos:
                    entry = expected[info.filename]
                    require(info.file_size == entry['bytes'], 'ZIP member size differs from exact manifest identity')
                    with archive.open(info, 'r') as member:
                        member_digest = hash_stream(member, entry['bytes'])
                    require(member_digest == entry['sha256'], 'ZIP member SHA-256 differs from exact manifest identity')
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError) as error:
            raise ValueError('ZIP structure or member stream failed integrity verification') from error
        require(snapshot(stream) == original, 'Archive changed during identity verification')
        require(pathname_snapshot(archive_path) == original,
                'Archive pathname identity changed during verification')
    require(manifest_path.read_bytes() == raw_manifest, 'Manifest changed during identity verification')
    require(pathname_snapshot(manifest_path) == original_manifest_path,
            'Manifest pathname identity changed during verification')
    result = dict(status='archive_and_exact_member_identities_verified', scope='identity_only',
        archive_reference_path=record['path'], archive_bytes=record['bytes'], archive_sha256=archive_digest,
        external_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        members_verified=len(expected), member_uncompressed_bytes_verified=total_uncompressed,
        declared_member_byte_ceiling=record['max_member_bytes'],
        declared_uncompressed_byte_ceiling=record['max_uncompressed_bytes'],
        exact_names_sizes_sha256_and_stream_crc_verified=True, safe_member_paths_verified=True,
        stable_requested_input_path_identities_verified=True,
        duplicate_members=0, encrypted_members=0, symlink_members=0,
        extraction_performed=False, network_used=False, model_states_loaded=0,
        numeric_replay_performed=False, training_verified=False, privacy_scan_performed=False,
        manifest_authenticity_verified=False,
        limitation='Identity against the supplied external manifest only. This does not prove manifest authenticity, privacy, scientific numeric replay, training or model inference.')
    with output_path.open('x') as output:
        output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.manifest, args.out), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
