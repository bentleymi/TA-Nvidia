#!/usr/bin/env python3
"""Shared collection logic for TA-Nvidia scripted inputs."""

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

NUM_RE = re.compile(
    r"([+-]?(?:\d+\.?\d*|\.\d+))(?: *([A-Za-z%°/]+(?:/[A-Za-z%°/]+)*)?)?"
)

PRUNE_PREFIXES = ("supported_clocks.",)


def keep_key(path):
    return not path.startswith(PRUNE_PREFIXES)


def run_nvidia_smi(args):
    proc = subprocess.run(
        ["nvidia-smi"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def flatten_element(elem, prefix, out):
    children = list(elem)
    if not children:
        text = (elem.text or "").strip()
        if text and keep_key(prefix):
            out[prefix] = text
        return
    counts = {}
    for child in children:
        counts[child.tag] = counts.get(child.tag, 0) + 1
    seen = {}
    for child in children:
        path = child.tag if not prefix else prefix + "." + child.tag
        key = path
        if counts[child.tag] > 1:
            idx = seen.get(child.tag, 0)
            seen[child.tag] = idx + 1
            path = "%s_%d" % (path, idx)
        if keep_key(path):
            for attr_name, attr_value in sorted(child.attrib.items()):
                out["%s_%s" % (path, attr_name)] = attr_value
        flatten_element(child, path, out)


def parse_dump(xml_text):
    root = ET.fromstring(xml_text)
    scalars = {}
    gpus = []
    for child in root:
        if child.tag == "gpu":
            flat = {}
            for attr_name, attr_value in sorted(child.attrib.items()):
                flat["gpu_" + attr_name] = attr_value
            flatten_element(child, "", flat)
            gpus.append(flat)
        else:
            for attr_name, attr_value in sorted(child.attrib.items()):
                scalars["%s_%s" % (child.tag, attr_name)] = attr_value
            flatten_element(child, child.tag, scalars)
    return scalars, gpus


def numeric_leaf(value):
    match = NUM_RE.fullmatch(value.strip())
    if not match:
        return None, None
    try:
        number = float(match.group(1))
    except ValueError:
        return None, None
    unit = (match.group(2) or "").strip()
    return number, unit


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def get_compute_processes():
    fields = "gpu_uuid,pid,process_name,used_gpu_memory"
    rc, out, _err = run_nvidia_smi(
        ["--query-compute-apps=%s" % fields, "--format=csv,noheader,nounits"]
    )
    if rc != 0:
        rc, out, _err = run_nvidia_smi(
            ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"]
        )
        fields = "gpu_uuid,pid,process_name,used_memory"
    if rc != 0:
        return []
    keys = [k.strip() for k in fields.split(",")]
    processes = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(", ")]
        if len(parts) != len(keys):
            continue
        processes.append(dict(zip(keys, parts)))
    return processes


def collect():
    rc, xml_text, err = run_nvidia_smi(["-q", "-x"])
    if rc != 0:
        raise RuntimeError("nvidia-smi -q -x failed (rc=%d): %s" % (rc, err))
    scalars, gpus = parse_dump(xml_text)
    gpus = [g for g in gpus if g.get("uuid", "").startswith("GPU-") or g.get("gpu_uuid", "").startswith("GPU-")]
    processes = get_compute_processes()
    return scalars, gpus, processes


def emit_events(scalars, gpus, processes):
    for index, gpu in enumerate(gpus):
        event = {
            "driver_version": scalars.get("driver_version", ""),
            "nvidia_smi_timestamp": scalars.get("timestamp", ""),
            "gpu_index": str(index),
        }
        event.update(gpu)
        uuid = gpu.get("uuid") or gpu.get("gpu_uuid", "")
        event["compute_processes"] = [p for p in processes if p.get("gpu_uuid") == uuid]
        sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def emit_metrics(scalars, gpus, processes):
    driver_version = scalars.get("driver_version", "")
    for index, gpu in enumerate(gpus):
        event = {
            "gpu_index": str(index),
            "gpu_uuid": gpu.get("uuid") or gpu.get("gpu_uuid", ""),
            "gpu_name": gpu.get("product_name", ""),
            "driver_version": driver_version,
        }
        for key, value in gpu.items():
            number, _unit = numeric_leaf(value)
            if number is None:
                continue
            event[sanitize(key)] = number
        sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main(mode):
    try:
        scalars, gpus, processes = collect()
    except Exception as exc:
        sys.stderr.write("TA-Nvidia collection error: %s\n" % exc)
        return 2
    if mode == "metrics":
        emit_metrics(scalars, gpus, processes)
    else:
        emit_events(scalars, gpus, processes)
    return 0
