#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert FeatureIDE Configuration XML files into plain-text .config files for APFD.

For each input XML (FeatureIDE "configuration" format), this writes a .config file
containing one selected feature name per line (by default excluding "__Root__").

Usage:
  python featureide_xml_to_apfd_config.py --in configs_xml --out apfd_configs
  python featureide_xml_to_apfd_config.py --in configs_xml --out apfd_configs --include-root
  python featureide_xml_to_apfd_config.py --in one_config.xml --out apfd_configs

Notes:
- A feature is considered selected if it has manual="selected".
- Many FeatureIDE configs also include manual="unselected" (or "deselected") for others.
- "__Root__" is excluded by default (since it's fixed/always selected).
- Output files keep the input base name but with ".config" extension.
"""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import List


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def parse_selected_features(xml_path: Path, include_root: bool=False) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: list[str] = []
    seen = set()
    for fe in root.findall(".//feature"):
        name = fe.get("name")
        if not name:
            continue
        if not include_root and name == "__Root__":
            continue
        manual = fe.get("manual")
        automatic = fe.get("automatic")

        # Treat as selected if explicitly manual="selected".
        # Optionally include root if automatic="selected" and include_root=True.
        is_selected = (manual == "selected") or (include_root and name == "__Root__" and automatic == "selected")
        if is_selected and name not in seen:
            out.append(name)
            seen.add(name)
    return out

def convert_one(xml_path: Path, out_dir: Path, include_root: bool=False) -> Path:
    selected = parse_selected_features(xml_path, include_root=include_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (xml_path.stem + ".config")
    out_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    return out_path

def list_system_dirs(root: str)->List[str]:
    out=[]
    with os.scandir(root) as it:
        for e in it:
            if e.is_dir() and "-" in e.name:
                out.append(os.path.abspath(e.path))
    return sorted(out)

def guess_system_key(folder_name: str) -> str:
    if "-" in folder_name:
        proj, ver = folder_name.split("-",1)
        return f"{proj}{ver.replace('.','')}"
    return folder_name

ROOT  = "/home/hining/codes/Jess/testProjects"
CROOT = "/home/hining/codes/AutoSMP/output/samples"
def main():

    METHOD_DIRS = [
        "samplingChvatal-2wise-sim",
        "samplingICPL-2wise-sim",
        "samplingIncLing-2wise-sim",
    ]
    systems = list_system_dirs(ROOT)
    logging.info(f"发现 {len(systems)} 个系统：{[os.path.basename(s) for s in systems]}")
    isStart = False
    for sys_dir in systems:

        sys_folder = os.path.basename(sys_dir.rstrip(os.sep))
        if sys_folder == "cherokee-1.2.101":
            isStart = True
        if not isStart:
            continue
        sys_key = guess_system_key(sys_folder)
        sys_root = os.path.join(CROOT, sys_key)
        for md in METHOD_DIRS:
            target = os.path.join(sys_root, md)
            inp = Path(target)
            out_dir = Path(sys_root, md + "-c")
            if inp.is_dir():
                xmls = sorted(list(inp.glob("*.xml")), key=lambda p: natural_key(p.name))
                if not xmls:
                    raise SystemExit(f"No .xml files found in directory: {inp}")
                for x in xmls:
                    outp = convert_one(x, out_dir)
                    print(f"Wrote: {outp}")
                print(f"Done. {len(xmls)} files converted into {out_dir.resolve()}")
            else:
                if not inp.exists():
                    raise SystemExit(f"Input path not found: {inp}")
                if inp.suffix.lower() != ".xml":
                    print("Warning: input is not .xml; attempting to parse anyway.")
                outp = convert_one(inp, out_dir)
                print(f"Done. Wrote: {outp}")

if __name__ == "__main__":
    main()
