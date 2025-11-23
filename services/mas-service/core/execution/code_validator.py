"""
Code-Validierung und -Korrektur
"""

import re
import ast
from typing import Tuple

def validate_python_code(python_code: str) -> str:
    """Validiert und korrigiert Python-Code"""
    if not python_code or not isinstance(python_code, str):
        return ''
    
    # Entferne Markdown-Code-Blocks (nur am Anfang/Ende)
    python_code = re.sub(r'^```python\s*', '', python_code, flags=re.MULTILINE)
    python_code = re.sub(r'^```json\s*', '', python_code, flags=re.MULTILINE)
    python_code = re.sub(r'^```\s*$', '', python_code, flags=re.MULTILINE)
    python_code = python_code.strip()
    
    # Prüfe auf grundlegende Syntax-Fehler
    try:
        ast.parse(python_code)
    except SyntaxError as e:
        # Versuche Syntax-Fehler zu korrigieren
        python_code = _fix_syntax_errors(python_code, e)
    
    # WICHTIG: Füge KEINE Imports hinzu, wenn sie bereits vorhanden sind
    # Das könnte zu doppelten Zeilen führen
    
    return python_code

def _fix_syntax_errors(code: str, syntax_error: SyntaxError) -> str:
    """Korrigiert bekannte Syntax-Fehler"""
    lines = code.split('\n')
    error_line = syntax_error.lineno - 1 if syntax_error.lineno else 0
    
    # Fehler: except ohne try
    if 'except' in syntax_error.msg.lower() or 'invalid syntax' in syntax_error.msg.lower():
        # Suche nach except ohne try
        for i in range(max(0, error_line - 10), min(len(lines), error_line + 1)):
            line = lines[i].strip()
            # Wenn except gefunden, prüfe ob try davor existiert
            if line.startswith('except'):
                # Suche nach try in den vorherigen Zeilen
                has_try = False
                for j in range(max(0, i - 20), i):
                    if 'try:' in lines[j] or 'try ' in lines[j]:
                        has_try = True
                        break
                
                # Wenn kein try gefunden, entferne except-Block oder füge try hinzu
                if not has_try:
                    # Versuche try-Block zu finden oder entferne except
                    # Suche nach passendem Code-Block
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    
                    # Versuche try vor dem except einzufügen
                    # Finde passende Einrückung
                    if i > 0:
                        prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                        if prev_indent < indent:
                            # Füge try vor except ein
                            try_line = ' ' * prev_indent + 'try:'
                            lines.insert(i, try_line)
                            # Füge pass ein falls nötig
                            if i + 2 < len(lines):
                                next_indent = len(lines[i+2]) - len(lines[i+2].lstrip())
                                if next_indent <= indent:
                                    pass_line = ' ' * (indent - 4) + 'pass'
                                    lines.insert(i+1, pass_line)
                            break
    
    return '\n'.join(lines)

def apply_deterministic_fixes(python_code: str, error_text: str) -> str:
    """Wendet deterministische Korrekturen für bekannte Fehler an"""
    try:
        fixed = python_code
        
        # Bekannte Fehler-Korrekturen
        if 'NameError' in error_text or 'is not defined' in error_text:
            # Versuche fehlende Imports hinzuzufügen
            if 'pd' in fixed and 'import pandas' not in fixed:
                fixed = 'import pandas as pd\n' + fixed
            if 'np' in fixed and 'import numpy' not in fixed:
                fixed = 'import numpy as np\n' + fixed
        
        # Syntax-Fehler: except ohne try
        if 'SyntaxError' in error_text and 'except' in error_text:
            fixed = _fix_except_without_try(fixed)
        
        return fixed
    except Exception:
        return python_code

def _fix_except_without_try(code: str) -> str:
    """Korrigiert except-Blöcke ohne try"""
    lines = code.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Wenn except gefunden, prüfe ob try davor existiert
        if line.startswith('except'):
            # Suche nach try in den letzten 20 Zeilen
            has_try = False
            for j in range(max(0, len(fixed_lines) - 20), len(fixed_lines)):
                if 'try:' in fixed_lines[j] or 'try ' in fixed_lines[j]:
                    has_try = True
                    break
            
            if not has_try:
                # Füge try vor except ein
                indent = len(lines[i]) - len(lines[i].lstrip())
                try_indent = max(0, indent - 4)
                fixed_lines.append(' ' * try_indent + 'try:')
                fixed_lines.append(' ' * (try_indent + 4) + 'pass')
        
        fixed_lines.append(lines[i])
        i += 1
    
    return '\n'.join(fixed_lines)

