"""
Datenverarbeitungs-Utilities
"""

import json
import re
from typing import Optional, Dict, Any, List

def filter_columns(
    columns: List[str],
    selected: Optional[List[str]] = None,
    excluded: Optional[List[str]] = None
) -> List[str]:
    """Filtert Spalten basierend auf selected/excluded"""
    if not columns:
        return []
    
    filtered = columns.copy()
    
    if selected:
        filtered = [c for c in filtered if c in selected]
    
    if excluded:
        filtered = [c for c in filtered if c not in excluded]
    
    return filtered

def filter_data_overview_for_features(
    llm_summary: str,
    selected: Optional[List[str]] = None,
    excluded: Optional[List[str]] = None
) -> str:
    """Filtert Data Overview für Features"""
    # Vereinfachte Implementierung - kann später erweitert werden
    return llm_summary

def extract_and_validate_json(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extrahiert und validiert JSON aus LLM-Response"""
    
    # Wenn Response bereits ein Dict ist und direkt JSON enthält, gib es zurück
    if isinstance(response, dict):
        expected_keys = ['targetVariable', 'generatedFeatures', 'algorithm', 'hyperparameters', 
                        'features', 'modelType', 'reasoning', 'dataSourceName']
        if any(key in response for key in expected_keys):
            print(f'[JSON Extraction] Response ist bereits ein gültiges JSON-Objekt')
            return _normalize_json_values(response)
        
        text = response.get('result', '') or response.get('content', '') or response.get('response', '')
        
        if isinstance(text, dict):
            print(f'[JSON Extraction] Response.result ist bereits ein Dict')
            if any(key in text for key in expected_keys):
                return _normalize_json_values(text)
            text = str(text)
        
        if not text:
            # Prüfe ob das gesamte Dict bereits das gewünschte Format hat
            if any(key in response for key in expected_keys):
                print(f'[JSON Extraction] Response-Dict hat bereits das gewünschte Format')
                return _normalize_json_values(response)
            text = str(response)
    else:
        text = str(response)
    
    if not text:
        raise ValueError('Leere Response vom LLM erhalten')
    
    text = _clean_json_text(text)
    
    # Debug: Zeige ersten 500 Zeichen
    print(f'[JSON Extraction] Response (erste 500 Zeichen der summary): {text[:500]}...')
    
    # Versuche direktes JSON-Parsing
    try:
        parsed = json.loads(text)
        print(f'[JSON Extraction] JSON erfolgreich direkt geparst')
        return _normalize_json_values(parsed)
    except json.JSONDecodeError:
        # Versuche mit object_pairs_hook (behandelt doppelte Keys automatisch)
        try:
            def remove_duplicates(pairs):
                """Behält nur den letzten Wert bei doppelten Keys"""
                result = {}
                for key, value in pairs:
                    result[key] = value  # Überschreibt automatisch bei doppelten Keys
                return result
            parsed = json.loads(text, object_pairs_hook=remove_duplicates)
            print(f'[JSON Extraction] JSON erfolgreich geparst (doppelte Keys entfernt)')
            return _normalize_json_values(parsed)
        except json.JSONDecodeError:
            pass
    
    # Versuche JSON-Extraktion mit Regex (suche nach JSON-Objekten)
    # Suche nach geschweiften Klammern, die JSON enthalten könnten
    json_patterns = [
        r'```json\s*(\{[\s\S]*?\})\s*```',  # JSON in Code-Block
        r'```\s*(\{[\s\S]*?\})\s*```',  # JSON in Code-Block ohne json-Tag
        r'\{[\s\S]*\}',  # Einfaches JSON-Objekt
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if json_match:
            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
            # Bereinige JSON-String
            json_str = _clean_json_text(json_str)
            try:
                result = json.loads(json_str)
                print(f'[JSON Extraction] JSON erfolgreich extrahiert mit Pattern: {pattern[:30]}...')
                return _normalize_json_values(result)
            except json.JSONDecodeError as e:
                print(f'[JSON Extraction] Fehler beim Parsen des extrahierten JSON: {e}')
                continue
    
    # Fallback: Versuche, JSON am Anfang oder Ende zu finden
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                cleaned_line = _clean_json_text(line)
                parsed = json.loads(cleaned_line)
                print(f'[JSON Extraction] JSON erfolgreich aus Zeile extrahiert')
                return _normalize_json_values(parsed)
            except json.JSONDecodeError:
                continue
    
    # Versuche, mehrzeiliges JSON zu finden (kann über mehrere Zeilen gehen)
    # Suche nach dem ersten { und dem letzten }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace:last_brace + 1]
        try:
            # Bereinige JSON-String
            json_candidate = _clean_json_text(json_candidate)
            # Versuche zuerst, doppelte Keys zu entfernen (falls vorhanden)
            cleaned_json = _remove_duplicate_keys(json_candidate)
            parsed = json.loads(cleaned_json)
            print(f'[JSON Extraction] JSON erfolgreich aus mehrzeiligem Text extrahiert')
            return _normalize_json_values(parsed)
        except json.JSONDecodeError as e:
            print(f'[JSON Extraction] Fehler beim Parsen des Kandidaten: {e}')
            # Versuche, abgeschnittenes JSON zu reparieren
            try:
                # Wenn das JSON abgeschnitten ist, versuche es zu vervollständigen
                # Suche nach dem letzten vollständigen Key-Value-Paar
                repaired_json = _repair_truncated_json(json_candidate)
                if repaired_json:
                    cleaned_json = _remove_duplicate_keys(repaired_json)
                    parsed = json.loads(cleaned_json)
                    print(f'[JSON Extraction] Abgeschnittenes JSON erfolgreich repariert')
                    return _normalize_json_values(parsed)
            except Exception as repair_error:
                print(f'[JSON Extraction] JSON-Reparatur fehlgeschlagen: {repair_error}')
    
    # Wenn kein JSON gefunden, gib Debug-Informationen
    print(f'[JSON Extraction] FEHLER: Kein gültiges JSON gefunden')
    print(f'[JSON Extraction] Response-Typ: {type(response)}')
    print(f'[JSON Extraction] Response-Länge: {len(text)} Zeichen')
    print(f'[JSON Extraction] Response (erste 1000 Zeichen): {text[:1000]}')
    if len(text) > 1000:
        print(f'[JSON Extraction] Response (letzte 500 Zeichen): ...{text[-500:]}')
    raise ValueError(f'Kein gültiges JSON in LLM-Response gefunden. Response: {text[:500]}...')

def _clean_json_text(text: str) -> str:
    """Bereinigt JSON-Text: Entfernt doppelte geschweifte Klammern (Markdown-Escaping)"""
    # Ersetze doppelte geschweifte Klammern durch einfache
    # {{ wird zu {, }} wird zu }
    text = text.replace('{{', '{').replace('}}', '}')
    return text

def _normalize_json_values(obj: Any) -> Any:
    """Normalisiert JSON-Werte: Konvertiert None-Strings zu None, extrahiert Listen zu einzelnen Werten"""
    if isinstance(obj, dict):
        return {key: _normalize_json_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Wenn Liste mit einem Element, extrahiere das Element (für Hyperparameter)
        if len(obj) == 1:
            return _normalize_json_values(obj[0])
        return [_normalize_json_values(item) for item in obj]
    elif isinstance(obj, str):
        # Konvertiere "None" (String) zu Python None
        if obj.strip() == 'None' or obj.strip() == 'null':
            return None
        # Konvertiere "True"/"False" zu bool
        if obj.strip() == 'True':
            return True
        if obj.strip() == 'False':
            return False
        return obj
    else:
        return obj

def _remove_duplicate_keys(json_str: str) -> str:
    """Entfernt doppelte Keys aus JSON-String (behält den letzten Wert)"""
    try:
        # Versuche zuerst direktes Parsing
        parsed = json.loads(json_str)
        # Wenn erfolgreich, gibt es keine doppelten Keys (JSON-Parser behält automatisch den letzten)
        return json_str
    except json.JSONDecodeError as e:
        # Wenn Parsing fehlschlägt, versuche doppelte Keys manuell zu entfernen
        # Das Problem: JSON erlaubt keine doppelten Keys, aber LLMs können sie generieren
        
        # Strategie: Finde alle Key-Value-Paare und behalte nur das letzte Vorkommen jedes Keys
        # Pattern: "key": value (kann über mehrere Zeilen gehen)
        key_pattern = r'"([^"]+)":\s*'
        
        # Finde alle Keys und ihre Positionen
        keys_positions = []
        for match in re.finditer(key_pattern, json_str):
            key = match.group(1)
            start_pos = match.start()
            keys_positions.append((key, start_pos))
        
        # Finde doppelte Keys (behalte nur die letzte Position)
        seen_keys = {}
        for key, pos in keys_positions:
            seen_keys[key] = pos
        
        # Wenn keine doppelten Keys gefunden, gib Original zurück
        if len(keys_positions) == len(seen_keys):
            return json_str
        
        # Erstelle neuen JSON-String ohne doppelte Keys
        # Das ist komplex - versuche stattdessen einen einfacheren Ansatz:
        # Entferne Zeilen, die doppelte Keys enthalten (grobe Näherung)
        lines = json_str.split('\n')
        seen_keys_in_lines = {}
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            key_match = re.search(key_pattern, line)
            if key_match:
                key = key_match.group(1)
                if key in seen_keys_in_lines:
                    # Überspringe diese Zeile (doppelter Key)
                    continue
                seen_keys_in_lines[key] = i
            cleaned_lines.append(line)
        
        cleaned_json = '\n'.join(cleaned_lines)
        
        # Versuche erneut zu parsen
        try:
            json.loads(cleaned_json)
            return cleaned_json
        except json.JSONDecodeError:
            # Wenn immer noch Fehler, versuche mit json.loads object_hook
            # Das ist der beste Ansatz: Python's JSON-Parser behält automatisch den letzten Wert
            try:
                # Verwende object_hook um doppelte Keys zu handhaben
                def remove_duplicates(pairs):
                    result = {}
                    for key, value in pairs:
                        result[key] = value  # Überschreibt automatisch bei doppelten Keys
                    return result
                
                parsed = json.loads(cleaned_json, object_pairs_hook=remove_duplicates)
                return json.dumps(parsed, ensure_ascii=False)
            except:
                # Wenn alles fehlschlägt, gib Original zurück
                return json_str

def _repair_truncated_json(json_str: str) -> Optional[str]:
    """Versucht, abgeschnittenes JSON zu reparieren"""
    if not json_str or not json_str.strip().startswith('{'):
        return None
    
    # Zähle öffnende und schließende Klammern
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    # Wenn mehr öffnende als schließende Klammern, füge fehlende hinzu
    if open_braces > close_braces:
        missing_braces = open_braces - close_braces
        # Versuche, das JSON zu vervollständigen
        repaired = json_str.rstrip()
        
        # Entferne möglicherweise abgeschnittene Strings
        # Suche nach dem letzten vollständigen Komma oder Doppelpunkt
        last_comma = repaired.rfind(',')
        last_colon = repaired.rfind(':')
        
        if last_comma > last_colon and last_comma > len(repaired) - 100:
            # Entferne den abgeschnittenen Teil nach dem letzten Komma
            repaired = repaired[:last_comma]
        
        # Füge fehlende schließende Klammern hinzu
        repaired += '}' * missing_braces
        
        # Prüfe, ob es jetzt gültiges JSON ist
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
    
    return None

def build_data_overview(
    analysis: Dict[str, Any],
    selected_features: Optional[List[str]] = None,
    excluded_features: Optional[List[str]] = None
) -> str:
    """Baut Data Overview aus Analyse-Daten"""
    columns = analysis.get('columns', [])
    filtered_columns = filter_columns(columns, selected_features, excluded_features)
    
    return (
        f"DATA INFORMATION:\n"
        f"- Available columns: {', '.join(filtered_columns)}\n"
        f"- Number of rows: {analysis.get('rowCount', 0)}"
    )

