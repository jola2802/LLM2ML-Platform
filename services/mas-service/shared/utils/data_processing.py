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
        # Prüfe ob es bereits das gewünschte Format hat (z.B. mit targetVariable, features, etc.)
        # Das sind die erwarteten Felder in einer LLM-Empfehlungs-Response
        expected_keys = ['targetVariable', 'generatedFeatures', 'algorithm', 'hyperparameters', 
                        'features', 'modelType', 'reasoning', 'dataSourceName']
        if any(key in response for key in expected_keys):
            print(f'[JSON Extraction] Response ist bereits ein gültiges JSON-Objekt')
            return response
        
        # Versuche 'result' oder 'content' zu extrahieren
        text = response.get('result', '') or response.get('content', '') or response.get('response', '')
        
        # Wenn 'result' bereits ein Dict ist, gib es zurück
        if isinstance(text, dict):
            print(f'[JSON Extraction] Response.result ist bereits ein Dict')
            # Prüfe ob es das gewünschte Format hat
            if any(key in text for key in expected_keys):
                return text
            # Sonst versuche es als Text zu behandeln
            text = str(text)
        
        # Wenn kein Text gefunden, versuche das gesamte Dict zu verwenden
        if not text:
            # Prüfe ob das gesamte Dict bereits das gewünschte Format hat
            if any(key in response for key in expected_keys):
                print(f'[JSON Extraction] Response-Dict hat bereits das gewünschte Format')
                return response
            text = str(response)
    else:
        text = str(response)
    
    if not text:
        raise ValueError('Leere Response vom LLM erhalten')
    
    # Debug: Zeige ersten 500 Zeichen der Response
    print(f'[JSON Extraction] Response (erste 400 Zeichen): {text[:400]}...')
    
    # Versuche direktes JSON-Parsing
    try:
        parsed = json.loads(text)
        print(f'[JSON Extraction] JSON erfolgreich direkt geparst')
        return parsed
    except json.JSONDecodeError:
        pass
    
    # Versuche JSON-Extraktion mit Regex (suche nach JSON-Objekten)
    # Suche nach geschweiften Klammern, die JSON enthalten könnten
    json_patterns = [
        r'\{[\s\S]*\}',  # Einfaches JSON-Objekt
        r'```json\s*(\{[\s\S]*?\})\s*```',  # JSON in Code-Block
        r'```\s*(\{[\s\S]*?\})\s*```',  # JSON in Code-Block ohne json-Tag
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if json_match:
            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
            try:
                result = json.loads(json_str)
                print(f'[JSON Extraction] JSON erfolgreich extrahiert mit Pattern: {pattern[:30]}...')
                return result
            except json.JSONDecodeError as e:
                print(f'[JSON Extraction] Fehler beim Parsen des extrahierten JSON: {e}')
                continue
    
    # Fallback: Versuche, JSON am Anfang oder Ende zu finden
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                parsed = json.loads(line)
                print(f'[JSON Extraction] JSON erfolgreich aus Zeile extrahiert')
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Versuche, mehrzeiliges JSON zu finden (kann über mehrere Zeilen gehen)
    # Suche nach dem ersten { und dem letzten }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace:last_brace + 1]
        try:
            parsed = json.loads(json_candidate)
            print(f'[JSON Extraction] JSON erfolgreich aus mehrzeiligem Text extrahiert')
            return parsed
        except json.JSONDecodeError as e:
            print(f'[JSON Extraction] Fehler beim Parsen des Kandidaten: {e}')
            # Versuche, abgeschnittenes JSON zu reparieren
            try:
                # Wenn das JSON abgeschnitten ist, versuche es zu vervollständigen
                # Suche nach dem letzten vollständigen Key-Value-Paar
                repaired_json = _repair_truncated_json(json_candidate)
                if repaired_json:
                    parsed = json.loads(repaired_json)
                    print(f'[JSON Extraction] Abgeschnittenes JSON erfolgreich repariert')
                    return parsed
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

