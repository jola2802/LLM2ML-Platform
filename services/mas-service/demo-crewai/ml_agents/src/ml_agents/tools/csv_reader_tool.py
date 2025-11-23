from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pandas as pd
import os


class CsvReaderToolInput(BaseModel):
    """Input schema for CsvReaderTool."""
    csv_path: str = Field(..., description="Der Pfad zur CSV-Datei, die gelesen werden soll.")


class CsvReaderTool(BaseTool):
    name: str = "csv_reader"
    description: str = (
        "Liest eine CSV-Datei EINMAL und gibt eine vollständige Datenanalyse zurück. "
        "WICHTIG: Rufe dieses Tool nur EINMAL pro CSV-Datei auf. Die Antwort enthält alle notwendigen Informationen. "
        "Wenn du die Daten bereits geladen hast, verwende die erhaltenen Informationen direkt für deine Analyse."
    )
    args_schema: Type[BaseModel] = CsvReaderToolInput

    def _run(self, csv_path: str) -> str:
        """Liest eine CSV-Datei und gibt eine Zusammenfassung zurück."""
        try:
            # Prüfe, ob die Datei existiert
            if not os.path.exists(csv_path):
                return f"FEHLER: Die Datei '{csv_path}' wurde nicht gefunden. Bitte überprüfe den Pfad."
            
            # Lese die CSV-Datei
            df = pd.read_csv(csv_path)
            
            # Prüfe auf fehlende Werte
            missing_values = df.isnull().sum()
            missing_info = ""
            if missing_values.sum() > 0:
                missing_info = f"\nFehlende Werte pro Spalte:\n{missing_values[missing_values > 0].to_string()}\n"
            else:
                missing_info = "\nKeine fehlenden Werte gefunden.\n"
            
            # Erstelle eine kompakte aber vollständige Zusammenfassung
            summary = f"""✅ CSV-DATEI ERFOLGREICH GELADEN

📊 DATENÜBERSICHT:
- Datei: {os.path.basename(csv_path)}
- Anzahl Zeilen: {len(df)}
- Anzahl Spalten: {len(df.columns)}
- Spaltennamen: {', '.join(df.columns.tolist())}

📋 DATENTYPEN:
{df.dtypes.to_string()}
{missing_info}
📈 STATISTISCHE ZUSAMMENFASSUNG:
{df.describe().to_string()}

📝 ERSTE 10 ZEILEN (Beispiel):
{df.head(10).to_string()}

✅ ANALYSE ABGESCHLOSSEN: Du hast jetzt alle notwendigen Informationen über die Daten. 
Verwende diese Informationen für deine weitere Analyse. Rufe dieses Tool NICHT erneut auf."""
            return summary
        except pd.errors.EmptyDataError:
            return f"FEHLER: Die CSV-Datei '{csv_path}' ist leer."
        except pd.errors.ParserError as e:
            return f"FEHLER: Die CSV-Datei '{csv_path}' konnte nicht geparst werden: {str(e)}"
        except Exception as e:
            return f"FEHLER beim Lesen der CSV-Datei: {str(e)}"

