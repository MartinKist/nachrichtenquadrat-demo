# Kommunikation-Demo: Schulz von Thun's Vier-Seiten-Modell

*Hinweis: Dieses README ist KI-generiert*

Dieses Projekt ist eine interaktive Demo, die das Kommunikationsmodell von Schulz von Thun (Vier-Seiten-Modell) illustriert. Es ermöglicht eine fiktive Konversation mit einer KI, die Nachrichten auf die vier Seiten analysiert und darauf basierend antwortet.

## Installation

1. **Voraussetzungen**:
    - Python 3.9 oder höher
    - [Einen OopenAI API Key](https://platform.openai.com/)

2. **Repository klonen**:
   ```bash
   git clone https://github.com/MartinKist/nachrichtenquadrat-demo.git
   cd nachrichtenquadrat-demo
   ```

3. **Virtuelle Umgebung erstellen und aktivieren**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Für Linux/Mac
    # .\venv\Scripts\activate  # Für Windows
    ```

4. **Abhängigkeiten installieren**:
    ```bash
    pip install -r requirements.txt
    ```

5. API-Key als Umgebungsvariable setzen

    **Für Linux/Mac**:
    ```bash
    export OPENAI_API_KEY=<dein API-Key>
    ```

    **Für Windows** (PowerShell):
    ```powershell
    $env:OPENAI_API_KEY="<dein API-Key>"
    ```

    **Für Windows** (Eingabeaufforderung):
    ```cmd
    set OPENAI_API_KEY=<dein API-Key>
    ```

## Bedienung

1. **Starten der Demo**:
   ```bash
   python demo.py
   ```

2. **Interaktion**:
   - Beschreibe die Situation, in der die Kommunikation stattfinden soll.
   - Passe die Sensitivität der "Ohren" (Sach-, Beziehungs-, Selbstoffenbarungs-, Appell-Ohr) der KI an.
   - Führe eine Konversation mit der KI, indem du Nachrichten eingibst.
   - Die KI analysiert deine Nachrichten und antwortet entsprechend.

3. **Optionale Funktionen**:
   - Generiere eine Illustration der beschriebenen Situation.

4. **Beenden**:
   - Gib `exit` ein, um die Demo zu beenden.

## Architektur

Das Projekt besteht aus mehreren Komponenten:

1. **Agenten**:
   - `message_analyzer`: Analysiert die Nachricht des Nutzers basierend auf dem Vier-Seiten-Modell.
   - `communication_partner`: Reagiert auf die analysierte Nachricht und simuliert einen menschlichen Kommunikationspartner.
   - `evaluator`: Bewertet die Antwort des Kommunikationspartners und gibt Feedback zur Verbesserung.

2. **Datenstrukturen**:
   - `MessageSide`: Repräsentiert eine der vier Seiten einer Nachricht (Sach-, Selbstoffenbarungs-, Beziehungs-, Appellseite).
   - `AnalyzedMessage`: Enthält die Analyse einer Nachricht.
   - `EvaluationFeedback`: Enthält Feedback zur Antwortqualität.

3. **Workflow**:
   - Der Nutzer gibt eine Nachricht ein.
   - Die Nachricht wird analysiert (`message_analyzer`).
   - Basierend auf der Analyse generiert der Kommunikationspartner (`communication_partner`) eine Antwort.
   - Die Antwort wird vom Evaluator (`evaluator`) überprüft und ggf. verbessert.

4. **Interaktive Schleife**:
   - Der Nutzer interagiert mit der KI in einer Schleife, bis er die Demo beendet.

## Beispiel

```markdown
Dies ist eine Demo, um das Vier-Seiten-Modell von Schulz von Thun zu veranschaulichen.  
Die Demo besteht aus einem fiktiven Gespräch, das du mit einer KI führen kannst.

Möchtest du die "Ohren" der KI anpassen? (j/n) j  
Wie sensibel soll das Sach-Ohr der KI sein (0-1)? 0.5  
Wie sensibel soll das Beziehungs-Ohr der KI sein (0-1)? 0.8  
Wie sensibel soll das Selbstoffenbarungs-Ohr der KI sein (0-1)? 0.3  
Wie sensibel soll das Appell-Ohr der KI sein (0-1)? 0.6  

Beschreibe nun die Situation, in der das Rollenspiel stattfinden soll:  
Eine Diskussion zwischen zwei Kollegen über eine verpasste Deadline.

Du sagst: "Ich denke, du hättest mich früher über die Verzögerung informieren sollen."  
Vier-Seiten-Analyse:  
- **Sachseite**: Die Deadline wurde verpasst. (Gewichtung: 0.8)  
- **Beziehungsseite**: Der Sender fühlt sich enttäuscht. (Gewichtung: 0.7)  
- **Selbstoffenbarungsseite**: Der Sender schätzt rechtzeitige Kommunikation. (Gewichtung: 0.6)  
- **Appellseite**: Der Sender möchte, dass der Empfänger die Kommunikation verbessert. (Gewichtung: 0.9)  

Die KI sagt: "Ich verstehe deinen Frust, aber ich denke, wir hätten beide besser kommunizieren können."
```

## Lizenz

Dieses Projekt steht unter der GPL-3.0-Lizenz. Weitere Informationen findest du in der Datei `LICENSE`.
