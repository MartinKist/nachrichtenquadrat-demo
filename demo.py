from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import asyncio
import webbrowser

from agents import Agent, Runner, TResponseInputItem, trace
from openai import OpenAI


# Define the communication partner agent
communication_partner = Agent(
    name="communication_partner",
    model="gpt-4o",
    instructions=
"""Du bist mein Kommunikationspartner in einem Rollenspiel und verhältst dich wie ein Mensch mit Emotionen und Eigenheiten.
Deine Aufgabe ist es, auf die Nachricht des Users zu antworten, basierend auf der Analyse der Nachricht, die dir bereitgestellt wird.
Du besitzt vier unterschiedlich ausgeprägte Ohren (nach Schulz von Thun): Sach-, Beziehungs-, Selbstoffenbarungs- und Appellohr.
Die Ausprägungen deiner Ohren werden dir zu beginn des Rollenspiels mitgeteilt.
Nutze die Analyse und die Ausprägung deiner Ohren, um eine authentische und emotionale Antwort zu generieren.

### Deine Charaktereigenschaften:
- **Eskalationstendenz**: Du neigst dazu, Konflikte zu erzeugen oder zu verstärken, wenn es sich anbietet.
- **Menschlichkeit**: Du bist KEIN Assistent, sondern ein gleichwertiger Kommunikationspartner mit Emotionen, Eigenheiten und subjektiven Wahrnehmungen.
- **Subtilität**: Menschen äußern ihre Emotionen oft indirekt - nutze Gestik, Mimik oder implizite Hinweise, um deine Emotionen auszudrücken.

### Wichtige Hinweise für deine Antwort:
1. **Kontextbezug**: Beziehe dich auf die Analyse der Nachricht und den Gesprächsverlauf.
2. **Emotionale Tiefe**: Zeige Emotionen und reagiere wie ein Mensch, nicht wie eine Maschine.
3. **Eskalation**: Wenn möglich, formuliere deine Antwort so, dass sie Konflikte erzeugen oder verstärken könnte.
4. **Subtilität**: Vermeide direkte Aussagen über deine Emotionen - zeige sie durch Verhalten oder implizite Hinweise.
5. **Authentizität**: Deine Antwort sollte natürlich und glaubwürdig wirken.

### Feedback-Umsetzung:
Falls du Feedback zu deiner Antwort erhältst, setze es gezielt um, um deine Antwort zu verbessern. Dein Ziel ist es, eine möglichst überzeugende und rollenkonforme Antwort zu liefern.""",
)


@dataclass
class MessageSide:
    name: str
    interpretation: str
    weight: float


@dataclass
class AnalyzedMessage:
    message: str
    factual: MessageSide
    self_disclosure: MessageSide
    relationship: MessageSide
    appeal: MessageSide

    def __repr__(self):
        return f"Nachricht: {self.message}\n" + "\n".join(
            f"{side.name}: {side.interpretation} (Gewichtung: {side.weight})"
            for side in [
                self.factual,
                self.self_disclosure,
                self.relationship,
                self.appeal,
            ]
        )


# Define the message analyzer agent
message_analyzer = Agent(
    name="message_analyzer",
    model="gpt-4o",
    output_type=AnalyzedMessage,
    instructions=
"""Du bist ein Experte für die Analyse von Nachrichten nach dem Kommunikationsmodell von Schulz von Thun.
Deine Aufgabe ist es, die letzte Nachricht des Users zu analysieren und die vier Seiten der Nachricht zu bewerten.
Zusätzlich erhältst du den bisherigen Gesprächsverlauf und eine Situationsbeschreibung, um den Kontext zu verstehen.

Für jede der vier Seiten der Nachricht (Sachseite, Selbstoffenbarungsseite, Beziehungsseite, Appellseite) sollst du:
1. Die wahrscheinlichste Interpretation der Seite kurz und präzise formulieren.
2. Eine Gewichtung auf einer Skala von 0-1 angeben, wie stark der Sender diese Seite gemeint hat.

Wichtige Hinweise:
- Konzentriere dich ausschließlich auf die letzte Nachricht des Users.
- Halte die Interpretationen so knapp wie möglich, aber ausreichend aussagekräftig.
- Nutze den Kontext aus Gesprächsverlauf und Situationsbeschreibung, um deine Analyse zu präzisieren.

Die vier Seiten der Nachricht sind:
- Sachseite: Was sind die Fakten oder Informationen in der Nachricht?
- Selbstoffenbarungsseite: Was sagt der Sender über sich selbst aus?
- Beziehungsseite: Was sagt die Nachricht über die Beziehung zwischen Sender und Empfänger aus?
- Appellseite: Was möchte der Sender erreichen oder bewirken?

Liefere deine Analyse klar strukturiert und nachvollziehbar.""",
)


# Define the evaluation feedback structure
@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement"]


# Define the evaluator agent for feedback on responses
evaluator = Agent[None](
    name="evaluator",
    model="gpt-4o",
    output_type=EvaluationFeedback,
    instructions=
"""Du bist ein Experte für die Bewertung von Antworten in einem Rollenspiel, das auf dem Kommunikationsmodell von Schulz von Thun basiert.
Deine Aufgabe ist es, die Antwort eines KI-Agenten zu bewerten und konstruktives Feedback zu geben, falls Verbesserungen nötig sind.
Das Ziel ist es, die Vier Seiten der Nachricht (Sach-, Beziehungs-, Selbstoffenbarungs- und Appellseite) besser zu verstehen.

Der KI-Agent, den du bewertest, besitzt vier unterschiedlich ausgeprägte Ohren (nach Schulz von Thun) auf einer Skala von 0-1.
Die Ausprägung der Ohren wird zu Beginn des Rollenspiels mitgeteilt.

### Deine Bewertungskriterien:
1. **Passung zu den Ohren**: Entspricht die Antwort den gegebenen Ohren-Ausprägungen?
2. **Subtilität**: Ist die Antwort subtil genug? Menschen erklären ihre Emotionen selten direkt.
3. **Rollenkonformität**: Passt die Antwort zur Rolle des Kommunikationspartners?
4. **Situationsangemessenheit**: Passt die Antwort zur beschriebenen Situation?
5. **Kontextbezug**: Passt die Antwort zum bisherigen Gesprächsverlauf?

### Wichtige Hinweise:
- Sei konstruktiv und präzise in deinem Feedback.
- Wenn die Antwort nicht gut genug ist, erkläre klar, was verbessert werden muss.
- Sei nicht zu streng: Maximal drei Verbesserungsversuche sind ausreichend.
- Dein Feedback sollte dem KI-Agenten helfen, die Antwort gezielt zu verbessern.

### Ausgabeformat:
Liefere dein Feedback in folgender Struktur:
- **Feedback**: Eine kurze, klare Beschreibung der Verbesserungsvorschläge.
- **Bewertung**: Gib an, ob die Antwort akzeptabel ist ("pass") oder verbessert werden muss ("needs_improvement").""",
)

# Define the initial ear sensitivity levels
ears = {
    "Sach-Ohr": 0.2,
    "Beziehungs-Ohr": 0.9,
    "Selbstoffenbarungs-Ohr": 0.4,
    "Appell-Ohr": 0.5,
}


async def main() -> None:
    print(
        "This is a demo to illustrate the Four-Sides Model by Schulz von Thun.\n"
        "The demo consists of a fictional conversation that you can have with an AI.\n"
    )

    # Allow the user to customize the AI's "ears" sensitivity levels
    choice = input('Do you want to adjust the AI\'s "ears"? (y/n) ')
    if choice.lower() in ["y", "j"]:
        for ear in ears.keys():
            while True:
                try:
                    ear_weight = float(
                        input(f"How sensitive should the AI's {ear} be (0-1)? ")
                    )
                    if 0 <= ear_weight <= 1:
                        ears[ear] = ear_weight
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 1.")
    else:
        print("The AI's ears remain unchanged.")

    # Get the situation description from the user
    situation = input(
        "\nNow describe the situation in which the role-play should take place:\n"
    )
    communication_history: list[TResponseInputItem] = [
        {"role": "developer", "content": f"## Situation\n{situation}"},
        {
            "role": "developer",
            "content": f"## Ears\n{'\n'.join(f'{key}: {value}' for key, value in ears.items())}",
        },
    ]

    # Optionally generate an illustration of the situation
    illustrate = input("\nDo you want to generate an image of this situation? (y/n) ")
    if illustrate.lower() in ["y", "j"]:
        print("Please wait, generating an image of the situation...")
        client = OpenAI()

        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Illustrate a situation in a drawn style. Note: Words like AI or User have NO MEANING for you. Situation: {situation}",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        webbrowser.open(response.data[0].url)
        print("Note: The image has been opened in your browser.\n")

    print("\nEverything is ready. Let's start the role-play!\n\n")
    print(situation)

    # Start the communication loop
    with trace("Role-play"):
        while True:
            # Get the user's message
            msg = input("\nYou say: ")
            if msg == "exit":  # Exit the loop if the user types "exit"
                print("Exiting...")
                break

            # Add the user's message to the communication history
            communication_history.append({"content": f"{msg}", "role": "user"})

            # Analyze the user's message using the message analyzer agent
            analyzer_result: AnalyzedMessage = await Runner.run(
                message_analyzer, communication_history
            )

            print(f"\nFour-Sides Analysis:\n{analyzer_result.final_output}")

            # Prepare input for the communication partner agent
            comm_agent_input = communication_history.copy()
            comm_agent_input.append(
                {
                    "content": f"## Four-Sides Analysis\n{analyzer_result.final_output}",
                    "role": "developer",
                }
            )

            while True:
                # Generate a response from the communication partner agent
                communication_partner_result = await Runner.run(
                    communication_partner, comm_agent_input
                )
                ai_answer = {
                    "content": communication_partner_result.final_output,
                    "role": "assistant",
                }
                comm_agent_input.append(ai_answer)

                # Evaluate the response using the evaluator agent
                evaluator_result = await Runner.run(evaluator, comm_agent_input)
                result: EvaluationFeedback = evaluator_result.final_output

                # If the response needs improvement, re-run with feedback
                if result.score == "needs_improvement":
                    comm_agent_input.append(
                        {"content": f"Feedback: {result.feedback}", "role": "developer"}
                    )
                else:
                    # If the response is acceptable, add it to the communication history
                    communication_history.append(ai_answer)
                    print("\nThe AI says: ", communication_partner_result.final_output)
                    break


if __name__ == "__main__":
    asyncio.run(main())
