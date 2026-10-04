"""Route a batch of messages with the packed model and keep only confident answers.

    python examples/batch_routing.py

``policy`` is a distribution over the options, so its maximum is a natural confidence score;
messages below the threshold go to a human queue instead.
"""

from opendecision import Choice, Decision, OpenDecision

model = OpenDecision.from_pretrained("Tokz-labs/OpenDecision-Large-Packed")

messages = [
    "Where is my order? It was supposed to arrive on Monday.",
    "Please cancel my subscription before the next billing date.",
    "The export button does nothing when I click it.",
    "Do you offer a discount for non-profits?",
    "hi",
]
intent = Decision(
    "intent",
    Choice(["order_status", "cancel_subscription", "bug_report", "pricing_question", "other"]),
    description="What does the customer want?",
)

results = model.decide(messages, {"intent": intent}, batch_size=16)
for text, result in zip(messages, results):
    answer = result["intent"]
    confidence = answer.policy[answer.best]
    route = answer.best if confidence >= 0.6 else "human_review"
    print(f"{confidence:.2f}  {route:20s}  {text}")
