"""Several decisions about one support ticket in a single call.

python examples/quickstart.py
"""

from opendecision import Bool, Choice, Decision, MultiLabel, OpenDecision

model = OpenDecision.from_pretrained("Tokz-labs/OpenDecision-Large")

ticket = "I was charged twice for my May invoice and the app keeps logging me out."
out = model.decide(
    ticket,
    {
        "queue": Decision(
            "queue",
            Choice(["billing", "technical", "account", "sales"]),
            description="Which team should handle this ticket?",
        ),
        "tags": MultiLabel(["double_charge", "login_issue", "refund_request", "outage"]),
        "urgent": Bool(),
    },
)

print("queue :", out["queue"].best, out["queue"].ranked()[:2])
print("tags  :", out["tags"].selected)
print("urgent:", out["urgent"].policy)
