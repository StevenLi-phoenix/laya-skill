import json
import logging
import time

from laya import Router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
router = Router()
state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}
t=time.time(); r = router.predict(state, questions); print(f"first call (incl. load): {time.time()-t:.2f}s")
t=time.time(); r = router.predict(state, questions); print(f"warm call: {(time.time()-t)*1000:.0f}ms")
print(json.dumps(r["answers"], indent=1, default=str)[:1500]); print("routing:", r["routing"])
for text in ["मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
             "La aplicación se cierra cada vez que abro la configuración.",
             "我打开设置页面应用就闪退。"]:
    t=time.time(); x = router.predict(text, {"department": questions["department"]})
    print(f"{x['routing']['model']:>13} {x['answers']['department']['choice']:>10}  {(time.time()-t)*1000:.0f}ms  {text}")
