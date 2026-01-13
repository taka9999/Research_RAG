from pathlib import Path

base = Path("data/papers")

dirs = [
    base /"econometrics",
    base /"regime_switching",
    base /"stochastic_control",
    base /"optimization",
    base /"reinforcement_learning",
    base /"fixed_income",
    base /"misc",

]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

print("research_rag directory structure created.")