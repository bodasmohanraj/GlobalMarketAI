
import numpy as np

model = np.array([0.5088, 0.5041, 0.5031, 0.5093])
baseline = np.array([0.5326, 0.5409, 0.5145, 0.4798])

difference = model - baseline

print("\n--- FOLD CONSISTENCY ANALYSIS ---")

for i, (m, b, d) in enumerate(
    zip(model, baseline, difference), start=1
):
    print(
        f"Fold {i}: model={m:.4%}, "
        f"baseline={b:.4%}, "
        f"difference={d:+.4%}"
    )

print("\nMean model accuracy:", f"{model.mean():.4%}")
print("Mean baseline accuracy:", f"{baseline.mean():.4%}")
print("Mean difference:", f"{difference.mean():+.4%}")
print("Folds beating baseline:", int((difference > 0).sum()), "of", len(model))
print("Best fold difference:", f"{difference.max():+.4%}")
print("Worst fold difference:", f"{difference.min():+.4%}")
