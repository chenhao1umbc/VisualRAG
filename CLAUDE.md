## A6000 GPU Policy (budget-critical)

- Only ONE A6000 VM at a time.
- Create VM only when actively running a job. Destroy immediately when the job finishes.
- Never leave a VM idle. Before ending any session involving GPU work, confirm `hs.py down` was run.
- Use `scripts/hs.py` for all VM lifecycle operations.
