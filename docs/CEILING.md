# Ceiling: what this benchmark can and cannot weigh

n=120 on Qwen2.5-0.5B-Instruct, one subject model, caps this benchmark's
scientific weight regardless of execution quality. No amount of gating,
bootstrapping or documentation raises it: at 60 rows per subset with base
rates of 38% and 80% incorrect, the analytic half-width is about 0.145, so
only large effects separate from chance — and a single 0.5B model's error
pattern says nothing about uncertainty in general. No project should be
scored above its scope; say so first. No hedging follows: until run #3 and a
second subject model exist, the honest score of this repository as science is
bounded, however high its engineering marks climb.

What lifts the ceiling is run #3 at n=600 on a GPU (pre-registered in
`docs/PREREGISTRATION.md`, pre-flighted in `configs/run3_gpu.yaml`), plus a
second subject model from a different family on the same frozen question set,
so at least one finding generalises beyond one model.
