# Tensor Elements Ideas

This file collects ideas that are intentionally out of scope for the current implementation round.

## Potential Future Views

- `log_magnitude`
- `sparsity`
- `nan_inf`
- `topk`
- `slice`
- `reduce`
- `profiles`
- `spectrum`

## Potential Future Controls

- shared color scale across tensors
- robust scaling by percentiles
- symmetric color scaling around zero
- reference tensor comparison
- outlier highlighting

## Notes

- Prefer additions that fit the current Matplotlib control style.
- Avoid turning `TensorElementsConfig` into a catch-all config object.
- Keep `show_tensor_elements(...)` aligned with the current package API style.
