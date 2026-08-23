# CE2 population operators

This directory contains the canonical machine-readable CE2 population operators used by both Experiments II and III.

- `Lhat_D3Q19.csv`: D3Q19 normalized CE2 observability operator.
- `Lhat_D3Q27.csv`: D3Q27 normalized CE2 observability operator.

The independently supplied operator files for Experiments II and III were verified to be byte-for-byte identical (SHA-256 shown below), so a single shared copy is retained to avoid duplication and ambiguity.

- D3Q19: `258b5dcd2ec4b4f8e735e4f180fe7fd91eb927bcb7471db12c1812dcd05c8a05`
- D3Q27: `b749b00ff3315789e1b1443d9d0568eed8b75a5ec49dddda61c9f721bcc1b5ca`

Both CSV files include the lattice velocity components, quadrature weights, and the 15 curvature-coordinate columns in the ordering expected by the numerical implementations.
