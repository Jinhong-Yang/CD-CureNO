# AS4/8552 thermochemical parameter provenance

All solver calculations use kelvin, seconds, metres, kilograms, and joules.
No value in this table was fitted to the public Case1 temperature labels.

| Quantity | Solver value | Unit | Source and location | Status |
|---|---:|---|---|---|
| \(A\) | \(1.528\times10^5\) | s\(^{-1}\) | Niaki et al. (2021), Table 1, citing Hubert et al. (2001) | transcribed |
| \(\Delta E\) | \(6.650\times10^4\) | J mol\(^{-1}\) | Niaki et al. (2021), Table 1 | transcribed |
| \(M\) | 0.8129 | 1 | Niaki et al. (2021), Table 1 | transcribed |
| \(N\) | 2.7360 | 1 | Niaki et al. (2021), Table 1 | transcribed |
| \(C\) | 43.09 | 1 | Niaki et al. (2021), Table 1 | transcribed |
| \(C_0\) | -1.6840 | 1 | Niaki et al. (2021), Table 1 | transcribed |
| \(C_T\) | \(5.475\times10^{-3}\) | K\(^{-1}\) | Niaki et al. (2021), Table 1 | transcribed |
| \(R\) | 8.314 | J mol\(^{-1}\) K\(^{-1}\) | Niaki et al. (2021), Table 1 | transcribed |
| \(H_r\) | \(5.40\times10^5\) | J kg-resin\(^{-1}\) | Johnston (1997), Table C.11 | transcribed |
| Initial \(\alpha\) | 0.05 | 1 | Johnston (1997), Sec. 6.1.7; public Case1 starts at 0.05000147 | prescribed |
| AS4 \(v_f\) | 0.574 | 1 | Niaki et al. (2021), Table 2 | transcribed |
| 8552 \(v_r\) | 0.426 | 1 | Niaki et al. (2021), Table 2 | transcribed |
| AS4 \(\rho_f\) | 1790 | kg m\(^{-3}\) | Niaki et al. (2021), Table 2; Johnston (1997), Table C.11 | transcribed |
| 8552 \(\rho_r\) | 1300 | kg m\(^{-3}\) | Niaki et al. (2021), Table 2; Johnston (1997), Table C.11 | transcribed |
| Invar \(\rho_t\) | 8150 | kg m\(^{-3}\) | Niaki et al. (2021), Table 2 | transcribed |
| AS4 \(C_{p,f}\) | 914.0 | J kg\(^{-1}\) K\(^{-1}\) | Niaki et al. (2021), Table 2 | transcribed |
| 8552 \(C_{p,r}\) | 1304.2 | J kg\(^{-1}\) K\(^{-1}\) | Niaki et al. (2021), Table 2 | transcribed |
| Invar \(C_{p,t}\) | 510.0 | J kg\(^{-1}\) K\(^{-1}\) | Niaki et al. (2021), Table 2 | transcribed |
| AS4 \(k_f\) | 3.960 | W m\(^{-1}\) K\(^{-1}\) | Niaki et al. (2021), Table 2 | transcribed |
| 8552 \(k_r\) | 0.212 | W m\(^{-1}\) K\(^{-1}\) | Niaki et al. (2021), Table 2 | transcribed |
| AS4/8552 longitudinal \(k_{c11}\) | \(4.49+9.12\times10^{-3}T_{^\circ C}\) | W m\(^{-1}\) K\(^{-1}\) | Johnston (1997), Fig. 6.5; measured cured unidirectional laminate | transcribed |
| P4 \(k_x\) reference | 4.6724 | W m\(^{-1}\) K\(^{-1}\) | Johnston relation at 20 °C; fibres aligned with benchmark \(x\) | frozen numerical reference |
| P4 \(k_z\) reference | 0.6369803 | W m\(^{-1}\) K\(^{-1}\) | Springer--Tsai relation using the public Case1 constants | frozen public-anchor reference |
| Invar \(k_t\) | 13.0 | W m\(^{-1}\) K\(^{-1}\) | Niaki et al. (2021), Table 2 | transcribed |
| Lower/tool \(h_t\) | 70 | W m\(^{-2}\) K\(^{-1}\) | Chen et al. preprint (2021), implementation section | transcribed |
| Upper/composite \(h_c\) | 120 | W m\(^{-2}\) K\(^{-1}\) | Chen et al. preprint (2021), implementation section | transcribed |

## P4 anisotropy contract

The P4 rectangular benchmark is a two-dimensional \(x\)-\(z\) section of a
unidirectional AS4/8552 laminate with fibres aligned to \(x\). Johnston's
measured longitudinal relation therefore supplies \(k_x\); the public
Case1/Springer--Tsai transverse relation supplies \(k_z\). At the frozen
293.15 K reference, \(k_x/k_z=7.335\). This preserves the exact 1-D extrusion
anchor because laterally homogeneous solutions have zero \(x\)-gradient.

The first benchmark keeps both conductivities constant. Johnston measured a
clear temperature dependence and reports the longitudinal relation for cured
material, so temperature- and cure-dependent anisotropic conductivity remains
an explicit model-form limitation rather than an unreported assumption.
The source PDF retrieved from the Library and Archives Canada URL in
`material_parameter_sources.bib` had SHA-256
`4e6aa717dcde5eb1a60db5a2850485522daf1c1ec1a562515e5f18b5cfc111dd`
when the Figure 6.5 equation was visually verified.

## Kinetic-law provenance conflict

Johnston (1997), Eq. 6.13, prints the diffusion-control denominator as
\(1+\exp[C(\alpha-(C_0+C_TT))]\). Niaki et al. (2021), Eq. 31, prints
only the exponential term while citing the later Hubert et al. model. The immutable
public Case1 degree-of-cure arrays are much closer to the Johnston form when their
temperature histories are supplied directly: the Johnston form has approximately
1% global relative error, whereas the no-offset transcription has approximately 47%.
The solver therefore uses the source-supported Johnston form and records the printed
Niaki variant as a sensitivity/provenance issue rather than silently selecting it.

## Public geometry conflict

The public arrays and pinned upstream code define tool indices 0--20 and composite
indices 21--50 on a 0--50 mm coordinate. Niaki et al. (2021) also specify a 20 mm
tool and 30 mm composite. The final Chen et al. article text reverses those labels in
one passage. Public-data validation follows the immutable array mask and reports this
conflict explicitly.
