# -----------------------------------------------------------------------------
# Time series
# -----------------------------------------------------------------------------

# Indices of eng.dat variables
eng_vars_dict = {
    "kinetic energy": 1,
    "potential energy": 2,
    "total energy": 3,
    "geostrophic energy": 4,
    "wave energy": 5,
    "Rossby number": 6,
    "vertical Froude number": 7,
    "horizontal Froude number": 8,
    "enstrophy": 9
}

# Labels for eng.dat variables
eng_labels_arr = (
    r"$KE$", r"$PE$", r"$E$", r"$E_{\mathrm{geo}}$",
    r"$E_{\mathrm{wave}}$", r"$\mathrm{Ro}$",
    r"$\mathrm{Fr}_z$", r"$\mathrm{Fr}_h$", r"$Z$"
)

# Indices of eps.dat variables
eps_vars_dict = {
    "KE dissipation": 1,
    "PE dissipation": 2,
    "E dissipation": 3,
    "horizontal KE dissipation": 4,
    "vertical KE dissipation": 5,
    "horizontal PE dissipation": 6,
    "vertical PE dissipation": 7
}

# Labels for eps.dat variables
eps_labels_arr = (
    r"$\varepsilon_{\mathrm{KE}}$", r"$\varepsilon_{\mathrm{PE}}$",
    r"$\varepsilon$", r"$\varepsilon_{\mathrm{KE},h}$",
    r"$\varepsilon_{\mathrm{KE},z}$", r"$\varepsilon_{\mathrm{PE},h}$",
    r"$\varepsilon_{\mathrm{PE},z}$",
)

# -----------------------------------------------------------------------------
# Spectra
# -----------------------------------------------------------------------------

# Indices of spc.dat variables
spc_vars_dict = {
    "kinetic energy": 1,
    "potential energy": 2,
    "total energy": 3,
    "geostrophic energy": 4,
    "wave energy": 5,
    "buoyancy flux": 6,
    "KE dissipation": 7,
    "PE dissipation": 8,
    "KE in u": 9,
    "KE in v": 10,
    "KE in w": 11,
    "rotational KE": 12,
    "divergent KE": 13
}

# Labels for spc.dat variables
spc_labels_arr = (
    r"$KE$", r"$PE$", r"$E$", r"$E_{\mathrm{geo}}$",
    r"$E_{\mathrm{wave}}$", r"$B$", r"$\varepsilon_{\mathrm{KE}}$",
    r"$\varepsilon_{\mathrm{PE}}$", r"$KE_x$", r"$KE_y$", r"$KE_z$",
    r"$KE_{\mathrm{rot}}$", r"$KE_{\mathrm{div}}$"
)

# Indices of trn.dat variables
trn_vars_dict = {
    "geostrophic energy transfer": 1,
    "wave energy transfer": 2,
    "geo + wave energy transfer": 3,
    "kinetic energy transfer": 4,
    "potential energy transfer": 5,
    "kinetic + potential energy transfer": 6
}

# Labels for spc.dat variables
trn_labels_arr = (
    r"$T_{\mathrm{geo}}$", r"$T_{\mathrm{wave}}$",
    r"$T_{\mathrm{geo}} + T_{\mathrm{wave}}$",
    r"$T_{\mathrm{KE}}$", r"$T_{\mathrm{PE}}$",
    r"$T_{\mathrm{KE}} + T_{\mathrm{PE}}$"
)

# -----------------------------------------------------------------------------