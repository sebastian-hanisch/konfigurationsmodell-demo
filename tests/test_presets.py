"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keine Verdoppelung; Auswahlfelder nur mit erlaubten Werten."""

import kfm_constants as C
import kfm_evaluation as ev
import kfm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_the_options():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.GAMMA_MIN <= p["gamma"] <= C.GAMMA_MAX and C.KMIN_MIN <= p["kmin"] <= C.KMIN_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["cutoff"] in C.CUTOFFS and p["sampler"] in C.SAMPLERS, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (γ = 2,5, strukturelle Obergrenze)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_select_specs_have_options():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, spec in P.SETTING_SPECS.items():
        assert (spec.options is not None) == (spec.lo is None), key
        if spec.options is not None:
            assert spec.default in spec.options


def test_presets_are_pairwise_different_and_differ_from_the_default_in_more_than_one_setting():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)
    default = next(iter(C.PRESETS.values()))
    for name, p in list(C.PRESETS.items())[1:]:
        assert sum(p[k] != default[k] for k in default) >= 2, name


def test_dictionaries_of_cutoffs_and_samplers_match_the_defaults():
    assert C.DEFAULT_CUTOFF in C.CUTOFFS and C.DEFAULT_SAMPLER in C.SAMPLERS and set(C.SAMPLERS) == {"stub", "swap", "chunglu"}
