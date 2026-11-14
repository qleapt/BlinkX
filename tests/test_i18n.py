"""English / Turkish: the translator's rules (spx/i18n.py)."""
from spx import i18n


def test_english_is_untouched_and_html_is_skipped():
    assert i18n.tr("Detect stars", "en") == "Detect stars"
    assert i18n.tr("<div>Detect stars</div>", "tr") == "<div>Detect stars</div>"


def test_exact_template_reorder_and_known_words_inside():
    assert i18n.tr("Detect stars") == "Yıldızları bul"
    assert i18n.tr("  Go  ") == "  Git  "
    assert i18n.tr("No change above 5σ between Epoch 1 and Epoch 3.") == \
        "Dönem 1 ile Dönem 3 arasında 5σ üstünde değişim yok."   # values are translated too ("Epoch {}")
    assert i18n.tr("**It moved**: 10.3″ toward the north, more than 3× the uncertainty (5.1″).") == \
        "**Hareket etti**: kuzey yönünde 10.3″, belirsizliğin 3 katından fazla (5.1″)."


def test_text_built_from_parts_is_translated_piece_by_piece():
    s = ("the Gaia star list is still loading; the older WISE (2010) / 2MASS (~2000) photos are still downloading. "
         "They are added here automatically as soon as they arrive.")
    out = i18n.tr(s)
    assert "Gaia yıldız listesi hâlâ yükleniyor" in out and "Gelir gelmez" in out and "still" not in out
    lines = i18n.tr("How to look at the photos:\n\n- **Blink**: flip between dates: movers jump")
    assert lines == "Fotoğraflara nasıl bakılır:\n\n- **Blink**: tarihler arasında geçiş: hareket edenler sıçrar"


def test_values_inside_a_template_are_translated_by_templates_too():
    out = i18n.tr("It moved 12.50″ per hour within 3.0 hours. No star moves measurably in days (the fastest, "
                  "Barnard's Star, drifts 0.028″ per day).")
    assert "hours" not in out and "per hour" not in out and "saat" in out


def test_motion_verdict_heading_is_translated_whole():
    from spx import motion
    for name in motion.CLASS_NAME.values():
        assert i18n.tr(f"What it probably is: {name} (91%)") == f"Muhtemelen ne: {i18n.TR[name]} (%91)"


def test_every_view_name_and_step_has_turkish():
    for s in ("Blink", "Subtract", "Every exposure", "Spectrum", "Motion check", "Find target",
              "Search NASA archive", "Download photos", "Line up & average", "Sky viewer", "Photo list"):
        assert s in i18n.TR
