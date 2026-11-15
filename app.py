"""SPHEREx Blink Viewer: type a sky position or object name, watch SPHEREx images of it blink."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import json
import time
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from astropy.time import Time

from spx import (archive, catalogs, config, cutouts, detect, eta, i18n, motion, photometry, prefetch, render, stack,
                 targets, teleport, viewer, warp)

st.set_page_config(page_title="SPHEREx Blink Viewer", layout="wide")
# English / Turkish: everything Streamlit shows goes through i18n.tr() when Turkish is chosen
i18n.install(lambda: st.session_state.get("lang", "en"))

PRESET_BAND = targets.PRESET_BAND
MODES = {"blink": "Blink epochs", "sub": "Subtract", "movie": "Every exposure",
         "spectrum": "Spectrum & light curve", "motion": "Motion check"}
MODE_LABEL = {"blink": "Blink", "sub": "Subtract", "movie": "Every exposure",
              "spectrum": "Spectrum", "motion": "Motion check"}
MODE_HELP = {"blink": "flip between dates: movers jump",
             "sub": "newer − older: see only what changed", "movie": "one frame per single photo",
             "spectrum": "brightness at each infrared colour", "motion": "does it move as predicted?"}
HOW_TO = {
    "blink": (
        "1. **Choose a target** in the sidebar: a challenge target, any name (e.g. *M 42*), or a sky position in "
        "degrees (e.g. `269.452 4.693`). Press **Go**.\n"
        "2. **Press Play** under the image. Each frame is one *epoch*: one pass of the survey, about six months "
        "apart, with all its photos averaged. A star that *jumps* is moving; one that *flickers* changes "
        "brightness.\n"
        "3. **Compare two dates** in the viewer's toolbar: *Blink A ↔ B* flips between the two dates you pick, "
        "*Side by side* shows them next to each other with a linked crosshair, *Swipe* slides one over the other, "
        "*Difference* subtracts them.\n"
        "4. **Mark a find**: choose *Mark*, click an object, go to another date (Next or the timeline) and click the "
        "same object again. Three or more dates are best. *Check my find* under the viewer then fits one path "
        "through all your marks, says what kind of object it probably is, and predicts where it goes.\n"
        "5. **Dark, Bright and the scale** (asinh, log, sqrt, linear) only change how the picture looks, never the "
        "measurements: the ? next to the scale in the viewer explains each one.\n\n"
        "Tip: switch on *Include older photos* in the sidebar to add 2MASS (~2000) and WISE (2010): 15–25 years "
        "make even slow stars jump."),
    "sub": (
        "1. **Pick two photos**: *Older photo (A)* and *Newer photo (B)*. The picture is B − A, pixel by pixel: "
        "everything that stayed the same cancels out.\n"
        "2. **Read the colours**: **red** = more light now (in B), **blue** = more light before (in A), "
        "**white / grey** = no change. The number under the mouse is the change in surface brightness, in MJy/sr "
        "(megajansky per steradian: how much infrared energy arrives from that patch of sky); + means brighter "
        "in B, − brighter in A, and the bigger the number, the bigger the change.\n"
        "3. **Recognise what changed**: a **moving star** leaves a blue + red pair (blue where it was, red where it "
        "is now); a star that **brightened** leaves a single red spot, one that **faded** a single blue spot. Thin "
        "rings around very bright stars are leftovers of imperfect cancelling, not real changes.\n"
        "4. **Hide the noise**: *Hide changes smaller than (× noise)* turns every pixel whose change is below that "
        "many noise levels (σ) white. 3 keeps only changes that are very likely real.\n"
        "5. **Measure it**: *Mark* the blue end and the red end of a pair (or the object on the two photos) and use "
        "*Check my find*.\n\n"
        "Note: with an older survey photo (2MASS, WISE) only *positions* can be compared: another telescope's "
        "brightness is scaled for display, so its red / blue strength means nothing."),
    "movie": (
        "1. **Every single exposure**, one per frame, not averaged: noisier, but nothing is smoothed away. This is "
        "the view for things that change within hours: **asteroids and comets** (they move during one epoch), "
        "flares, and cosmic-ray hits (one frame only).\n"
        "2. **Order by time** to follow fast movers: an asteroid steps along a line from frame to frame. **Order by "
        "wavelength** to step through the infrared colours inside the band: each SPHEREx exposure sees a "
        "slightly different colour, so a star whose brightness changes with colour fades in or out as you flip.\n"
        "3. **Hide photos with warnings at the target** leaves out exposures where the pipeline marked the target's "
        "pixels as unreliable (overexposed, cosmic rays…).\n"
        "4. **Measure a fast mover**: *Mark* it on several exposures hours apart, then *Check my find*: the speed "
        "comes out in ″ per hour and the path is predicted for the next hours.\n\n"
        "At most 60 exposures are shown, spread evenly over the dates."),
    "spectrum": (
        "1. **Each point is one exposure** measured at the target: across is the wavelength (the infrared colour, "
        "in µm), up is the brightness (flux, in mJy). SPHEREx's 102 colours together make a spectrum from 0.75 to "
        "5 µm.\n"
        "2. **Colours are dates** (epochs). Compare them: if one date sits higher everywhere, the object "
        "brightened; if only part of the spectrum changed, its colour changed. *Dates to show* hides or shows a date.\n"
        "3. **Hollow points** have flagged pixels in the measuring circle (often an overexposed centre): treat them "
        "with care.\n"
        "4. **Shapes to look for**: ordinary stars get fainter towards long wavelengths; cold brown dwarfs rise "
        "towards 4–5 µm and dip at 3.3 µm (methane); dips near 3.0 µm (water ice) and 4.27 µm (CO₂ ice) come "
        "from icy dust in front of the star; a bump at 3.3 µm is glowing dust in star-forming regions.\n"
        "5. **The light curve** below shows the brightness in the band chosen in the sidebar against time, and "
        "*What changed?* sums up each band in words.\n"
        "6. *Logarithmic flux axis* spreads out faint and bright values; *Load more exposures* fills the spectrum "
        "in.\n\n"
        "From a find: *Spectrum of the first mark* in *Check my find* measures your own marked object here."),
    "motion": (
        "1. **A named target** (e.g. Barnard's Star): the app measures where its light is centred on every epoch "
        "photo and compares it with where the catalogue says it should be (grey line). Below, the same positions "
        "are fitted on their own: speed, parallax, what kind of object, and the path for the coming years.\n"
        "2. **Your own find**: in any view, *Mark* the same object on two or more photos (three or more is much "
        "better), choose the marks in *Check my find* and press *Analyse*. All marks are fitted together, every "
        "pair is compared, and every mark is tested against the others, so a wrong mark stands out.\n"
        "3. **Read the verdict**: the bars show how strongly the evidence points to *standing still*, a *moving "
        "star*, an *asteroid / comet* or *not one object*; each line below says why.\n"
        "4. **The future**: the dashed line continues the path, with a date at each point and a band that shows "
        "the uncertainty growing with time. Change *Predict ahead* to look further.\n"
        "5. **A typed sky position** has no catalogue path: mark objects yourself, then *Check my find*.\n\n"
        "Optional: *Also measure it on the other photos* adds positions where your marks predict the object "
        "(off by default)."),
}
BAND_HELP = (
    "SPHEREx sees infrared light, invisible to our eyes, in six bands. Shorter wavelength = more energetic light: "
    "one photon carries E = 1.24 / λ electronvolts (λ in µm); visible light is 1.8–3.1 eV.\n\n"
    "- **D1** 0.75–1.09 µm (1.65–1.14 eV): just beyond red light; ordinary and hot stars are brightest here.\n"
    "- **D2** 1.10–1.62 µm (1.13–0.77 eV): cool red stars peak here; brown dwarfs show water dips (1.4 µm).\n"
    "- **D3** 1.63–2.41 µm (0.76–0.51 eV): sees through dust; red giants, young stars hidden in dusty clouds.\n"
    "- **D4** 2.42–3.82 µm (0.51–0.32 eV): water ice (3.0 µm), glowing dust (3.3 µm); methane makes cold brown "
    "dwarfs faint here.\n"
    "- **D5** 3.83–4.41 µm (0.32–0.28 eV): CO₂ ice (4.27 µm); cold brown dwarfs are bright from 4 to 5 µm.\n"
    "- **D6** 4.42–5.00 µm (0.28–0.25 eV): CO ice (4.67 µm); the coldest brown dwarfs (WISE 0855) are brightest "
    "here; warm dust.\n\n"
    "Brightness numbers say how much of that light arrives: MJy/sr for a patch of sky, mJy for one star. "
    "µm = micrometre.")
HOW_TO_TR = {
    "blink": (
        "1. Kenar çubuğundan **bir hedef seçin**: örnek bir hedef, herhangi bir ad (örn. *M 42*) ya da derece cinsinden "
        "bir gökyüzü konumu (örn. `269.452 4.693`). **Git**'e basın.\n"
        "2. Resmin altındaki **Oynat**'a basın. Her kare bir *dönemdir*: taramanın yaklaşık altı ay arayla yaptığı bir "
        "geçiş, tüm fotoğrafları ortalanmış olarak. *Sıçrayan* bir yıldız hareket ediyordur; *kırpışan* bir yıldızın "
        "parlaklığı değişiyordur.\n"
        "3. Görüntüleyicinin araç çubuğunda **iki tarihi karşılaştırın**: *Blink A ↔ B* seçtiğiniz iki tarih arasında "
        "geçiş yapar, *Yan yana* bağlı bir artı imleciyle ikisini yan yana gösterir, *Kaydır* birini ötekinin üzerinde "
        "kaydırır, *Fark* ikisini çıkarır.\n"
        "4. **Bir buluşu işaretleyin**: *İşaretle*'yi seçin, bir cisme tıklayın, başka bir tarihe geçin (Sonraki veya "
        "zaman çizelgesi) ve aynı cisme yeniden tıklayın. Üç veya daha fazla tarih en iyisidir. Görüntüleyicinin "
        "altındaki *Bulduğumu denetle* tüm işaretlerinizden tek bir yol geçirir, muhtemelen ne tür bir cisim olduğunu "
        "söyler ve nereye gideceğini tahmin eder.\n"
        "5. **Koyu, Parlak ve ölçek** (asinh, log, sqrt, linear) yalnızca resmin görünüşünü değiştirir, ölçümleri asla: "
        "görüntüleyicide ölçeğin yanındaki ? her birini açıklar.\n\n"
        "İpucu: 2MASS (~2000) ve WISE (2010) fotoğraflarını eklemek için kenar çubuğunda *Eski fotoğrafları ekle*'yi "
        "açın: 15–25 yıl, yavaş yıldızları bile sıçratır."),
    "sub": (
        "1. **İki fotoğraf seçin**: *Eski fotoğraf (A)* ve *Yeni fotoğraf (B)*. Resim piksel piksel B − A'dır: aynı "
        "kalan her şey birbirini götürür.\n"
        "2. **Renkleri okuyun**: **kırmızı** = şimdi daha çok ışık (B'de), **mavi** = önceden daha çok ışık (A'da), "
        "**beyaz / gri** = değişim yok. Farenin altındaki sayı, yüzey parlaklığındaki değişimdir; birimi MJy/sr "
        "(steradyan başına megajansky: gökyüzünün o parçasından ne kadar kızılötesi enerji geldiği). + B'de daha "
        "parlak, − A'da daha parlak demektir; sayı ne kadar büyükse değişim o kadar büyüktür.\n"
        "3. **Neyin değiştiğini tanıyın**: **hareket eden bir yıldız** mavi + kırmızı bir çift bırakır (olduğu yerde "
        "mavi, şimdi olduğu yerde kırmızı); **parlayan** bir yıldız tek bir kırmızı leke, **sönen** bir yıldız tek bir "
        "mavi leke bırakır. Çok parlak yıldızların çevresindeki ince halkalar kusurlu çıkarmanın artıklarıdır, gerçek "
        "değişim değildir.\n"
        "4. **Gürültüyü gizleyin**: *Şundan küçük değişimleri gizle (× gürültü)*, değişimi o kadar gürültü düzeyinin "
        "(σ) altında kalan her pikseli beyaz yapar. 3, yalnızca büyük olasılıkla gerçek olan değişimleri bırakır.\n"
        "5. **Ölçün**: bir çiftin mavi ucunu ve kırmızı ucunu (ya da cismi iki fotoğrafta) *İşaretle* ve *Bulduğumu "
        "denetle*'yi kullanın.\n\n"
        "Not: eski bir tarama fotoğrafıyla (2MASS, WISE) yalnızca *konumlar* karşılaştırılabilir: başka bir teleskobun "
        "parlaklığı görüntüleme için ölçeklenmiştir, bu yüzden kırmızı / mavi gücü bir şey ifade etmez."),
    "movie": (
        "1. **Her tek poz**, kare başına bir tane, ortalanmadan: daha gürültülü ama hiçbir şey yumuşatılıp yok "
        "edilmez. Saatler içinde değişen şeylerin görünümü budur: **asteroitler ve kuyruklu yıldızlar** (bir dönem "
        "içinde hareket ederler), parlamalar ve kozmik ışın çarpmaları (yalnızca bir karede).\n"
        "2. Hızlı hareket edenleri izlemek için **zamana göre sıralayın**: bir asteroit kareden kareye bir çizgi boyunca "
        "ilerler. Bant içindeki kızılötesi renkler arasında adım adım ilerlemek için **dalga boyuna göre sıralayın**: "
        "her SPHEREx pozu biraz farklı bir renk görür, bu yüzden parlaklığı renge göre değişen bir yıldız siz kareleri "
        "çevirdikçe belirir ya da söner.\n"
        "3. **Hedefte uyarı olan fotoğrafları gizle**, işlem hattının hedef piksellerini güvenilmez olarak işaretlediği "
        "pozları (aşırı pozlama, kozmik ışın…) dışarıda bırakır.\n"
        "4. **Hızlı bir cismi ölçün**: onu saatler arayla birkaç pozda *İşaretle*, sonra *Bulduğumu denetle*: hız "
        "saatte ″ olarak çıkar ve yol sonraki saatler için tahmin edilir.\n\n"
        "En fazla 60 poz gösterilir, tarihlere eşit dağıtılarak."),
    "spectrum": (
        "1. **Her nokta hedefte ölçülen bir pozdur**: yatay eksen dalga boyu (kızılötesi renk, µm), dikey eksen "
        "parlaklıktır (akı, mJy). SPHEREx'in 102 rengi birlikte 0,75'ten 5 µm'ye bir tayf oluşturur.\n"
        "2. **Renkler tarihlerdir** (dönemler). Karşılaştırın: bir tarih her yerde daha yüksekse cisim parlamıştır; "
        "tayfın yalnızca bir kısmı değiştiyse rengi değişmiştir. *Gösterilecek tarihler* bir tarihi gizler veya "
        "gösterir.\n"
        "3. **İçi boş noktalar** ölçüm dairesinde işaretli piksel içerir (çoğu zaman aşırı pozlanmış bir merkez): "
        "bunlara dikkatle yaklaşın.\n"
        "4. **Aranacak biçimler**: sıradan yıldızlar uzun dalga boylarına doğru sönükleşir; soğuk kahverengi cüceler "
        "4–5 µm'ye doğru yükselir ve 3,3 µm'de (metan) çöker; 3,0 µm (su buzu) ve 4,27 µm (CO₂ buzu) yakınındaki "
        "çukurlar yıldızın önündeki buzlu tozdan gelir; 3,3 µm'deki bir tümsek yıldız oluşum bölgelerindeki parlayan "
        "tozdur.\n"
        "5. Aşağıdaki **ışık eğrisi** kenar çubuğunda seçilen banttaki parlaklığı zamana karşı gösterir; *Ne değişti?* "
        "her bandı sözcüklerle özetler.\n"
        "6. *Logaritmik akı ekseni* sönük ve parlak değerleri açar; *Daha fazla poz yükle* tayfı doldurur.\n\n"
        "Bir buluştan: *Bulduğumu denetle*'deki *İlk işaretin tayfı* kendi işaretlediğiniz cismi burada ölçer."),
    "motion": (
        "1. **Adı olan bir hedef** (örn. Barnard Yıldızı): uygulama, ışığının her dönem fotoğrafında nerede "
        "toplandığını ölçer ve kataloğun olması gerektiğini söylediği yerle (gri çizgi) karşılaştırır. Aşağıda aynı "
        "konumlar kendi başlarına uydurulur: hız, paralaks, ne tür bir cisim ve önümüzdeki yıllar için yol.\n"
        "2. **Kendi buluşunuz**: herhangi bir görünümde aynı cismi iki veya daha fazla fotoğrafta *İşaretle* (üç veya "
        "daha fazlası çok daha iyidir), *Bulduğumu denetle*'de işaretleri seçin ve *Çözümle*'ye basın. Tüm işaretler "
        "birlikte uydurulur, her çift karşılaştırılır ve her işaret diğerlerine karşı sınanır; böylece yanlış bir "
        "işaret göze çarpar.\n"
        "3. **Kararı okuyun**: çubuklar kanıtların *duruyor*, *hareketli yıldız*, *asteroit / kuyruklu yıldız* veya "
        "*tek cisim değil* seçeneklerini ne kadar güçlü gösterdiğini gösterir; altındaki her satır nedenini söyler.\n"
        "4. **Gelecek**: kesikli çizgi yolu sürdürür; her noktada bir tarih ve zamanla büyüyen belirsizliği gösteren bir "
        "bant vardır. Daha ileriye bakmak için *İleriye tahmin*'i değiştirin.\n"
        "5. **Yazılmış bir gökyüzü konumunun** katalog yolu yoktur: cisimleri kendiniz işaretleyin, sonra *Bulduğumu "
        "denetle*.\n\n"
        "İsteğe bağlı: *Diğer fotoğraflarda da ölç*, işaretlerinizin cismi öngördüğü yerlerdeki konumları ekler "
        "(varsayılan olarak kapalı)."),
}
BAND_HELP_TR = (
    "SPHEREx gözlerimizin göremediği kızılötesi ışığı altı bantta görür. Daha kısa dalga boyu = daha enerjik ışık: "
    "bir foton E = 1,24 / λ elektronvolt taşır (λ µm cinsinden); görünür ışık 1,8–3,1 eV'dir.\n\n"
    "- **D1** 0,75–1,09 µm (1,65–1,14 eV): kırmızının hemen ötesi; sıradan ve sıcak yıldızlar burada en parlaktır.\n"
    "- **D2** 1,10–1,62 µm (1,13–0,77 eV): soğuk kırmızı yıldızlar burada tepe yapar; kahverengi cücelerde su "
    "çukurları (1,4 µm).\n"
    "- **D3** 1,63–2,41 µm (0,76–0,51 eV): tozun içini görür; kırmızı devler, tozlu bulutlarda saklı genç "
    "yıldızlar.\n"
    "- **D4** 2,42–3,82 µm (0,51–0,32 eV): su buzu (3,0 µm), parlayan toz (3,3 µm); metan soğuk kahverengi cüceleri "
    "burada sönük yapar.\n"
    "- **D5** 3,83–4,41 µm (0,32–0,28 eV): CO₂ buzu (4,27 µm); soğuk kahverengi cüceler 4–5 µm arasında parlaktır.\n"
    "- **D6** 4,42–5,00 µm (0,28–0,25 eV): CO buzu (4,67 µm); en soğuk kahverengi cüceler (WISE 0855) burada en "
    "parlaktır; ılık toz.\n\n"
    "Parlaklık sayıları o ışıktan ne kadar geldiğini söyler: gökyüzü parçası için MJy/sr, tek yıldız için mJy. "
    "µm = mikrometre.")
i18n.TR.update({HOW_TO[k]: HOW_TO_TR[k] for k in HOW_TO})
i18n.TR.update({BAND_HELP: BAND_HELP_TR})
PRESET_HELP = {"Barnard's Star": "Fastest-moving star in the sky (10″ a year)",
               "Epsilon Indi B": "Pair of cold brown dwarfs, 12 light-years away",
               "WISE 0855-0714": "Coldest known brown dwarf (about as cold as a freezer); fast mover",
               "RW Aurigae": "Young star that dims and flares unpredictably",
               "R Draconis": "Pulsating giant star (brightens and fades over ~8 months)"}
BAND_LABEL = {b: f"{b} · {lo:.2f}–{hi:.2f} µm" for b, (lo, hi, _) in config.BANDS.items()}

qp = st.query_params
ss = st.session_state


def _init():
    if ss.get("_init"):
        return
    ss._init = True
    if "ra" in qp and "dec" in qp:
        ss.target_text = f"{qp['ra']} {qp['dec']}"
    else:
        ss.target_text = qp.get("target", "Barnard's Star")[:100]
    try:
        f = float(qp.get("fov", 5))
    except ValueError:
        f = 5
    ss.fov = min(config.FOV_CHOICES, key=lambda c: abs(c - f))
    p = targets.preset(ss.target_text)
    b = str(qp.get("band", "")).upper()
    ss.band = b if b in config.BANDS else PRESET_BAND.get(p.name if p else "", "D3")
    m = str(qp.get("mode", "blink")).lower()
    ss.mode = m if m in MODES else "blink"
    ss.stretch = "asinh"
    ss.speed = 700
    ss.gaia = qp.get("gaia", "0") == "1"      # extras start off; the URL can switch them on
    ss.survey = qp.get("old", "0") == "1"
    ss.showflags = False
    lang = str(qp.get("lang", "en")).lower()
    ss.lang = lang if lang in i18n.LANGS else "en"


def _pick(name):
    ss.target_text = name
    ss.band = PRESET_BAND.get(name, ss.band)


_init()
if targets.is_teleport(ss.get("target_text", "")):   # /RTP: a random spot on the sky, written back as RA Dec
    ss.target_text = teleport.take(teleport.settings(ss.mode, ss.band, ss.fov, ss.gaia, ss.survey))  # loaded ahead
    ss["_rtp"] = ss.target_text


def _teleport_ahead():
    """Keep the next /RTP spots ready (on disk), loading them quietly in the background for the current view."""
    teleport.prepare(teleport.settings(ss.mode, ss.band, ss.fov, ss.gaia, ss.survey))


def _input_hint(text):
    """One line under the search box saying how the text will be read."""
    if targets.preset(text):
        st.caption("A challenge target (built-in catalogue values)")
    elif (t := targets.parse_coords(text)) is not None:
        if ss.get("_rtp") == text:
            st.caption(f"Random teleport (/RTP): RA {t.ra:.5f}°, Dec {t.dec:+.5f}°. Type /RTP again for another spot.")
        else:
            st.caption(f"Read as a sky position: RA {t.ra:.5f}°, Dec {t.dec:+.5f}°")
    elif targets.looks_like_coords(text):
        st.caption("Warning: this looks like numbers but is not a valid position. Use two numbers in degrees: "
                   "RA from 0 to 360, then Dec from −90 to +90, e.g. `269.452 4.693`.")
    elif text.strip():
        st.caption("Will be looked up by name in the SIMBAD star catalogue")


GLOSSARY = """
- **RA, Dec** (right ascension, declination): a position on the sky, like longitude and latitude on Earth,
  in degrees (°). RA runs 0–360, Dec from −90 (south pole) to +90 (north pole).
- **″ arcsecond, ′ arcminute**: tiny angles. 60″ = 1′, 60′ = 1°. The full Moon is about 30′ (1,800″) wide.
- **Pixel**: one dot of the camera. A SPHEREx pixel covers 6.15″ of sky.
- **Band / µm**: SPHEREx sees infrared light, invisible to our eyes, split into six bands D1–D6.
  µm (micrometre) = a millionth of a metre; visible light is 0.4–0.7 µm.
- **Epoch**: one pass of the survey over this spot. SPHEREx comes back about every six months.
- **Proper motion**: a star's own slow drift across the sky, in ″ per year (″/yr).
- **Parallax**: a small yearly back-and-forth wobble of nearby stars, because we watch from a moving Earth.
- **MJy/sr, mJy**: units of brightness (megajansky per steradian for the sky, millijansky for a star).
- **Flagged pixels**: pixels the NASA pipeline marked as unreliable (overexposed, hit by a cosmic ray…).
- **σ (sigma)**: the measurement uncertainty. A change bigger than 3σ is very likely real.
- **Gaia, 2MASS, WISE, SIMBAD**: a star-mapping satellite (ESA), two older infrared sky surveys
  (~2000 and 2010) and a big catalogue of named objects.
"""

# --- sidebar ------------------------------------------------------------------------
i18n.GLOSSARY_EN = GLOSSARY

with st.sidebar:
    st.segmented_control("Language / Dil", list(i18n.LANGS), key="lang", required=True, format_func=i18n.LANGS.get,
                         help="English or Turkish / İngilizce veya Türkçe")
    st.header("SPHEREx Blink")
    st.markdown("**① What to look at**")
    st.text_input("Object name or sky position (RA Dec in degrees)", key="target_text",
                  placeholder="Barnard's Star  ·  M 42  ·  269.452 4.693  ·  /RTP",
                  help="Type the name of a star, galaxy or nebula (looked up in the SIMBAD catalogue), or a "
                       "sky position as two numbers in degrees: RA (0–360) then Dec (−90 to +90), for "
                       "example `269.452 4.693` or `269.452°, +4.693°`. Hours-minutes-seconds also work: "
                       "`17 57 48.5 +04 41 36`. Press Enter or Go. Type `/RTP` (random teleport) to jump to a "
                       "random spot on the sky.")
    st.button("Go", type="primary", use_container_width=True, help="Load this object or position")
    _input_hint(ss.target_text)
    st.caption("…or pick a challenge target:")
    cols = st.columns(2)
    for i, name in enumerate(targets.PRESETS):
        cols[i % 2].button(name, on_click=_pick, args=(name,), use_container_width=True,
                           help=PRESET_HELP.get(name))
    st.markdown("**② Colour and size**")
    st.selectbox("Band (which infrared colour)", list(config.BANDS), key="band", format_func=BAND_LABEL.get,
                 help=BAND_HELP)
    st.radio("Field of view (how much sky)", config.FOV_CHOICES, key="fov", horizontal=True,
             format_func=lambda f: f"{f}′",
             help="Width of the picture in arcminutes (′). 60′ = 1°; the full Moon is about 30′ wide. "
                  "Bigger fields take a little longer to load.")
    st.markdown("**③ Extras**")
    st.toggle("Load the Gaia star list", key="gaia",
              help="Fetches the stars catalogued by ESA's Gaia satellite. They stay hidden until you tick "
                   "*Gaia stars* in the viewer; then each is drawn as a blue circle where it should be on that "
                   "date. Hover a circle for its details.")
    st.toggle("Include older photos (2MASS ~2000, WISE 2010)", key="survey",
              help="Adds images from earlier infrared surveys before the SPHEREx frames, so motion over 15–25 "
                   "years becomes visible. Other telescopes: J → D1/D2, Ks → D3, W1 → D4, W2 → D5/D6.")
    st.caption("Brightness, contrast, speed and flagged pixels are set in the viewer itself.")
    with st.expander("Words used here"):
        st.markdown(GLOSSARY)
    # loading animation, and instant band changes for bands loaded ahead: installed once per page
    st.html(warp.SCRIPT + viewer.BAND_SCRIPT, unsafe_allow_javascript=True)


# write the state back to the URL so it can be shared
_t = targets.parse_coords(ss.target_text)
if _t:
    qp.pop("target", None)
    qp["ra"], qp["dec"] = f"{_t.ra:.5f}", f"{_t.dec:.5f}"
else:
    for _k in ("ra", "dec"):
        qp.pop(_k, None)
    qp["target"] = ss.target_text
qp["fov"], qp["band"], qp["mode"], qp["lang"] = str(ss.fov), ss.band, ss.mode, ss.lang


# --- data -------------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def resolve(text):
    return targets.resolve(text)


def search(ra, dec):  # cached on disk by archive.search (fast list first, complete list soon after)
    return archive.search(ra, dec)


_SAME = object()


class Loader:
    """The loading screen with the current step and the time left, counting down every second. It appears only when
    there is something to wait for: a step that asks NASA's archive (slow=True), or a page that has already taken
    config.LOADER_AFTER. What is at hand (a band or view loaded ahead, a display change) shows with no screen.

    Two looks: the star-field flight (spx/warp.py card), and for the Motion check (bar=True) one progress bar for the
    whole check, download to fitted path, with its steps. left: seconds still to wait for the whole page (None: no
    honest estimate, it says so; False: no countdown yet); the steps after the current one are included."""

    def __init__(self, bar=False):
        self.box, self.t0 = st.empty(), time.time()
        self.title, self.detail, self.stage, self.frac, self.ptext = "", "", 0, None, ""
        self.shown, self.bar = False, bar
        self.est = eta.Estimator()
        self.left, self.lid, self.overall = False, f"{self.t0:.3f}", 0.0
        self.tail = self.est.usual("motion") if bar else 0.0   # the motion analysis after the photos

    def _set_left(self, left):
        if left is _SAME:
            return
        if left is None or left is False:
            self.left = left
        else:
            self.left = max(0.0, left) + (self.tail if self.stage < 4 else 0.0)

    def _draw(self, now=False):
        if not (now or self.shown or time.time() - self.t0 > config.LOADER_AFTER):
            return
        self.shown = True
        T = lambda s: " · ".join(i18n.tr(p, ss.lang) for p in s.split(" · "))  # noqa: E731
        # (the card is HTML, so it is translated here, part by part: "3 of 12 photos · 14 s so far")
        words = dict(fs=T("{} s left"), fm=T("{} min left"), soon=T("almost done"), slow=T("taking longer than usual"),
                     now=T(warp.eta_text(self.left)) if self.left is not False else "")
        if self.bar:
            w = warp.MOTION_WEIGHTS
            k = min(self.stage, len(w))
            self.overall = max(self.overall, sum(w[:k]) + (w[k] * (self.frac or 0.0) if k < len(w) else 0.0))
            html = warp.bar_card(T(self.title), T(self.detail), self.stage, self.overall, T(self.ptext),
                                 stages=[T(s) for s in warp.MOTION_STAGES], left=self.left, lid=self.lid, words=words)
        else:
            html = warp.card(T(self.title), T(self.detail), self.stage, self.frac, T(self.ptext),
                             stages=[T(s) for s in warp.STAGES], left=self.left, lid=self.lid, words=words)
        self.box.markdown(html, unsafe_allow_html=True)

    def step(self, title, detail="", stage=0, slow=False, left=_SAME):
        if stage != self.stage:
            self.frac, self.ptext = None, ""
        self.title, self.detail, self.stage = title, detail, stage
        self._set_left(left)
        self._draw(slow)

    def progress(self, frac, text, left=_SAME):
        old = (self.frac, self.ptext, self.left)
        self.frac, self.ptext = min(1.0, frac), text
        self._set_left(left)
        if (self.frac, self.ptext, self.left) != old:
            self._draw()

    def note(self, text, left=_SAME):
        """A line under the steps without a progress bar (e.g. how long we have been waiting)."""
        old = (self.ptext, self.left)
        self.ptext = text
        self._set_left(left)
        if (self.ptext, self.left) != old:
            self._draw()

    def done(self, arrive=True):
        secs = time.time() - self.t0
        if self.shown:
            if self.bar and arrive:   # the bar fills up in a snap, then the check is there
                self.title, self.detail, self.ptext = "Motion check ready", "", ""
                self.stage, self.frac, self.left = len(warp.MOTION_STAGES), 1.0, False
                self._draw()
                time.sleep(0.15)
            elif arrive and secs > 0.8:  # long enough to have been seen: slow down and fade out. The browser keeps
                self.stage = 4           # that going by itself (warp.SCRIPT) while the photos are drawn underneath
                self._draw()
                time.sleep(warp.HANDOVER)
            self.box.empty()
            self.shown = False
        _teleport_ahead()   # the waiting is over: get the next /RTP spots ready in the background
        return secs


st.title("SPHEREx Blink Viewer")
st.caption("SPHEREx photographs the whole sky every six months in 102 infrared colours. "
           "Flip between its photos of one spot: whatever moves or changes brightness stands out.")
st.segmented_control("View", list(MODES), key="mode", required=True, format_func=MODE_LABEL.get,
                     label_visibility="collapsed",
                     help="How to look at the photos:\n\n" + "\n".join(f"- **{MODE_LABEL[m]}**: {MODE_HELP[m]}"
                                                                       for m in MODES))
st.caption(f"**{MODE_LABEL[ss.mode]}**: {MODE_HELP[ss.mode]}")
with st.expander(f"How to use {MODE_LABEL[ss.mode]} (30 seconds)"):
    st.markdown(HOW_TO[ss.mode])
    st.caption("Hover over the small question-mark icons for short explanations; *Words used here* in the sidebar "
               "explains the astronomy terms.")
load = Loader(bar=ss.mode == "motion")
_known = ss.setdefault("_resolved", set())   # names looked up already (answered from memory)
_lookup = not (targets.preset(ss.target_text) or targets.parse_coords(ss.target_text) or ss.target_text in _known)
load.step("Finding your target…", "Reading the position you typed, or looking the name up in the SIMBAD catalogue", 0,
          slow=_lookup, left=load.est.usual("resolve") + load.est.usual("search") + load.est.usual("fetch")
          if _lookup else False)
try:
    _t_resolve = time.time()
    tgt = resolve(ss.target_text)
    if _lookup and time.time() - _t_resolve > 0.3:
        load.est.learn_step("resolve", time.time() - _t_resolve)
    _known.add(ss.target_text)
except targets.TargetNotFound:
    load.done(arrive=False)
    if targets.looks_like_coords(ss.target_text):
        st.error(f"**{ss.target_text}** is not a valid sky position. Use two numbers in degrees: RA from 0 to 360, "
                 "then Dec from −90 to +90, for example `269.452 4.693` or `269.452, -4.693`.")
    else:
        st.error(f"Could not find **{ss.target_text}** in the SIMBAD catalogue. Check the spelling, try another "
                 "name (e.g. *M 42*, *HD 209458*), or type a position like `269.452 4.693`.")
    st.stop()
except Exception as e:  # SIMBAD down
    load.done(arrive=False)
    st.error(f"Name lookup failed ({e}). Coordinates like `269.452 4.693` always work.")
    st.stop()

ra0, dec0 = targets.position_at(tgt, config.REFERENCE_MJD)
band = ss.band
# Side requests that only need the position start now and run while SPHEREx images load.
_bg = ThreadPoolExecutor(2)
fut_gaia = (_bg.submit(catalogs.gaia_cone, round(ra0, 4), round(dec0, 4), round(ss.fov / 60 * 0.75, 3))
            if ss.gaia and ss.mode in ("blink", "sub", "movie") else None)
fut_survey = (_bg.submit(catalogs.survey_frames, round(ra0, 5), round(dec0, 5), band, ss.fov * 1.45)
              if ss.survey and ss.mode in ("blink", "sub", "motion") else None)
_bg.shutdown(wait=False)

def waiting(t0, what="", search=False):
    """Plain words for the loading card: how long we have waited, and why when it is long (the time left is the
    countdown above it). search: the archive search, which says so after 8 s."""
    el = time.time() - t0
    if search and el > 8:
        why = (" · NASA's archive is not answering quickly right now: both of its search services are being asked, "
               "the first answer wins")
    elif el > 12:
        why = (" · NASA's servers or the connection are slow right now: it keeps trying, and whatever arrives is "
               "saved, so the next visit is fast")
    else:
        why = ""
    return what + f"{el:.0f} s so far" + why


_searched = archive.is_cached(round(ra0, 4), round(dec0, 4))
load.step("Asking NASA's archive which photos cover this spot…",
          "SPHEREx has photographed every part of the sky several times since 2025", 1, slow=not _searched,
          left=(0.2 if _searched else load.est.usual("search")) + load.est.usual("fetch"))
if not _searched:      # a new spot: the photos come right after the search,
    cutouts.warm()     # so their connections open while the archive answers
try:
    _t_search, _ex = time.time(), ThreadPoolExecutor(1)
    _fs = _ex.submit(search, round(ra0, 4), round(dec0, 4))
    _ex.shutdown(wait=False)
    while True:  # keep the loading card alive while the archive answers
        try:
            exps = _fs.result(timeout=0.5)
            if time.time() - _t_search > 0.3:   # an answer from the archive, not from this computer's memory
                eta.Estimator().learn_step("search", time.time() - _t_search)   # better estimates next time
            break
        except FutureTimeout:
            _el = time.time() - _t_search
            _sl = load.est.step_left("search", _el)
            load.progress(min(0.95, _el / (_el + _sl)) if _sl is not None else 0.95, waiting(_t_search, search=True),
                          left=None if _sl is None else _sl + load.est.usual("fetch"))
except archive.ArchiveBusy:
    load.done(arrive=False)
    st.warning("The NASA/IPAC archive is busy right now. Please try again in a minute.")
    st.stop()
if not exps:
    load.done(arrive=False)
    st.info(f"SPHEREx has no images covering {tgt.name} yet. New data arrive weekly.")
    st.stop()


def fetch(todo, label, group_of=None, lineup=True):
    """Download the cutouts. With group_of (uri -> epoch), one photo per epoch comes first, and the page is shown
    once every epoch has one (after config.FETCH_SOFT_DEADLINE): the rest keep arriving and are filled in.
    lineup: the photos are lined up next (a second or so for 'Load all'); the spectrum only measures them."""
    have = {e.uri: cutouts.is_cached(e, ra0, dec0, ss.fov * 1.45) for e in todo}
    n_cached = sum(have.values())
    load.step(f"Flying in: downloading {len(todo)} SPHEREx photos ({label})…",
              "Reading only the pixels around your target from NASA's archive", 2, slow=n_cached < len(todo))
    stats, enough, lead_uris = {}, None, None
    if group_of:
        need = set(group_of.values())

        def enough(got):
            return need <= {group_of.get(x.uri) for x in got}
        done_k = {group_of.get(u) for u, c in have.items() if c}   # epochs with a photo at hand already
        lead = {}
        for e in todo:
            k = group_of.get(e.uri)
            if k not in done_k:
                lead.setdefault(k, e.uri)
        lead_uris = set(lead.values())
    est = load.est
    # the page is shown once every date has a photo: the countdown runs to then, not to the last photo
    new_needed = len(lead_uris) if lead_uris is not None else len(todo) - n_cached
    target = max(1, n_cached + new_needed)
    rest_new = len(todo) - n_cached - new_needed > 0
    align = eta.ALIGN_SECONDS * target if lineup else 0.0
    soft = config.FETCH_SOFT_DEADLINE if group_of and rest_new else 0.0
    left0 = max(est.start(new_needed, n_cached), soft) + align
    load.progress(0.0, f"0 of {len(todo)} photos", left=left0)
    t_fetch = time.time()

    def progress(d, n):
        el = time.time() - t_fetch
        left = est.update(min(d, target), target, el)
        left = None if left is None else max(left, soft - el) + align
        load.progress(d / n, waiting(t_fetch, f"{d} of {n} photos · "), left=left)
    out, nfail = cutouts.fetch_many(todo, ra0, dec0, ss.fov * 1.45, progress,
                                    deadline=config.FETCH_SOFT_DEADLINE if group_of else None,
                                    enough=enough, stats=stats, first=lead_uris)
    if not out:
        load.done(arrive=False)
        st.warning("No photos could be downloaded (the archive or your connection is busy). "
                   "Please try again in a minute.")
        st.stop()
    ss["_skipped"] = stats.get("skipped", 0)
    est.learn(max(0, len(out) - n_cached), time.time() - t_fetch)   # better estimates next time
    if n_cached < len(todo):
        est.learn_step("fetch", time.time() - t_fetch)
    load.progress(1.0, f"{len(out)} photos" + (f" ({nfail} failed)" if nfail else ""))
    load.step("Lining the photos up…", "Turning every photo north-up on one grid and averaging each epoch", 3,
              slow=lineup and len(out) > 40,   # 'Load all': a second or so of work
              left=eta.ALIGN_SECONDS * len(out) if lineup else 0.0)
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _gaia_near(ra, dec):
    return catalogs.gaia_cone(ra, dec, config.FIND_CONE_DEG)[0]


def _drop_find():
    ss.pop("find", None)


def resolve_find():
    """The object of 'Check my find' in this field: (Target, plain-English note), or (None, "")."""
    f = ss.get("find")
    if not f or f.get("field") != tgt.name:
        return None, ""
    m = f["marks"][0] if f.get("marks") else f["mark"]
    try:
        star, sep = catalogs.nearest(_gaia_near(round(m["ra"], 5), round(m["dec"], 5)),
                                     m["ra"], m["dec"], m["mjd"], config.FIND_MATCH_ARCSEC)
        trouble = ""
    except Exception:
        star, sep = None, None
        trouble = "The Gaia catalogue could not be reached right now. "
    if star:
        return catalogs.as_target(star, name=f"Your find (Gaia DR3 {star['id']})"), (
            f"Your mark ({m.get('date', '')}) at RA {m['ra']:.5f}°, Dec {m['dec']:+.5f}° matches Gaia DR3 "
            f"{star['id']} (G = {star['g']:.1f}), {sep:.1f}″ away on that date.")
    from spx.targets import Target
    return Target("Your find", m["ra"], m["dec"], m["mjd"], source="your mark (no catalogue match)"), (
        trouble + f"No Gaia star within {config.FIND_MATCH_ARCSEC:g}″ of your mark (RA {m['ra']:.5f}°, "
        f"Dec {m['dec']:+.5f}°), so there is no prediction: it is treated as standing still at the spot you "
        "marked, and what SPHEREx measures is its own. Very cold or very faint objects are often missing from Gaia.")


if ss.mode == "spectrum":
    spec_t, find_note = resolve_find()
    spec_t = spec_t or tgt
    spec_key = f"specall|{tgt.name}"
    spec_all = ss.get(spec_key, False)
    todo, epoch_of = [], {}
    for b in config.BANDS:
        for k, g in archive.select(exps, b, quick=not spec_all, cap=config.SPEC_CAP_PER_BAND,
                                   quick_total=config.SPEC_QUICK_PER_BAND).items():
            for e in g:
                todo.append(e)
                epoch_of[e.uri] = k + 1
    nep = max(epoch_of.values())
    fold = nep > len(render.EPOCH_COLORS)  # dense fields: merge time bins so colours never repeat
    _nbx = ThreadPoolExecutor(1)   # Gaia neighbours of the target (for the warning below), while the photos load
    fut_nb = _nbx.submit(catalogs.gaia_cone, *(round(v, 4) for v in targets.position_at(spec_t, config.REFERENCE_MJD)),
                         0.02)
    _nbx.shutdown(wait=False)
    stamps = fetch(todo, "all-band", {e.uri: e.band for e in todo}, lineup=False)   # shown once every band has one
    rows = []
    for s in stamps:
        m = photometry.measure(s, *targets.position_at(spec_t, s.mjd))
        k = epoch_of[s.uri]
        m.update(date=targets.mjd_to_date(s.mjd), datetime=Time(s.mjd, format="mjd").iso,
                 epoch=1 + (k - 1) * len(render.EPOCH_COLORS) // nep if fold else k)
        rows.append(m)
    by_epoch = {}
    for r in sorted(rows, key=lambda r: (r["epoch"], r["wavelength"])):
        by_epoch.setdefault(r["epoch"], []).append(r)
    load_secs = load.done()
    prefetch.start(tgt, ra0, dec0, ss.fov, band, exps, include=True)   # every band's view, ahead
    st.subheader(f"{spec_t.name}: SPHEREx spectrum, 0.75–5.0 µm")
    st.caption(f"{len(stamps)} images loaded in {load_secs:.1f} s. " + find_note)
    if find_note:
        st.button(f"← Back to {tgt.name}", on_click=_drop_find, help="Forget this find and measure the target again.")
    close, nb_pending = {}, False
    try:  # warn when a Gaia neighbour falls inside the 12" aperture (SPHEREx pixels are 6.15")
        nb, _ = fut_nb.result(timeout=config.WAIT_EXTRAS)   # not there yet: added when it arrives (below)
        mids = [min(r["mjd"] for r in rows), max(r["mjd"] for r in rows)]
        pr0, pd0 = targets.position_at(spec_t, mids[0])
        gr0, gd0 = catalogs.gaia_at(nb, mids[0])
        sep0 = np.hypot((gr0 - pr0) * np.cos(np.radians(pd0)), gd0 - pd0) * 3600
        own = [s["g"] for s, d in zip(nb, sep0) if d < 2.0]
        g_limit = (min(own) + 3) if own else 99  # only neighbours within 3 mag of the target matter
        for m in mids:
            pr, pd_ = targets.position_at(spec_t, m)
            gr, gd = catalogs.gaia_at(nb, m)
            sep = np.hypot((gr - pr) * np.cos(np.radians(pd_)), gd - pd_) * 3600
            for s, d in zip(nb, sep):
                if 2.0 < d < 15 and s["g"] < g_limit:  # < 2" is the target's own Gaia entry
                    close.setdefault(s["id"], (s["g"], []))[1].append(d)
        for g, ds in close.values():
            dist = f"{min(ds):.0f}″" if round(min(ds)) == round(max(ds)) else f"{min(ds):.0f}–{max(ds):.0f}″"
            st.warning(f"A Gaia star (G = {g:.1f}) lies {dist} from the target on these dates. "
                       "SPHEREx pixels are 6.15″, so its light blends into the measurement — especially at short "
                       "wavelengths where a faint, cold target is dark.")
    except FutureTimeout:
        nb_pending = True
    except Exception:
        pass
    _have = {s.uri for s in stamps}
    _missing = [e for e in todo if e.uri not in _have] if ss.get("_skipped") else []
    if _missing or nb_pending:   # still arriving (photos, or the Gaia check): the page fills it in by itself
        _akey, _t_wait = f"_auto|spec|{tgt.name}|{ss.fov}|{spec_all}", time.time()

        @st.fragment(run_every=config.REFRESH_EVERY)
        def spec_watch():
            photos_in = bool(_missing) and all(cutouts.is_cached(e, ra0, dec0, ss.fov * 1.45) for e in _missing)
            if (photos_in or (not _missing and fut_nb.done())) and ss.get(_akey, 0) < 5:
                ss[_akey] = ss.get(_akey, 0) + 1
                st.rerun()
            if _missing and time.time() - _t_wait > 3:
                st.info(f"{len(_missing)} more SPHEREx photos are still on their way; the spectrum fills in by itself "
                        "as soon as they arrive.")
        spec_watch()
    s1, s2 = st.columns([3, 1], vertical_alignment="bottom")
    ep_label = {k: f"Epoch {k}: {v[0]['date']}" for k, v in by_epoch.items() if v}
    show_eps = s1.multiselect("Dates to show", list(ep_label), default=list(ep_label), format_func=ep_label.get,
                              help="Pick two dates to compare their spectra directly; remove one to hide it.")
    logy = s2.toggle("Logarithmic flux axis", value=False,
                     help="Spreads out faint and bright values; useful when the spectrum spans a big range.")
    st.plotly_chart(render.spectrum_fig({k: v for k, v in by_epoch.items() if k in show_eps}, logy),
                    use_container_width=True)
    st.caption("Each point is one exposure; SPHEREx sees each spot at a slightly different wavelength every "
               "time, so many exposures together make a spectrum. Colours are survey epochs"
               + (" (time bins merged)" if fold else "") + ". Hollow points have flagged pixels in the aperture "
               "(often saturation): treat them with care. Quick look, not calibrated; for science use the "
               f"[IRSA SPHEREx Spectrophotometry Tool]({config.SPECTROPHOTOMETRY_TOOL}).")
    nbad = sum(not np.isfinite(r["flux"]) for r in rows)
    if nbad:
        sat = sum(bool(r["flags"]) and r["flags"][0].startswith("SATURATED") for r in rows)
        bands_sat = sorted({r["band"] for r in rows if r["flags"] and r["flags"][0].startswith("SATURATED")})
        st.info(f"{nbad} of {len(rows)} exposures could not be measured"
                + (f"; in {sat} of them the target is so bright that the pixels at its centre are saturated "
                   f"(bands {', '.join(bands_sat)}). They are listed in the table, not plotted." if sat else "."))
    st.subheader(f"Light curve in {band} (choose the band in the sidebar)")
    st.plotly_chart(render.lightcurve_fig(by_epoch, band, logy), use_container_width=True)
    wing_rows = {k: [dict(r, flux=r["wing"], err=0.0, flags=[]) for r in v
                     if r["band"] == band and np.isfinite(r.get("wing", np.nan))] for k, v in by_epoch.items()}
    if any(wing_rows.values()):
        st.markdown(f"**Saturated exposures: brightness index from the star's wings ({band})**")
        st.plotly_chart(render.lightcurve_fig(wing_rows, band, False,
                                              ytitle="Wing brightness (MJy/sr, relative)"),
                        use_container_width=True)
        st.caption("When the centre of a very bright star is saturated, the light 3–5 pixels from the centre still "
                   "rises and falls with the star's brightness. This index shows the trend only; it is not a flux, "
                   "and wavelength differences inside the band add scatter.")
    # plain-language change report per band: median clean flux, first vs last epoch
    notes = []
    for b in config.BANDS:
        per = {k: [r for r in v if r["band"] == b and not r["flags"] and np.isfinite(r["flux"])]
               for k, v in by_epoch.items()}
        per = {k: v for k, v in per.items() if len(v) >= 2}
        flagged_note = ""
        if len(per) < 2:  # bright stars: every point is flagged; use them, and say so
            per = {k: [r for r in v if r["band"] == b and np.isfinite(r["flux"])] for k, v in by_epoch.items()}
            per = {k: v for k, v in per.items() if len(v) >= 2}
            flagged_note = " (from flagged points)"
        if len(per) >= 2:
            k1, k2 = min(per), max(per)
            f1, f2 = (np.median([r["flux"] for r in per[k]]) for k in (k1, k2))
            e = np.hypot(*(1.25 * np.std([r["flux"] for r in per[k]]) / np.sqrt(len(per[k])) for k in (k1, k2)))
            sig = abs(f2 - f1) / e if e > 0 else 0
            word = "brighter" if f2 > f1 else "fainter"
            notes.append(f"**{b}**: epoch {k2} is {abs(f2 / f1 - 1) * 100:.0f}% {word} than epoch {k1}"
                         + (" — a real change is likely" if sig > 3 else " — within the scatter")
                         + f" ({sig:.1f}σ){flagged_note}")
    if notes:
        st.markdown("**What changed?** (median of unflagged points; different wavelengths inside a band add "
                    "scatter)\n\n- " + "\n- ".join(notes)
                    + ("\n\nNote: the neighbouring star above moves relative to the target, so part of any change "
                       "can come from it entering or leaving the aperture." if close else ""))
    with st.expander("Table of measurements"):
        df = pd.DataFrame([{k: (", ".join(v) if k == "flags" else v) for k, v in r.items()
                            if k not in ("x", "y", "datetime")} for r in rows]).sort_values(["band", "mjd"])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), f"spherex_{tgt.name.replace(' ', '_')}.csv")
    if not spec_all:
        st.button(f"Load more exposures (up to {config.SPEC_CAP_PER_BAND} per band)",
                  on_click=lambda: ss.__setitem__(spec_key, True))
    st.stop()

load_key = f"all|{tgt.name}|{band}"
load_all = ss.get(load_key, False)
groups = {k + 1: v for k, v in archive.select(exps, band, quick=not load_all).items()}
n_band = sum(e.band == band for e in exps)
todo = [e for g in groups.values() for e in g]
# cutouts are 1.45x the field so rotated frames cover the square grid
stamps = fetch(todo, band, {e.uri: k for k, g in groups.items() for e in g})

wgrid, shape = stack.grid(ra0, dec0, ss.fov)
arcsec_px = config.PIXEL_SCALE_ARCSEC / config.OVERSAMPLE


@st.cache_resource(max_entries=8, show_spinner=False)
def aligned_frames(key, _stamps, _groups, _tgt):
    """Line every photo up on the grid and average each epoch. Kept in memory, so changing a display
    control or the view never lines the photos up again. key: target, field, band, photos. A /RTP spot
    or another band loaded ahead was lined up in the background already."""
    return teleport.aligned(key) or prefetch.aligned(key) or stack.epoch_frames(_stamps, _groups, _tgt, wgrid, shape)


by_uri, frames = aligned_frames(stack.frames_key(tgt.name, ra0, dec0, ss.fov, band, groups, stamps),
                                stamps, groups, tgt)
frames = list(frames)
sphx_frames = list(frames)


@st.cache_resource(max_entries=4, show_spinner=False)
def old_survey_frames(key, _items, ref):
    """WISE / 2MASS frames on the same grid (kept in memory like aligned_frames)."""
    return sorted([f for f in (stack.survey_frame(it, wgrid, shape, ref, band) for it in _items) if f is not None],
                  key=lambda f: f.mjd_start)


survey_pending = False
if fut_survey is not None and frames:
    try:
        if not fut_survey.done():
            load.step("Adding the older WISE and 2MASS photos…", "From 2010 and around 2000, to compare with today", 3)
        try:  # never hold the SPHEREx movie back for long: the download continues and is cached
            items = fut_survey.result(timeout=config.WAIT_EXTRAS)
        except FutureTimeout:
            items, survey_pending = [], True
        ref = float(np.nanmax([np.nanpercentile(f.image, 99.5) for f in sphx_frames]))
        skey = (round(ra0, 6), round(dec0, 6), ss.fov, band,
                tuple((it["survey"], it["band"], tuple(it["mjds"]), len(it["images"])) for it in items))
        frames = list(old_survey_frames(skey, items, ref)) + frames
    except Exception as e:
        st.caption(f"Older survey images unavailable ({type(e).__name__}).")


def target_at(mjd, t=None):
    """Where the target (or t) is on that date, in grid pixels (a ring), or the typed position (a cross)."""
    if t is not None:
        x, y = wgrid.world_to_pixel_values(*targets.position_at(t, mjd))
        return dict(x=float(x), y=float(y), label=f"{t.name}: predicted position on {targets.mjd_to_date(mjd)}")
    if tgt.source == "coordinates":  # nothing to track: just mark the position that was asked for
        x, y = wgrid.world_to_pixel_values(tgt.ra, tgt.dec)
        return dict(x=float(x), y=float(y), cross=True,
                    label=f"RA {tgt.ra:.5f}, Dec {tgt.dec:+.5f} (the position you typed)")
    x, y = wgrid.world_to_pixel_values(*targets.position_at(tgt, mjd))
    return dict(x=float(x), y=float(y), label=f"{tgt.name}: predicted position on {targets.mjd_to_date(mjd)}")


gaia_stars, gaia_note, gaia_pending = [], "", False
if fut_gaia is not None:
    try:  # the photos are ready: wait only briefly for the star list (it is cached once it arrives)
        gaia_stars, src = fut_gaia.result(timeout=config.WAIT_EXTRAS)
        gaia_note = f"{len(gaia_stars)} Gaia DR3 stars (G < {config.GAIA_MAX_G:g}, via {src})"
    except FutureTimeout:
        gaia_pending = True
        gaia_note = "Gaia star list still loading"
    except Exception:
        gaia_note = "Gaia catalogue unavailable right now."
gaia_text = [catalogs.gaia_hover(s) for s in gaia_stars]


def gaia_on(mjd):
    """Gaia stars moved to that date, in grid pixels, with their hover text."""
    if not gaia_stars:
        return []
    r, d = catalogs.gaia_at(gaia_stars, mjd)
    x, y = wgrid.world_to_pixel_values(r, d)
    return [dict(x=round(float(a), 2), y=round(float(b), 2), text=t)
            for a, b, t in zip(np.atleast_1d(x), np.atleast_1d(y), gaia_text)
            if -2 < a < shape[1] + 1 and -2 < b < shape[0] + 1]


COORD_NOTE = ("You typed a sky position, so nothing is followed here (the orange cross marks the spot). "
              "Look for anything that moves or changes yourself, then **Mark** it on two dates to measure it.")


def view_data(shown_frames, b, n_all, n_sphx, diverging=False, key_extra="", stars=None, changes=None, follow=None):
    """What the sky viewer gets for these frames of band b: (data, store key, index of the first SPHEREx frame).
    n_all / n_sphx: how many frames the band has in all / from SPHEREx (the rest are older surveys)."""
    cx, cy = (float(v) - 1 for v in wgrid.wcs.crpix)
    data = viewer.payload(shown_frames, target=[target_at(f.mjd_mid, follow) for f in shown_frames],
                          gaia=[gaia_on(f.mjd_mid) for f in shown_frames], scale=arcsec_px, center=(cx, cy),
                          radec=(ra0, dec0), diverging=diverging, mode=ss.mode,
                          title=f"{tgt.name}|{b}|{ss.fov}|{key_extra}", stars=stars, changes=changes)
    first_sphx = len(shown_frames) - n_sphx if len(shown_frames) == n_all else 0
    return data, f"{tgt.name}|{ss.fov}|{b}|{key_extra}", first_sphx


def sky_viewer(shown_frames, diverging=False, key_extra="", stars=None, changes=None, follow=None, wkey=None):
    """The sky viewer (spx/viewer) on the given frames. Returns the user's marks (list of dicts).
    stars / changes: results of the optional helpers (None while they are switched off). wkey: a second viewer on
    the same page (object B of a comparison) needs a key of its own."""
    data, store, first_sphx = view_data(shown_frames, band, len(frames), len(sphx_frames), diverging, key_extra,
                                        stars, changes, follow)
    res = viewer.show(data, key=wkey or f"viewer|{ss.mode}", store_key=store,
                      settings=dict(stretch=ss.stretch, speed=ss.speed, gaia=False, flags=ss.showflags, lang=ss.lang,
                                    a=first_sphx))
    if isinstance(res, dict) and res.get("key") == "spx-measure:" + store:
        ss["_marks"] = dict(res, store=store)
    got = ss.get("_marks") or {}
    return got.get("marks", []) if got.get("store") == store else []


def pending_notes():
    """Say so when part of the data is still arriving over a slow connection; redraw when it has."""
    msgs = []
    if ss.get("_skipped"):
        msgs.append(f"{ss['_skipped']} more SPHEREx photos are still on their way (__LEFT__); each epoch already "
                    "has at least one")
    if survey_pending:
        msgs.append("the older WISE (2010) / 2MASS (~2000) photos are still downloading")
    if gaia_pending:
        msgs.append("the Gaia star list is still loading")
    if not msgs:
        return
    have = {s.uri for s in stamps}
    missing = [e for e in todo if e.uri not in have] if ss.get("_skipped") else []
    akey = f"_auto|{tgt.name}|{band}|{ss.fov}"
    t_first = ss.setdefault(akey + "|t", time.time())
    t_wait = time.time()
    per = eta.Estimator().per_photo
    clock = dict(end=t_wait + per * len(missing), n=len(missing))   # when the last one should be in

    @st.fragment(run_every=config.REFRESH_EVERY)
    def watch():  # checks every second; redraws the whole page once everything still missing has arrived
        still = [e for e in missing if not cutouts.is_cached(e, ra0, dec0, ss.fov * 1.45)]
        arrived = ((not survey_pending or fut_survey is None or fut_survey.done())
                   and (not gaia_pending or fut_gaia is None or fut_gaia.done()) and not still)
        if arrived and ss.get(akey, 0) < 5 and time.time() - t_first < 600:
            ss[akey] = ss.get(akey, 0) + 1
            st.rerun()
        now = time.time()
        if still and len(still) < clock["n"]:   # some came in: the speed seen so far says when the rest will be
            got = len(missing) - len(still)
            end = now + len(still) * (now - t_wait) / got
            if end < clock["end"] or end - clock["end"] > 3:
                clock["end"] = end
            clock["n"] = len(still)
        left = clock["end"] - now
        when = (f"about {round(left / 60)} min left" if left > 89 else f"{int(left) + 1} s left" if left > 0
                else "any moment now")
        if now - t_wait > 3:   # a note only when it takes a while (usually it is over within a second or two)
            st.info("; ".join(msgs).replace("__LEFT__", when)
                    + ". They are added here automatically as soon as they arrive.")
    watch()


def _go_track(marks, marks_b=None):
    ss["find"] = dict(marks=marks, marks_b=marks_b or [], field=tgt.name)
    ss["mode"] = "motion"


def _go_spectrum(mark):
    ss["find"] = dict(mark=mark, field=tgt.name)
    ss["mode"] = "spectrum"


def check_my_find(marks, key):
    """'Check my find', only on request: every chosen mark of one object is compared with all the others."""
    if not marks:
        return
    fmt = lambda i: f"#{i + 1} · {marks[i].get('date', '')} · RA {marks[i]['ra']:.5f}°, Dec {marks[i]['dec']:+.5f}°"  # noqa
    with st.container(border=True):
        st.markdown("**Check my find**")
        if len(marks) < 2:
            st.caption("Mark the same object on two or more photos (different dates) to measure how it moves. Every "
                       "mark is compared with every other one, and the path is predicted into the future.")
            st.button("Spectrum of mark #1", on_click=_go_spectrum, args=(marks[0],), key=f"find_sp|{key}",
                      help="Measures the marked object's brightness at every infrared colour SPHEREx sees.")
            return
        # your marks sorted into objects (marks on one photo are different objects); only a first guess to change
        groups = motion.group_marks(marks)
        two = len(groups) >= 2 and all(len({marks[i].get("uid") or round(marks[i]["mjd"] * 86400) for i in g}) >= 2
                                       for g in groups[:2])
        pick = st.multiselect("Marks of object A (all of them are fitted together)", range(len(marks)),
                              default=groups[0] if two else list(range(len(marks))), format_func=fmt,
                              key=f"find_pick|{key}|{len(marks)}",
                              help="Keep the marks you made on the same object, on different photos. Leave out marks "
                                   "of other objects. Three or more marks let the app check that they lie on one "
                                   "path and spot a mark that does not belong.")
        pick_b = st.multiselect("Marks of object B, to compare with A (optional)",
                                [i for i in range(len(marks)) if i not in pick],
                                default=[i for i in groups[1] if i not in pick] if two else [], format_func=fmt,
                                key=f"find_pick_b|{key}|{len(marks)}",
                                help="Mark a second object on the same photos (for example a star next to an "
                                     "asteroid, on two exposures minutes apart) and put its marks here. Both get "
                                     "their own motion check, and the app compares them: their distance on each "
                                     "photo, how fast they move relative to each other, whether they drift apart or "
                                     "close in, and how far apart they will be.")
        sel = [dict(marks[i], label=f"#{i + 1}") for i in sorted(pick)]
        sel_b = [dict(marks[i], label=f"#{i + 1}") for i in sorted(pick_b)]
        photos = lambda ms: len({m.get("uid") or round(m["mjd"] * 86400) for m in ms})   # noqa: E731
        n_photos = photos(sel)   # exposures can be minutes apart
        b1, b2 = st.columns(2)
        b1.button(f"Analyse and compare A ({len(sel)} marks) and B ({len(sel_b)} marks)" if sel_b else
                  f"Analyse the motion of these {len(sel)} marks", type="primary", on_click=_go_track,
                  args=(sel, sel_b), key=f"find_go|{key}", use_container_width=True,
                  disabled=n_photos < 2 or (bool(sel_b) and photos(sel_b) < 2),
                  help="Fits one path through all the marks at once, tests whether the object stands still, drifts "
                       "like a star (with or without the yearly parallax wobble) or curves like an asteroid, says "
                       "what kind of object it probably is and why, and predicts where it will be in the coming "
                       "years. Each mark is measured only on the photo it was made on.")
        b2.button("Spectrum of the first mark", on_click=_go_spectrum, args=((sel or marks)[0],),
                  key=f"find_sp|{key}", use_container_width=True,
                  help="Measures the marked object's brightness at every infrared colour SPHEREx sees.")
        if n_photos < 2:
            st.caption("These marks are all on the same photo: pick marks made on at least two different photos.")
        elif sel_b and photos(sel_b) < 2:
            st.caption("Object B needs marks on at least two different photos too.")


def motion_check(t, note=""):
    """Measured positions of target t in every frame against its predicted path (plot, metrics, table)."""
    r_ref, d_ref = targets.position_at(t, config.REFERENCE_MJD)

    def offset(px, py):  # arcsec east / north of t's catalogue position on 2026-01-01
        r, d = wgrid.pixel_to_world_values(px, py)
        de = ((r - r_ref + 180) % 360 - 180) * np.cos(np.radians(d_ref)) * 3600
        return float(de), float((d - d_ref) * 3600)

    t_motion = time.time()
    usual = load.est.usual("motion")
    load.step("Measuring where the object is in each photo…", "The centre of its light on every date", 4,
              left=usual)
    st.subheader(f"Motion check: {t.name} in {band}")
    if note:
        st.caption(note)
    pts = []
    for i_f, f in enumerate(frames):
        load.progress(i_f / max(1, len(frames)), f"{i_f} of {len(frames)} photos measured",
                      left=max(0.1, usual - (time.time() - t_motion)))
        r, d = targets.position_at(t, f.mjd_mid)
        par = targets.predicted_offsets(t, [f.mjd_mid], config.REFERENCE_MJD)[0] - \
            targets.predicted_offsets(t, [f.mjd_mid], config.REFERENCE_MJD, with_parallax=False)[0]
        gx, gy = wgrid.world_to_pixel_values(r, d)
        gx, gy = float(gx) - par[0] / arcsec_px, float(gy) + par[1] / arcsec_px  # east is -x
        cx, cy = photometry.centroid(f.image, gx, gy, r=4)
        if not np.isfinite(cx):
            continue
        de, dn = offset(cx, cy)
        err = 0.6 * (1 if f.kind == "survey" else 2)
        if f.kind == "stack" and len(f.stamps) >= 3:  # scatter of single-exposure centroids
            singles = [photometry.centroid(by_uri[s.uri].image, cx, cy, r=4) for s in f.stamps if s.uri in by_uri]
            singles = np.array([offset(*c) for c in singles if np.isfinite(c[0])])
            if len(singles) >= 3:
                err = max(0.3, float(np.hypot(*singles.std(axis=0)) / np.sqrt(len(singles))))
        pe, pn = targets.predicted_offsets(t, [f.mjd_mid], config.REFERENCE_MJD)[0]
        k = int(f.label.split(":")[0].split()[-1]) if f.label.startswith("Epoch") else 0
        ra_m, dec_m = (float(v) for v in wgrid.pixel_to_world_values(cx, cy))
        pts.append(dict(ra=ra_m, dec=dec_m, kind=f.kind,
                        label=f.label.split(" · ")[0] if k == 0 else f"Epoch {k}", epoch=k, mjd=f.mjd_mid,
                        date=targets.mjd_to_date(f.mjd_mid), de=de, dn=dn, err=err,
                        off=float(np.hypot(de - pe, dn - pn)), par_e=par[0], par_n=par[1], pred_e=pe, pred_n=pn))
    load.step("Fitting its path and working out what it is…",
              "All positions together: standing still, drifting, wobbling or curving", 5,
              left=max(0.1, usual - (time.time() - t_motion)))
    if len(pts) < 2:
        load.done()
        st.info("The object needs to be visible in at least two photos. Try a band where it is bright, or a "
                "larger field of view.")
        return
    m_lo, m_hi = min(p["mjd"] for p in pts) - 60, max(p["mjd"] for p in pts) + 60
    grid_mjd = np.linspace(m_lo, m_hi, int(min(2000, max(200, (m_hi - m_lo) / 3))))
    tr = targets.predicted_offsets(t, grid_mjd, config.REFERENCE_MJD)
    st.plotly_chart(render.motion_fig(tr[:, 0], tr[:, 1], pts), use_container_width=True)
    pending_notes()
    st.plotly_chart(render.miss_fig(pts), use_container_width=True)
    yrs = [(p["mjd"] - config.REFERENCE_MJD) / 365.25 for p in pts]
    fit = photometry.fit_motion(yrs, [p["de"] - p["par_e"] for p in pts], [p["dn"] - p["par_n"] for p in pts],
                                [p["err"] for p in pts])
    cat_e, cat_n = t.pmra / 1000, t.pmdec / 1000
    has_cat = bool(t.pmra or t.pmdec or t.parallax)
    if fit:
        pe, pn, ee, _ = fit
        mu, cat_mu = np.hypot(pe, pn), np.hypot(cat_e, cat_n)
        diff = np.hypot(pe - cat_e, pn - cat_n)
        c1, c2, c3 = st.columns(3)
        c1.metric("Measured motion", f"{mu:.2f} ± {ee:.2f} ″/yr",
                  help="How fast SPHEREx sees the object drift across the sky, in arcseconds per year: the slope "
                       "of a straight line fitted through the measured positions after removing the parallax "
                       "wobble. ± is the uncertainty (1σ).")
        if has_cat:
            c2.metric("Catalogue motion", f"{cat_mu:.2f} ″/yr",
                      help=f"What the catalogue says ({t.source}): the speed the grey line is drawn with.")
            c3.metric("Difference", f"{diff:.2f} ″/yr", f"{diff / ee:.1f} σ", delta_color="off",
                      help="Measured minus catalogue. σ (sigma) counts how many uncertainties apart they are: "
                           "below about 3σ they agree.")
        if t.has_motion:
            verdict = (f"**Agrees** with the catalogue: SPHEREx sees {t.name} move {mu:.2f}″ per year, "
                       f"the catalogue says {cat_mu:.2f}″." if diff < 3 * ee + 0.1 * cat_mu else
                       f"**Differs** from the catalogue by {diff:.2f}″/yr ({diff / ee:.1f}σ). Saturation "
                       "(an overexposed star), a blended neighbour or too few epochs can cause this; check the blink.")
        else:
            verdict = (f"**Moves**: {mu:.2f} ± {ee:.2f}″/yr, more than 3× the uncertainty." if mu > 3 * ee else
                       f"No motion detected: less than about {3 * ee:.2f}″/yr (3σ).")
        st.markdown(verdict)
    st.caption("**Reading the plot:** each coloured point is where the object's light is centred in one photo "
               "(east to the left, as on the sky), relative to its catalogue position on 2026-01-01 (the 0, 0 "
               "point). The grey line is where it *should* be. Error bars show the measuring uncertainty. "
               "Saturated (overexposed) stars and blended neighbours make centres less reliable.")
    obs = [motion.Obs(mjd=p["mjd"], ra=p["ra"], dec=p["dec"], err=p["err"], label=p["label"]) for p in pts]
    mfit = motion.fit(obs)
    mcls = motion.classify(obs, mfit, gaia=dict(pm=(t.pmra, t.pmdec), who="The catalogue",
                                                 text=f"{t.name} is in the catalogue ({t.source})")
                           if t.source != "coordinates" else None)
    load.progress(0.9, "", left=0.1)
    load.est.learn_step("motion", time.time() - t_motion)
    st.markdown("---")
    st.markdown(f"#### What SPHEREx's own measurements say about {t.name}")
    st.caption("The same analysis as for your own finds: all measured positions fitted together, the kind of object "
               "estimated from the evidence, and the path predicted into the future"
               + (", next to the catalogue's prediction." if has_cat else "."))
    verdict_block(mfit, mcls)
    future_section(mfit, mcls, obs, key=f"cat|{t.name}", cat_target=t if has_cat else None)
    details_block(mfit, obs)

    plx = t.parallax / 1000
    with st.expander("How the prediction (grey line) and the measurement are calculated", expanded=False):
        st.markdown(
            f"**The prediction** starts from the catalogue ({t.source}) and adds two effects:\n\n"
            f"1. **Proper motion** (the star's own drift): **{cat_e:+.3f}″/yr** east and **{cat_n:+.3f}″/yr** north. "
            "After *t* years the star has drifted *t* × that. *t* is counted from 2026-01-01.\n"
            f"2. **Parallax** (the yearly wobble from watching on a moving Earth): **π = {plx:.4f}″**. Earth's "
            "position around the Sun on each date, *X, Y, Z* in AU (Earth–Sun distances), comes from the planetary "
            "ephemeris in astropy. With the star's RA α and Dec δ:\n\n"
            "   - east shift = π · (X sin α − Y cos α)\n"
            "   - north shift = π · (X cos α sin δ + Y sin α sin δ − Z cos δ)\n"
            "3. **Predicted position** = drift + wobble. Repeating this for many dates draws the grey line; for a "
            "nearby star it is a slanted line with loops (the loops are the parallax).\n\n"
            "**The measurement:** in each epoch photo the app looks within 4 display pixels (12″) of the predicted "
            "spot and takes the *centroid*, the brightness-weighted average position of the light, which can be "
            "pinned down to a fraction of a pixel. The error bar is the spread between the single photos of that "
            "epoch divided by √(number of photos). Finally the parallax part is subtracted from every point and "
            "a straight line is fitted through position versus time (weighted least squares): its slope is the "
            "**measured motion** above.\n\n"
            "**Step by step for this object** (all values in arcseconds, ″):")
        df = pd.DataFrame([{
            "Photo": p["label"], "Date": p["date"],
            "Years from 2026-01-01": round((p["mjd"] - config.REFERENCE_MJD) / 365.25, 3),
            "Drift east": round(p["pred_e"] - p["par_e"], 2), "Drift north": round(p["pred_n"] - p["par_n"], 2),
            "Wobble east": round(p["par_e"], 2), "Wobble north": round(p["par_n"], 2),
            "Predicted east": round(p["pred_e"], 2), "Predicted north": round(p["pred_n"], 2),
            "Measured east": round(p["de"], 2), "Measured north": round(p["dn"], 2),
            "± error": round(p["err"], 2), "Miss (measured − predicted)": round(p["off"], 2)} for p in pts])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("Drift = proper motion × years. Wobble = parallax shift. Predicted = drift + wobble. "
                   "Miss = distance between the measured and predicted points.")
        st.download_button("Download this table (CSV)", df.to_csv(index=False),
                           f"motion_{t.name.replace(' ', '_')}_{band}.csv", help="Save the step-by-step numbers.")
    with st.expander("Watch it move: the photos with the predicted position (orange ring) on each date"):
        sky_viewer(frames, key_extra=f"motion|{t.name}", follow=t)
    load.done()   # the whole check is on the page: the bar fills up and goes


def photo_of(m):
    """The photo a mark was made on (epoch stack, older survey or single exposure): by the photo's id, so two
    single exposures taken minutes apart stay two photos; marks saved before ids existed are found by date."""
    uid = m.get("uid") or ""
    if uid:
        for f in frames:
            if viewer.photo_uid(f) == uid:
                return f
        if uid.startswith("exp:") and uid[4:] in by_uri:
            return stack.single_frame(by_uri[uid[4:]], list(by_uri).index(uid[4:]) + 1)
    cands = [(abs(f.mjd_mid - m["mjd"]), k, f) for k, f in enumerate(frames)]
    cands += [(abs(0.5 * (a.stamp.mjd + a.stamp.mjd_end) - m["mjd"]), len(frames) + i, a)
              for i, a in enumerate(by_uri.values())]
    d, k, best = min(cands, key=lambda c: c[:2]) if cands else (1.0, 0, None)
    if best is None or d > 1e-4:   # 9 seconds: closer than any two exposures
        return None
    return best if k < len(frames) else stack.single_frame(best, k - len(frames) + 1)


def where_on(f, m):
    """Where the mark is on its own photo, and its uncertainty in arcsec. A mark stays exactly where it was
    clicked (even on dark sky); only when 'Snap to star centre' was on is it re-centred on the light."""
    if not m.get("snap"):
        r, d = wgrid.pixel_to_world_values(float(m["x"]), float(m["y"]))
        return float(r), float(d), float(arcsec_px), (float(m["x"]), float(m["y"]))   # a click: about one display pixel
    cx, cy = photometry.centroid(f.image, float(m["x"]), float(m["y"]), r=3)
    if not np.isfinite(cx) or np.hypot(cx - m["x"], cy - m["y"]) > 3:
        cx, cy = float(m["x"]), float(m["y"])
    err = 0.6 if f.kind == "survey" else 1.2
    if f.kind == "stack" and len(f.stamps) >= 3:  # scatter of the single exposures of that photo
        cs = [photometry.centroid(by_uri[s.uri].image, cx, cy, r=3) for s in f.stamps if s.uri in by_uri]
        cs = np.array([c for c in cs if np.isfinite(c[0])])
        if len(cs) >= 3:
            err = max(0.3, float(np.hypot(*cs.std(axis=0)) / np.sqrt(len(cs)) * arcsec_px))
    r, d = wgrid.pixel_to_world_values(cx, cy)
    return float(r), float(d), err, (cx, cy)


MODEL_NAME = {"static": "standing still", "linear": "steady drift (proper motion)",
              "linear+parallax": "steady drift + yearly parallax wobble", "quadratic": "curved path"}
MODEL_SHORT = {"static": "still", "linear": "drift", "linear+parallax": "drift + parallax", "quadratic": "curved"}


def _pm(v, e, unit):
    d = 1 if abs(v) >= 1 else 2
    return f"{v:.{d}f} ± {e:.{d}f}{unit}"


def gaia_for(obs):
    """Does one Gaia star sit at every mark (on each mark's date)? (dict | False | None, plain note)."""
    try:
        hits = [catalogs.nearest(_gaia_near(round(o.ra, 3), round(o.dec, 3)), o.ra, o.dec, o.mjd,
                                 config.FIND_MATCH_ARCSEC)[0] for o in obs]
    except Exception:
        return None, "The Gaia catalogue could not be reached right now, so the catalogue check is left out."
    ids = {s["id"] for s in hits if s}
    if not ids:
        return False, f"No Gaia star within {config.FIND_MATCH_ARCSEC:g}″ of any mark (on each mark's date)."
    if len(ids) == 1 and all(hits):
        s = hits[0]
        pm = tuple(0.0 if not np.isfinite(v) else float(v) for v in (s["pmra"], s["pmdec"]))
        return dict(g=s["g"], pm=pm, id=s["id"], star=s), (
            f"Every mark matches the same Gaia DR3 star, {s['id']} (G = {s['g']:.1f}).")
    return None, ("Some marks match Gaia stars and some do not (or different ones), so the catalogue check is "
                  "left out. If the object moves, your marks may be on neighbouring stars.")


def catalogue_offsets(t, fit, mjds):
    """A catalogue target's predicted path (proper motion + parallax) as offsets in the fit's frame (″)."""
    mjds = np.atleast_1d(np.asarray(mjds, float))
    ra, dec = targets.position_at(t, mjds)
    e = ((np.asarray(ra) - fit.ref_ra + 180) % 360 - 180) * np.cos(np.radians(fit.ref_dec)) * 3600
    n = (np.asarray(dec) - fit.ref_dec) * 3600
    if t.parallax > 0:
        fe, fn = motion.parallax_factors(t.ra, t.dec, mjds)
        e, n = e + t.parallax / 1000 * fe, n + t.parallax / 1000 * fn
    return e, n


def verdict_block(fit, cls):
    """What the object probably is, how sure, and why; then the key numbers."""
    best = cls.best
    T = lambda x: i18n.tr(x, ss.lang)  # noqa: E731  (text built from parts: translate each part whole first)
    if cls.prob[best] >= 0.6:
        st.markdown(f"#### What it probably is: {motion.CLASS_NAME[best]} ({100 * cls.prob[best]:.0f}%)")
    else:
        st.markdown(f"#### Not clear yet. Most likely: {motion.CLASS_NAME[best]} ({100 * cls.prob[best]:.0f}%)")
    st.plotly_chart(render.class_fig(cls.prob, {k: T(v) for k, v in motion.CLASS_SHORT.items()}),
                    use_container_width=True)
    lines = [f"- **{T('For' if r['w'] > 0 else 'Against')} “{T(motion.CLASS_SHORT[r['cls']])}”:** {T(r['text'])}"
             for r in cls.reasons]
    st.markdown("**The evidence, step by step**\n\n" + "\n".join(lines))
    if cls.variable:
        st.info(cls.variable)
    st.caption("An algorithmic estimate from the evidence above, not a certainty: always look at the photos too. "
               "The percentages say how strongly the evidence points to each kind of object.")
    c1, c2, c3, c4 = st.columns(4)
    hours = fit.span_days < 5
    if fit.model == "static":
        lim = (f"under {fit.mu_limit / motion.YEAR / 1440:.2f}″/min" if fit.span_days < 1 / 24 else
               f"under {fit.mu_limit / motion.YEAR / 24:.2f}″/h" if hours else f"under {fit.mu_limit:.1f}″/yr")
        c1.metric("Speed", lim, help="No motion was found; it is slower than this (3σ).")
        c2.metric("Direction", "–")
    else:
        sp = (_pm(fit.mu / motion.YEAR / 1440, fit.mu_err / motion.YEAR / 1440, "″/min") if fit.span_days < 1 / 24
              else _pm(fit.mu / motion.YEAR / 24, fit.mu_err / motion.YEAR / 24, "″/h") if hours else
              _pm(fit.mu, fit.mu_err, "″/yr"))
        c1.metric("Speed", sp, help="How fast it moves across the sky, from one path fitted through all the marks "
                                    "at once (not from two marks only). ± is the uncertainty (1σ).")
        pa = float(np.degrees(np.arctan2(fit.mu_e, fit.mu_n)) % 360)
        c2.metric("Direction", f"{pa:.0f}° ({detect.COMPASS[int(round(pa / 45)) % 8]})",
                  help="Direction of motion, in degrees from north towards east.")
    c3.metric("Path model", MODEL_SHORT[fit.model],
              help=f"{MODEL_NAME[fit.model].capitalize()}. The simplest kind of path that the marks do not clearly reject: standing still, steady drift, "
                   "drift plus the yearly parallax wobble, or a curved path.")
    if np.isfinite(fit.plx):
        c4.metric("Distance (from parallax)", f"{1 / fit.plx:.1f} pc",
                  f"π = {fit.plx:.3f} ± {fit.plx_err:.3f}″", delta_color="off",
                  help="Parallax π is the yearly wobble; distance in parsecs = 1 / π (1 pc = 3.26 light-years).")
    elif fit.dof > 0:
        c4.metric("Consistency (χ²ν)", f"{fit.rchi2:.1f}",
                  help="How well the marks follow one path: about 1 = within the measuring errors; much more than "
                       "1 = the marks scatter more than they should (a wrong mark, a blend, or different objects).")
    else:
        c4.metric("Consistency (χ²ν)", "–", help="With two marks any path fits exactly: mark a third photo to test it.")


def future_section(fit, cls, obs, key, cat_target=None):
    """The path through the marks continued into the future, on the sky and against time, with a table."""
    st.markdown("#### Where it goes next: predicted path")
    first, last = float(np.min(fit.mjd)), float(np.max(fit.mjd))
    today = float(Time.now().mjd)
    short = cls.best == "solar_system" or fit.span_days < 5
    if cls.best == "mismatch":
        st.warning("The marks probably do not belong to one object, so the prediction below is not reliable. "
                   "Leave out the mark that does not fit (see *Every mark against the others* below) and try again.")
    if short and fit.span_days < 1 / 24:   # single exposures minutes apart: think in minutes
        ahead = st.slider("Predict ahead (minutes)", 1, 180, 30, key=f"ahead_m|{key}",
                          help="How far past the last mark to continue the path.")
        end = last + ahead / 1440
        step = max(1, ahead // 8)
        tick_mjd = [last + k / 1440 for k in range(step, ahead + 1, step)]
        tick_txt = [f"+{k} min" for k in range(step, ahead + 1, step)]
    elif short:
        ahead = st.slider("Predict ahead (hours)", 1, 72, 24, key=f"ahead_h|{key}",
                          help="How far past the last mark to continue the path.")
        end = last + ahead / 24
        step = max(1, ahead // 8)
        tick_mjd = [last + h / 24 for h in range(step, ahead + 1, step)]
        tick_txt = [f"+{h} h" for h in range(step, ahead + 1, step)]
        st.caption("Asteroids and comets follow curved orbits, so this straight-line guess holds for hours to a few "
                   "days only. To follow a new one for longer, its positions go to the Minor Planet Center, which "
                   "computes an orbit.")
    else:
        years = st.slider("Predict ahead (years)", 1, 50, 10, key=f"ahead_y|{key}",
                          help="How many years past today to continue the path.")
        end = max(last, today) + years * motion.YEAR
        y0 = int(targets.mjd_to_date(last)[:4]) + 1
        y1 = int(targets.mjd_to_date(end)[:4])
        step = max(1, int(np.ceil((y1 - y0 + 1) / 12)))
        yrs = list(range(y0, y1 + 1, step))
        tick_mjd = [float(Time(f"{y}-01-01").mjd) for y in yrs]
        tick_txt = [str(y) for y in yrs]
    path_mjd = np.linspace(first, last, 240)
    fut_mjd = np.unique(np.concatenate([np.linspace(last, end, 480), tick_mjd]))
    ticks = [(int(np.argmin(np.abs(fut_mjd - m))), t) for m, t in zip(tick_mjd, tick_txt)]
    pe, pn, _, _ = fit.predict_offsets(path_mjd)
    fe, fn, fse, fsn = fit.predict_offsets(fut_mjd)
    obs = sorted(obs, key=lambda o: o.mjd)
    marks = [dict(e=float(e), n=float(n), err=float(o.err), label=o.label, date=targets.mjd_to_date(o.mjd),
                  kind=o.kind, datetime=Time(o.mjd, format="mjd").isot) for o, e, n in zip(obs, fit.oe, fit.on)]
    cat = None
    if cat_target is not None and (cat_target.pmra or cat_target.pmdec or cat_target.parallax):
        ce, cn = catalogue_offsets(cat_target, fit, np.concatenate([path_mjd, fut_mjd]))
        cat = dict(e=ce, n=cn)
    st.plotly_chart(render.future_fig(marks, dict(e=pe, n=pn), dict(e=fe, n=fn, se=fse, sn=fsn, ticks=ticks), cat,
                                      "step" if short else "year"), use_container_width=True)
    T = lambda x: i18n.tr(x, ss.lang)  # noqa: E731
    st.caption(T("**Reading the plot:** coloured points are the marks (east to the left, as on the sky), with their "
                 "measuring error. The solid line is the one path fitted through all of them; the dashed line "
                 "continues it into the future, with the date of each predicted point. The shaded band is the "
                 "uncertainty (1σ): it widens the further ahead we look, because a small error in the speed grows "
                 "with time.") + (" " + T("The dotted grey line is the catalogue's own prediction.") if cat else ""))
    lo = first - 0.05 * max(end - first, 1 / 24)
    t_mjd = np.unique(np.concatenate([np.linspace(lo, last, 200), fut_mjd]))
    te, tn, tse, tsn = fit.predict_offsets(t_mjd)
    st.plotly_chart(render.track_time_fig(marks, dict(datetime=list(Time(t_mjd, format="mjd").isot), e=te, n=tn,
                                                      se=tse, sn=tsn, future=t_mjd > last),
                                          Time(today, format="mjd").isot if lo < today < end else None),
                    use_container_width=True)
    st.caption("The same path against time: east and north position separately. A steady drift is a straight line; "
               "the yearly parallax wobble makes it wave; an orbit makes it bend. The dotted line is today.")
    ra, dec, se, sn = fit.predict(np.array(tick_mjd))
    le, ln_, _, _ = fit.predict_offsets([last])
    ke, kn, _, _ = fit.predict_offsets(np.array(tick_mjd))
    rows = []
    for k, m in enumerate(tick_mjd):
        r = {"When": (targets.mjd_to_date(m) if not short else Time(m, format="mjd").iso[:16]) + f" ({tick_txt[k]})",
             "RA (°)": round(float(ra[k]), 6), "Dec (°)": round(float(dec[k]), 6),
             "± east (″)": round(float(se[k]), 2), "± north (″)": round(float(sn[k]), 2),
             "From the last mark (″)": round(float(np.hypot(ke[k] - le[0], kn[k] - ln_[0])), 1)}
        if cat is not None:
            ce1, cn1 = catalogue_offsets(cat_target, fit, [m])
            r["Catalogue minus measured (″)"] = round(float(np.hypot(ce1[0] - ke[k], cn1[0] - kn[k])), 2)
        rows.append(r)
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("Download the prediction (CSV)", df.to_csv(index=False),
                       f"prediction_{key.split('|')[-1].replace(' ', '_')}.csv", key=f"dl_pred|{key}",
                       help="Predicted positions with their uncertainty.")


def details_block(fit, obs):
    """Every pair of marks, every mark against the others, and the models that were tried."""
    obs = sorted(obs, key=lambda o: o.mjd)
    with st.expander("Every pair of marks compared"):
        pred = {}
        for i, o in enumerate(obs):
            e, n, _, _ = fit.predict_offsets([o.mjd])
            pred[i] = (e[0], n[0])
        rows = []
        for p in motion.pairs(obs):
            a, b = obs[p["i"]], obs[p["j"]]
            off = float(np.hypot(p["de"] - (pred[p["j"]][0] - pred[p["i"]][0]),
                                 p["dn"] - (pred[p["j"]][1] - pred[p["i"]][1])))
            days = p["days"]
            rows.append({"Pair": f"{a.label} → {b.label}",
                         "Time apart": (f"{days * 1440:.1f} min" if days < 1 / 24 else f"{days * 24:.1f} h" if days < 2
                                        else f"{days:.0f} days"),
                         "Moved (″)": round(p["dist"], 2), "± (″)": round(p["err"], 2),
                         "Direction": f"{p['way']} ({p['pa']:.0f}°)",
                         "Speed": ((f"{p['speed_hr'] / 60:.2f}″/min" if days < 1 / 24 else
                                    f"{p['speed_hr']:.2f}″/h" if days < 5 else f"{p['speed_yr']:.2f}″/yr")
                                   if days > 0 else "–"),
                         "Real move? (σ)": round(p["sig"], 1),
                         "Off the fitted path (″)": round(off, 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Each pair on its own: how far the object moved between the two photos, in which direction and "
                   "how fast. 'Real move?' counts measuring errors: above 3σ the move is real. 'Off the fitted path' "
                   "compares the pair's move with the path through all marks: pairs that agree have small values.")
    with st.expander("Every mark against the others"):
        if fit.loo:
            rows = []
            for o, x in zip(obs, fit.loo):
                ns = x["nsig"]
                rows.append({"Mark": o.label, "Date": targets.mjd_to_date(o.mjd),
                             "Off the path of the other marks (″)": round(x["miss"], 2),
                             "± (″)": round(x["sig"], 2), "σ": round(ns, 1),
                             "Verdict": "fits" if ns < 3 else "slightly off" if ns < 5 else "does not fit"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Each mark is left out in turn; the other marks draw a path, and the table says how far the "
                       "left-out mark is from it. A mark that 'does not fit' is probably on another object or "
                       "badly placed: leave it out in *Check my find* and analyse again.")
        else:
            st.caption("With two marks, each one is needed to draw the path, so they cannot check each other. "
                       "Mark the object on a third photo to test them.")
    with st.expander("The models that were tried"):
        rows = [{"Path model": MODEL_NAME[k], "χ²": round(v["chi2"], 2), "Degrees of freedom": v["dof"],
                 "Improvement": round(v["gain"], 1), "Chosen": "yes" if v["kept"] else ""}
                for k, v in fit.tried.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("χ² adds up how far each mark is from the path, in measuring errors (squared). A richer model is "
                   "chosen only when it lowers χ² by much more than chance would (improvement > 11.8 for motion, "
                   "> 9 for parallax, > 16 for a curve), after scaling by the scatter that is left.")
    with st.expander("How the motion analysis works"):
        st.markdown(
            "1. **Measure each mark on its own photo.** The app takes the light-weighted centre near your mark "
            "on the very photo you marked (not on any other), so a mark a little off the star does not matter. "
            "Its uncertainty comes from the spread between the single exposures behind that photo.\n"
            "2. **Fit one path through all marks at once** (weighted least squares): standing still, steady drift, "
            "drift plus the yearly parallax wobble (from Earth's position around the Sun on each date), and a curved "
            "path. The simplest model that the marks do not clearly reject is kept.\n"
            "3. **Check the marks against each other:** every pair is compared, and every mark is tested against "
            "the path drawn by the others (leave-one-out), so a mark on the wrong object stands out.\n"
            "4. **Weigh the evidence** for each kind of object: how fast it moves and over how long (no star moves "
            "measurably within days; no star moves more than about 10″ a year; an asteroid leaves a field within "
            "days), whether the path is steady, wobbles or curves, how far it is from the ecliptic (where "
            "asteroids are), whether Gaia knows a star there, whether something stays behind at the old position, "
            "and whether the single exposures confirm it.\n"
            "5. **Predict the future:** the fitted path is continued, and its uncertainty is carried forward from "
            "the fit (the band widens with time).")


def found_rows(fit0, used):
    """The object measured on the photos it was not marked on, at the spot the marks predict (on request)."""
    out = []
    for f in frames:
        if viewer.photo_uid(f) in used:
            continue
        pra, pdec, se, sn = fit0.predict([f.mjd_mid])
        gx, gy = (float(v) for v in wgrid.world_to_pixel_values(pra[0], pdec[0]))
        if not (3 < gx < shape[1] - 4 and 3 < gy < shape[0] - 4):
            continue
        tol = min(6.0, max(1.5, 3 * float(np.hypot(se[0], sn[0])) / arcsec_px))
        cx, cy = photometry.centroid(f.image, gx, gy, r=3)
        if not np.isfinite(cx) or np.hypot(cx - gx, cy - gy) > tol:
            continue
        fl, npx = detect.aperture(f.image, cx, cy)
        nz = detect.noise(f.image)
        if not (np.isfinite(fl) and fl > 5 * detect.OVERSAMPLE_NOISE * np.sqrt(max(npx, 1)) * nz):
            continue
        out.append(dict(m=dict(x=cx, y=cy, mjd=f.mjd_mid, label="", snap=True), f=f))
    return out


def object_check(marks, key="a", title=None, name="Your find"):
    """Motion of one object from every mark the user made of it, each measured on its own photo, fitted together.
    Returns (fit, positions) for a comparison, or None when there are not enough marks."""
    st.subheader(title or f"Motion of your {len(marks)} marks ({band})")
    rows, missing = [], []
    for i, m in enumerate(sorted(marks, key=lambda m: m["mjd"])):
        m = dict(m, label=m.get("label") or f"#{i + 1}")
        f = photo_of(m)
        if f is None:
            missing.append(m["label"])
            continue
        rows.append(dict(m=m, f=f))
    if missing:
        st.warning(f"Marks {', '.join(missing)} were made on photos that are not in this view (another band or "
                   "field?), so they are left out.")
    if len({viewer.photo_uid(r["f"]) for r in rows}) < 2:
        st.warning("At least two of the marks must be on different photos of this view. Go back and mark the "
                   "object on two or more dates.")
        return None

    def measure(rs):
        for r in rs:
            f, m = r["f"], r["m"]
            ra, dec, err, (cx, cy) = where_on(f, m)
            fl, fe = float("nan"), float("nan")
            if f.kind != "survey":   # older surveys are scaled for display: their brightness is not comparable
                fl, npx = detect.aperture(f.image, cx, cy)
                fe = detect.OVERSAMPLE_NOISE * np.sqrt(max(npx, 1)) * detect.noise(f.image)
            singles = [by_uri[s.uri].image for s in f.stamps if s.uri in by_uri] if f.kind == "stack" else None
            r.update(cx=cx, cy=cy, seen=detect.seen_fraction(singles, cx, cy) if singles and len(singles) >= 3
                     else None, obs=motion.Obs(mjd=f.mjd_mid, ra=ra, dec=dec, err=err, label=m["label"], flux=fl,
                                               flux_err=fe, kind=m.get("kind", "mark")))
        return rs

    rows = measure(rows)
    rec = st.toggle("Also measure it on the other photos, at the spot your marks predict", key=f"rec|{tgt.name}|{key}",
                    help="Off by default: the app does not look for things by itself. When on, the path through your "
                         "marks predicts where the object should be on every other photo of this view; where a clear "
                         "spot sits there (5σ, within the prediction's uncertainty), it is measured and added "
                         "(diamonds, F1, F2…). More dates make the path, the parallax and the prediction much better.")
    if rec:
        fit0 = motion.fit([r["obs"] for r in rows])
        if fit0.span_days >= 5:
            extra = found_rows(fit0, {viewer.photo_uid(r["f"]) for r in rows})
            for k, r in enumerate(sorted(extra, key=lambda r: r["f"].mjd_mid)):
                r["m"].update(label=f"F{k + 1}", kind="found")
            rows = sorted(rows + measure(extra), key=lambda r: r["f"].mjd_mid)
            st.caption(f"Found on {len(extra)} more photo(s) at the predicted spot." if extra else
                       "Nothing clear at the predicted spot on the other photos.")
        else:
            st.caption("Your marks are hours apart: the other photos are months away, too far to follow an "
                       "asteroid there.")
    obs = [r["obs"] for r in rows]
    lb = []
    for i, j in combinations(range(len(rows)), 2):
        a, b = rows[i], rows[j]
        if viewer.photo_uid(a["f"]) == viewer.photo_uid(b["f"]) or np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) * arcsec_px < 6:
            continue
        fb, _ = detect.aperture(b["f"].image, b["cx"], b["cy"])
        fab, _ = detect.aperture(b["f"].image, a["cx"], a["cy"])
        if np.isfinite(fb) and np.isfinite(fab) and fb > 0:
            lb.append((i, j, max(0.0, fab / fb)))
    conf = [(i, r["seen"]) for i, r in enumerate(rows) if r["seen"] is not None]
    gaia, gaia_note = gaia_for(obs)
    fit = motion.fit(obs)
    cls = motion.classify(obs, fit, gaia=gaia, left_behind=lb, confirmed=conf)
    days = fit.span_days
    st.caption(f"{len(obs)} positions on {fit.n_dates} photos, spanning "
               + (f"{days * 1440:.1f} minutes" if days < 1 / 24 else f"{days * 24:.1f} hours" if days < 2 else f"{days:.0f} days" if days < 730 else
                  f"{days / 365.25:.1f} years")
               + ". Every mark is measured only on the photo it was made on, then all are fitted together. "
               + i18n.tr(gaia_note, ss.lang))
    verdict_block(fit, cls)
    cat = catalogs.as_target(gaia["star"], name=f"Gaia DR3 {gaia['id']}") if isinstance(gaia, dict) else None
    future_section(fit, cls, obs, key=f"find|{tgt.name}|{key}", cat_target=cat)
    details_block(fit, obs)
    st.dataframe(pd.DataFrame([{"Mark": r["obs"].label, "Photo": r["f"].label,
                                "RA (°)": round(r["obs"].ra, 6), "Dec (°)": round(r["obs"].dec, 6),
                                "± (″)": round(r["obs"].err, 2),
                                "Seen in single exposures": "–" if r["seen"] is None else f"{r['seen'] * 100:.0f}%"}
                               for r in rows]), hide_index=True, use_container_width=True)
    follow = fit.as_target(name)
    with st.expander("Watch it move: every photo with the predicted spot (orange ring)"):
        show = frames if days >= 5 else list({id(r["f"]): r["f"] for r in rows}.values())
        sky_viewer(show, key_extra=f"track|{key}", follow=follow, wkey=None if key == "a" else f"viewer|{ss.mode}|{key}")
        st.caption("The orange ring follows the path fitted through your marks, also on the photos you did not "
                   "mark: if the object sits in the ring there too, the prediction works.")
    return fit, obs


def track_check(marks, marks_b=None):
    """Your marks' motion check; with a second object's marks, both checks and the two compared. The progress bar
    stays up until all of it is worked out."""
    t0 = time.time()
    usual = load.est.usual("motion") * (2 if marks_b else 1)
    load.step("Measuring your marks…", "Each mark on the photo it was made on", 4, left=usual)
    if not marks_b:
        object_check(marks)
        load.done()
        return
    top = st.container()
    ta, tb = st.tabs([f"Object A: its motion check ({len(marks)} marks)",
                      f"Object B: its motion check ({len(marks_b)} marks)"])
    with ta:
        res_a = object_check(marks, "a", f"Object A: {len(marks)} marks ({band})", "Object A")
    load.progress(0.5, "Object A done", left=max(0.1, usual - (time.time() - t0)))
    with tb:
        res_b = object_check(marks_b, "b", f"Object B: {len(marks_b)} marks ({band})", "Object B")
    load.step("Comparing the two objects…", "Their distance on each photo and how they move relative to each other",
              5, left=0.2)
    with top:
        compare_block(res_a, res_b)
    load.est.learn_step("motion", (time.time() - t0) / 2)
    load.done()


def compare_block(res_a, res_b):
    """Object B against object A: distance, relative motion, apart or closer, and where it goes."""
    st.subheader("Object A and object B compared")
    if res_a is None or res_b is None:
        st.info("Both objects need marks on at least two different photos of this view to be compared.")
        return
    (fa, oa), (fb, ob) = res_a, res_b
    c = motion.compare(fa, fb, oa, ob)
    u = lambda x: f"{x / c.per:.2f}{c.unit}"   # noqa: E731
    m1, m2, m3 = st.columns(3)
    m1.metric("Distance at the last mark", f"{c.sep_now:.1f}″",
              help="How far apart A and B are on the sky at the last date you marked, along their fitted paths. "
                   "1″ is an arcsecond; a SPHEREx pixel is 6.15″.")
    m2.metric("B's speed relative to A", u(c.rel), f"± {u(c.rel_err)} · toward the {c.way}", delta_color="off",
              help="B's motion minus A's motion: how fast B moves as seen from A. The ± is the uncertainty (1σ).")
    m3.metric("Drifting apart" if c.rate >= 0 else "Closing in", u(abs(c.rate)),
              f"{abs(c.rate) / c.rate_err:.1f} σ" if c.rate_err > 0 and np.isfinite(c.rate_err) else None,
              delta_color="off",
              help="How fast their distance grows (apart) or shrinks (closing in). Below about 3σ the change is "
                   "within the measuring error.")
    st.markdown("\n".join(f"- {i18n.tr(s, ss.lang)}" for s in motion.comparison_text(c)))
    iso = lambda m: Time(m, format="mjd").iso[:19]   # noqa: E731
    path = dict(datetime=[iso(m) for m in c.path["mjd"]], sep=c.path["sep"], se=c.path["se"],
                future=c.path["future"])
    pts = [dict(datetime=iso(s["mjd"]), sep=s["sep"], err=s["err"], label=f"A {s['la']} · B {s['lb']}")
           for s in c.shared]
    st.plotly_chart(render.separation_fig(path, pts, today=iso(c.t_now)), use_container_width=True)
    st.caption("The line is the distance between the two fitted paths (dashed: predicted after your last mark, with "
               "its uncertainty); the white points are the distances measured on the photos where you marked both.")
    if c.shared:
        st.dataframe(pd.DataFrame([{"Photo date": iso(s["mjd"]), "Mark of A": s["la"], "Mark of B": s["lb"],
                                    "B east of A (″)": round(s["de"], 2), "B north of A (″)": round(s["dn"], 2),
                                    "Distance (″)": round(s["sep"], 2), "± (″)": round(s["err"], 2),
                                    "B is to the": s["way"]} for s in c.shared]),
                     hide_index=True, use_container_width=True)


if ss.mode == "motion":
    f_ = ss.get("find") or {}
    if tgt.source == "coordinates" and not (f_.get("field") == tgt.name and f_.get("marks")):
        load_secs = load.done()
    else:   # the bar stays up through the analysis: motion_check / track_check close it
        load_secs = time.time() - load.t0
    prefetch.start(tgt, ra0, dec0, ss.fov, band, exps, survey=fut_survey is not None)   # the other bands, ahead
    if f_.get("field") == tgt.name and f_.get("marks"):
        st.button(f"Back to {tgt.name}" if tgt.source != "coordinates" else "Back", on_click=_drop_find,
                  help="Forget these marks and show the usual motion check.")
        track_check(f_["marks"], f_.get("marks_b"))
        st.stop()
    if tgt.source == "coordinates":
        st.subheader(f"Motion check at {tgt.name} ({band})")
        st.info("A motion check compares where a **known object** is predicted to be with where SPHEREx sees it. "
                "You typed a sky position, so nothing is followed here.\n\n"
                "**Mark it yourself:** choose *Mark*, click an object on the first date, switch to a later "
                "date with Next or the timeline, and click the *same* object again. The panel shows how far it moved "
                "and its speed in ″ per year. Then **Check my find** below compares it with the Gaia catalogue.")
        check_my_find(sky_viewer(frames, key_extra="motion"), "motion")
        pending_notes()
        st.caption(f"{len(stamps)} photos loaded in {load_secs:.1f} s. Stars move very little: anything below "
                   "about 3″ (half a SPHEREx pixel) between dates is within the measuring error.")
        st.stop()
    motion_check(tgt, f"{len(stamps)} photos loaded in {load_secs:.1f} s")
    st.stop()

load_secs = load.done()
prefetch.start(tgt, ra0, dec0, ss.fov, band, exps, survey=fut_survey is not None)   # the other bands, ahead
if tgt.source == "coordinates":
    st.caption(COORD_NOTE)


def compose(mode_, frames_x, by_uri_x, view_):
    """What the viewer shows in this mode from one band's frames: (frames shown, diverging colours).
    view_: the page's choices (A/B and the noise cut for Subtract, order and hiding for Every exposure)."""
    if mode_ == "sub" and len(frames_x) >= 2:
        diff = stack.subtract(frames_x[view_["ib"]], frames_x[view_["ia"]])
        if view_["cut"] > 0:
            import dataclasses
            nz = detect.noise(diff.image)
            diff = dataclasses.replace(diff, image=np.where(np.abs(diff.image) < view_["cut"] * nz, 0.0, diff.image))
        return [diff], True
    if mode_ == "movie":
        al_ = sorted(by_uri_x.values(), key=lambda a: a.stamp.mjd if view_["order"] == "time" else a.stamp.wavelength)
        if view_["hide"]:
            al_ = [a for a in al_ if not a.bad_near_target] or al_
        return [stack.single_frame(a, i + 1) for i, a in enumerate(archive.evenly(al_, config.MOVIE_CAP))], False
    return frames_x, False


def band_alts(mode_, view_, labels_ab_, helpers_on):
    """The other bands, lined up in the background, handed to the browser in this view's settings: choosing one in
    the sidebar then shows it at once (viewer.BAND_SCRIPT), and the page catches up a moment later."""
    if helpers_on or fut_survey is not None or mode_ not in ("blink", "sub", "movie"):
        return   # helpers and the older surveys are made by the page itself: those band changes go the usual way
    vkey = json.dumps([tgt.name, ss.fov, mode_, view_, len(gaia_stars), len(exps)])
    memo = ss.setdefault("_alts", {})
    if memo.get("vkey") != vkey:
        memo.clear()
        memo["vkey"] = vkey
    others = [b for b in config.BANDS if b != band]

    def alt(b):
        if b in memo or ss.get(f"all|{tgt.name}|{b}"):   # 'Load all' in that band: not what was loaded ahead
            return memo.get(b)
        r = prefetch.ready(tgt.name, ra0, dec0, ss.fov, b, exps)
        if r is None:
            return None
        by_uri_b, frames_b = r[0], list(r[1])
        v = dict(view_)
        if mode_ == "sub" and len(frames_b) >= 2 and [f.label.split(":")[0] for f in frames_b] != labels_ab_:
            v["ia"], v["ib"] = 0, len(frames_b) - 1   # another list of dates: the A/B boxes start at their defaults
        if mode_ == "movie" and v["hide"]:            # the same switch only when it has the same counts
            v["hide"] = [sum(bool(a.bad_near_target) for a in by_uri_b.values()), len(by_uri_b)] == v["counts"]
        shown_b, div_b = compose(mode_, frames_b, by_uri_b, v)
        data, store, a = view_data(shown_b, b, len(frames_b), len(frames_b), div_b, key_extra=mode_)
        html = viewer.alt_html(vkey, f"{tgt.name}|{b}|{ss.fov}|{mode_}", dict(data=data, store_key=store,
                                                                               settings=dict(a=a)))
        memo[b] = html if len(html) < 3_000_000 else ""   # very long movies of big fields: the usual way
        return memo[b]

    @st.fragment(run_every=config.REFRESH_EVERY)
    def deliver():   # a band lined up in the background reaches the browser within a second
        for b in others:
            st.html(alt(b) or "<span></span>", unsafe_allow_javascript=True)

    with st.sidebar, st.container(key="spxalts"):
        deliver()


tab_view, tab_exp = st.tabs(["Sky viewer", "Photo list"])
with tab_view:
    mode = ss.mode
    view = dict(ia=0, ib=0, cut=0.0, order="time", hide=False, counts=None)   # the choices this view is made with
    labels_ab = None
    if mode == "sub" and len(frames) < 2:
        st.info("Subtract needs photos from at least two dates (epochs) in this band.")
    elif mode == "sub":
        labels_ab = [f.label.split(":")[0] for f in frames]
        c1, c2 = st.columns(2)
        view["ia"] = ia = c1.selectbox("Older photo (A)", range(len(frames)), index=len(frames) - len(sphx_frames),
                                       format_func=lambda i: labels_ab[i], help="The 'before' picture.")
        view["ib"] = ib = c2.selectbox("Newer photo (B)", range(len(frames)), index=len(frames) - 1,
                                       format_func=lambda i: labels_ab[i],
                                       help="The 'after' picture. The view shows B − A.")
        view["cut"] = st.slider("Hide changes smaller than (× noise)", 0.0, 6.0, 0.0, 0.5,
                                help="Sets every pixel whose change is smaller than this many noise levels (σ) to "
                                     "'no change' (white), so only clear changes stay coloured. 0 shows everything.")
        with st.expander("How to read the colours"):
            st.markdown("**Red** = brighter now (in B) · **white / grey** = no change (steady stars cancel out) · "
                        "**blue** = was brighter before (in A).\n\nA **moving star** leaves a blue + red pair — blue "
                        "where it *was*, red where it *is now*. A star that **changed brightness** leaves a single red "
                        "(brightened) or blue (faded) spot. Small rings around very bright stars are leftovers of "
                        "imperfect cancelling, not real changes. The stronger the colour, the bigger the change.")
    elif mode == "movie":
        view["order"] = st.radio("Order", ["time", "wavelength"], horizontal=True,
                                 help="time: oldest to newest. wavelength: from the bluest to the reddest infrared "
                                      "colour inside the band.")
        al = list(by_uri.values())
        nflag = sum(bool(a.bad_near_target) for a in al)
        view["counts"] = [nflag, len(al)]
        if nflag == len(al):
            st.caption(f"All {nflag} photos have warnings at the target (it is probably overexposed in this band), "
                       "so none are hidden. Try another band.")
        elif nflag:
            view["hide"] = st.toggle(f"Hide photos with warnings at the target ({nflag} of {len(al)})", value=False,
                                     help="Leaves out single photos whose pixels at the target were marked unreliable "
                                          "by the NASA pipeline (overexposed, cosmic-ray hits…).")
    shown, diverging = compose(mode, frames, by_uri, view)

    # Optional helpers, off until switched on (the user finds things; the tools measure them). Their
    # switches sit below the viewer; their values are known here already, so the viewer can draw results.
    ss.setdefault("h_sigma", 5.0)
    labels = [f.label.split(" · ")[0] for f in frames]
    h_stars, h_change = ss.get("h_stars", False), ss.get("h_change", False) and mode in ("blink", "sub")
    if mode == "sub" and len(frames) >= 2:
        pair = (ia, ib)
    else:
        for k_, d_ in (("h_a", len(frames) - len(sphx_frames)), ("h_b", len(frames) - 1)):
            if not 0 <= ss.get(k_, -1) < len(frames):
                ss[k_] = max(0, d_)
        pair = (ss.h_a, ss.h_b)
    def singles_of(f):  # the aligned single exposures behind an epoch photo: each star must be in most of them
        return [by_uri[x.uri].image for x in f.stamps if x.uri in by_uri] if f.kind == "stack" else None
    found = [detect.find_stars(f.image, ss.h_sigma, singles=singles_of(f)) for f in shown] if h_stars else None
    changed = []
    if h_change and len(frames) >= 2 and pair[0] != pair[1]:
        changed = detect.changes(frames[pair[0]].image, frames[pair[1]].image, ss.h_sigma,
                                 singles_a=singles_of(frames[pair[0]]), singles_b=singles_of(frames[pair[1]]))[:40]
        yrs = abs(frames[pair[1]].mjd_mid - frames[pair[0]].mjd_mid) / 365.25
        for i, c in enumerate(changed):
            c["label"] = f"C{i + 1}"
            if c["kind"] == "moved":  # in arcseconds, and per year, like the marking tool
                arc = c["dist_px"] * arcsec_px
                c["text"] = (f"moved {arc:.1f}″ toward the {detect.COMPASS[int(round(c['pa_deg'] / 45)) % 8]}"
                             + (f" ({arc / yrs:.1f}″ per year)" if yrs > 0.05 else ""))
    my_marks = sky_viewer(shown, diverging, key_extra=mode, stars=found, changes=changed)
    band_alts(mode, view, labels_ab, h_stars or h_change)
    pending_notes()
    check_my_find(my_marks, mode)

    with st.expander("Helpers — off until you switch them on", expanded=bool(h_stars or h_change)):
        st.caption("The app does not point things out by itself: look first, then let a helper check what you saw.")
        h1, h2, h3 = st.columns(3)
        h1.toggle("Detect stars", key="h_stars",
                  help="Finds spots well above the noise on each photo and draws small green crosses on them. "
                       "Faint or crowded stars can be missed: one SPHEREx pixel is 6.15″ wide.")
        if mode in ("blink", "sub"):
            h2.toggle("Find what changed", key="h_change",
                      help="Compares two dates and lists stars that moved, brightened or faded, strongest first. "
                           "Labels C1, C2… appear on the image.")
        h3.slider("Detection threshold (σ)", 3.0, 10.0, key="h_sigma", step=0.5,
                  help="How far above the noise a spot must be. Higher = fewer but surer detections. "
                       "σ (sigma) is the noise level of the photo.")
        if h_stars and found:
            st.caption(f"Detected stars: {', '.join(f'{len(s)}' for s in found)} (per photo, above {ss.h_sigma:g}σ).")
        if h_change:
            if mode == "blink":
                p1, p2 = st.columns(2)
                p1.selectbox("Compare from (older)", range(len(frames)), key="h_a", format_func=lambda i: labels[i],
                             help="The 'before' photo for Find what changed.")
                p2.selectbox("…to (newer)", range(len(frames)), key="h_b", format_func=lambda i: labels[i],
                             help="The 'after' photo.")
            if any(frames[i].kind == "survey" for i in pair):
                st.caption("Note: one photo is from an older survey (another telescope, scaled for display): only "
                           "*moved* results mean something; brightness changes between telescopes do not.")
            if changed:
                rows = []
                for c in changed:
                    r1, d1 = wgrid.pixel_to_world_values(c["x"], c["y"])
                    rows.append({"label": c["label"], "what": c["kind"], "details": c["text"],
                                 "RA (°)": round(float(r1), 5), "Dec (°)": round(float(d1), 5),
                                 "strength (σ)": round(c["score"], 1)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                             column_config={"strength (σ)": st.column_config.NumberColumn(
                                 help="How many noise levels the change is; above ~5 it is very likely real")})
                st.caption(f"{labels[pair[0]]} → {labels[pair[1]]}. Check each one by eye: rings around very bright "
                           "stars and stars at the edge can be false alarms.")
            elif pair[0] == pair[1]:
                st.info("Pick two different photos to compare.")
            else:
                st.info(f"No change above {ss.h_sigma:g}σ between {labels[pair[0]]} and {labels[pair[1]]}.")
        st.caption("*Note: the helpers are experimental. They can miss things or report false alarms, so always check "
                   "their results by eye.*")

    c1, c2, c3 = st.columns([2, 2, 3])
    if not load_all and n_band > len(todo):
        c1.button(f"Load all {band} photos ({min(n_band, config.LOAD_ALL_CAP)} of {n_band})",
                  on_click=lambda: ss.__setitem__(load_key, True),
                  help="The quick look averages a few photos per date. Loading all gives a cleaner, deeper image "
                       "but takes longer.")
    _gif_args = (shown, ss.stretch, diverging, ss.speed)  # built only when the button is pressed
    c2.download_button("Download GIF", lambda: render.to_gif(*_gif_args),
                       file_name=f"spherex_{tgt.name.replace(' ', '_')}_{band}_{mode}.gif", mime="image/gif",
                       help="Save the animation as a GIF to share.")
    with c3.expander("What am I looking at?"):
        st.markdown(
            "Each frame is a SPHEREx photo of the same patch of sky, taken months apart and averaged per "
            "*epoch* (one pass of the survey). North is up, east is left; the white bar is 1 arcminute (1′). "
            + ("The small orange cross marks the position you typed. "
               if tgt.source == "coordinates" else
               "The orange ring is where the target should be on that date, from its catalogue proper motion. ")
            + "Blue circles are Gaia stars. A star that moves (like Barnard's Star) shifts against the "
            "background; a variable star changes brightness. *Compare* in the viewer puts two dates side by "
            "side, under a swipe divider or subtracted. Motion between epochs is often smaller than one SPHEREx "
            "pixel (6.15″), so the image is shown 2× finer than the camera's pixels. Your marks stay in your "
            "browser (remembered on this computer).")

with tab_exp:
    rows = [dict(date=targets.mjd_to_date(a.stamp.mjd), mjd=round(a.stamp.mjd, 4), band=a.stamp.band,
                 wavelength_um=round(a.stamp.wavelength, 4), flags_at_target=", ".join(a.bad_near_target),
                 file=a.stamp.uri and f"{config.IRSA}/{a.stamp.uri}") for a in by_uri.values()]
    st.dataframe(pd.DataFrame(rows).sort_values("mjd"), hide_index=True, use_container_width=True,
                 column_config={
                     "date": st.column_config.TextColumn("date", help="Day the photo was taken"),
                     "mjd": st.column_config.NumberColumn("MJD", help="Modified Julian Date: astronomers' day count"),
                     "wavelength_um": st.column_config.NumberColumn("wavelength (µm)",
                                                                    help="Infrared colour at the target in this photo"),
                     "flags_at_target": st.column_config.TextColumn("warnings at target",
                                                                    help="Pipeline flags on the pixels at the target "
                                                                         "(e.g. overexposed)"),
                     "file": st.column_config.LinkColumn("file", display_text="FITS",
                                                         help="The original data file at NASA/IPAC")})
    st.caption(f"{len(exps)} SPHEREx photos cover this position in all bands; {n_band} in {band}.")

last, versions = archive.data_through(exps)
where = (f"Position RA {ra0:.5f}, Dec {dec0:+.5f}. " if tgt.source == "coordinates" else
         f"{tgt.name}: RA {ra0:.5f}, Dec {dec0:+.5f} on 2026-01-01 ({tgt.source}). ")
st.caption(where +
           f"SPHEREx data through {targets.mjd_to_date(last)} · processing {', '.join(versions)} · "
           f"{gaia_note} · loaded in {load_secs:.1f} s · "
           "Data: NASA/IPAC Infrared Science Archive (IRSA, via its AWS copy), ESA Gaia.")
