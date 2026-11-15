"""English / Turkish.

The app is written in English. When the user picks Turkish, every text Streamlit shows (labels, help
tips, captions, messages, table headers, chart titles) goes through tr() on its way to the screen, so
the app code does not change. tr() looks the text up:

1. exactly (surrounding spaces ignored);
2. as a template: '{}' stands for any text (numbers, names); '{0}', '{1}' reorder them in Turkish.
   What a placeholder caught is translated too when it is a known word ('north' -> 'kuzey');
3. piece by piece: line by line (keeping list bullets), then sentence by sentence, then by '; ' and ' · '.
   Text the app builds from parts is still translated when each part is known.

Anything unknown stays English. tests/test_i18n.py checks the main screens are covered.
"""
from __future__ import annotations

import re

LANGS = {"en": "English", "tr": "Türkçe"}

TR: dict[str, str] = {
    # --- views ------------------------------------------------------------------------------
    "Blink": "Blink", "Subtract": "Çıkar", "Every exposure": "Her poz", "Spectrum": "Tayf",
    "Motion check": "Hareket kontrolü",
    "flip between dates: movers jump": "tarihler arasında geçiş: hareket edenler sıçrar",
    "newer − older: see only what changed": "yeni − eski: yalnızca değişeni gör",
    "one frame per single photo": "her tek fotoğraf için bir kare",
    "brightness at each infrared colour": "her kızılötesi renkte parlaklık",
    "does it move as predicted?": "öngörüldüğü gibi hareket ediyor mu?",
    "How to look at the photos:": "Fotoğraflara nasıl bakılır:",
    "**{}**: {}": "**{}**: {}",
    "Fastest-moving star in the sky (10″ a year)": "Gökyüzündeki en hızlı yıldız (yılda 10″)",
    "Pair of cold brown dwarfs, 12 light-years away": "12 ışık yılı uzaklıkta iki soğuk kahverengi cüce",
    "Coldest known brown dwarf (about as cold as a freezer); fast mover":
        "Bilinen en soğuk kahverengi cüce (bir dondurucu kadar soğuk); hızlı hareket eder",
    "Young star that dims and flares unpredictably": "Beklenmedik biçimde sönükleşip parlayan genç yıldız",
    "Pulsating giant star (brightens and fades over ~8 months)":
        "Zonklayan dev yıldız (~8 ayda parlayıp söner)",
    # --- sidebar ----------------------------------------------------------------------------
    "Language / Dil": "Language / Dil",
    "SPHEREx Blink": "SPHEREx Blink",
    "**① What to look at**": "**① Neye bakılacak**",
    "Object name or sky position (RA Dec in degrees)": "Cisim adı veya gökyüzü konumu (derece cinsinden RA Dec)",
    "Type the name of a star, galaxy or nebula (looked up in the SIMBAD catalogue), or a sky position as two "
    "numbers in degrees: RA (0–360) then Dec (−90 to +90), for example `269.452 4.693` or `269.452°, +4.693°`. "
    "Hours-minutes-seconds also work: `17 57 48.5 +04 41 36`. Press Enter or Go. Type `/RTP` (random teleport) to jump "
    "to a random spot on the sky.":
        "Bir yıldız, gökada veya bulutsu adı yazın (SIMBAD kataloğunda aranır) ya da derece cinsinden iki sayı "
        "olarak bir gökyüzü konumu: önce RA (0–360), sonra Dec (−90 ile +90 arası), örneğin `269.452 4.693` veya "
        "`269.452°, +4.693°`. Saat-dakika-saniye de olur: `17 57 48.5 +04 41 36`. Enter'a veya Git'e basın. "
        "Gökyüzünde rastgele bir noktaya gitmek için `/RTP` (rastgele ışınlanma) yazın.",
    "Go": "Git", "Load this object or position": "Bu cismi veya konumu yükle",
    "A challenge target (built-in catalogue values)": "Örnek hedef (yerleşik katalog değerleri)",
    "Read as a sky position: RA {}°, Dec {}°": "Gökyüzü konumu olarak okundu: RA {}°, Dec {}°",
    "Warning: this looks like numbers but is not a valid position. Use two numbers in degrees: RA from 0 to 360, "
    "then Dec from −90 to +90, e.g. `269.452 4.693`.":
        "Uyarı: bu sayıya benziyor ama geçerli bir konum değil. Derece cinsinden iki sayı kullanın: 0–360 arası RA, "
        "sonra −90 ile +90 arası Dec, örn. `269.452 4.693`.",
    "Will be looked up by name in the SIMBAD star catalogue": "SIMBAD yıldız kataloğunda adıyla aranacak",
    "…or pick a challenge target:": "…ya da örnek bir hedef seçin:",
    "**② Colour and size**": "**② Renk ve boyut**",
    "Band (which infrared colour)": "Bant (hangi kızılötesi renk)",
    "SPHEREx splits infrared light into six bands: D1 has the shortest wavelengths (just beyond red light), D6 the "
    "longest (sees cool dust and cold brown dwarfs). µm = micrometre.":
        "SPHEREx kızılötesi ışığı altı banda ayırır: D1 en kısa dalga boylarına (kırmızının hemen ötesi), D6 en "
        "uzunlarına sahiptir (soğuk tozu ve soğuk kahverengi cüceleri görür). µm = mikrometre.",
    "Field of view (how much sky)": "Görüş alanı (ne kadar gökyüzü)",
    "Width of the picture in arcminutes (′). 60′ = 1°; the full Moon is about 30′ wide. Bigger fields take a little "
    "longer to load.":
        "Resmin yay dakikası (′) cinsinden genişliği. 60′ = 1°; dolunay yaklaşık 30′ genişliğindedir. Daha geniş "
        "alanların yüklenmesi biraz daha uzun sürer.",
    "**③ Extras**": "**③ Ekstralar**",
    "Load the Gaia star list": "Gaia yıldız listesini yükle",
    "Fetches the stars catalogued by ESA's Gaia satellite. They stay hidden until you tick *Gaia stars* in the "
    "viewer; then each is drawn as a blue circle where it should be on that date. Hover a circle for its details.":
        "ESA'nın Gaia uydusunun kataloğundaki yıldızları getirir. Görüntüleyicide *Gaia yıldızları* kutusunu "
        "işaretleyene kadar gizli kalırlar; sonra her biri o tarihte olması gereken yerde mavi bir halka olarak "
        "çizilir. Ayrıntılar için halkanın üzerine gelin.",
    "Include older photos (2MASS ~2000, WISE 2010)": "Eski fotoğrafları ekle (2MASS ~2000, WISE 2010)",
    "Adds images from earlier infrared surveys before the SPHEREx frames, so motion over 15–25 years becomes "
    "visible. Other telescopes: J → D1/D2, Ks → D3, W1 → D4, W2 → D5/D6.":
        "SPHEREx karelerinin önüne eski kızılötesi taramaların görüntülerini ekler; böylece 15–25 yıllık hareket "
        "görünür hâle gelir. Diğer teleskoplar: J → D1/D2, Ks → D3, W1 → D4, W2 → D5/D6.",
    "Brightness, contrast, speed and flagged pixels are set in the viewer itself.":
        "Parlaklık, kontrast, hız ve işaretli pikseller görüntüleyicinin içinden ayarlanır.",
    "Words used here": "Burada kullanılan terimler",
    # --- page top -----------------------------------------------------------------------------
    "SPHEREx Blink Viewer": "SPHEREx Blink Görüntüleyici",
    "SPHEREx photographs the whole sky every six months in 102 infrared colours. Flip between its photos of one "
    "spot: whatever moves or changes brightness stands out.":
        "SPHEREx tüm gökyüzünü altı ayda bir 102 kızılötesi renkte fotoğraflar. Aynı noktanın fotoğrafları "
        "arasında geçiş yapın: hareket eden veya parlaklığı değişen her şey göze çarpar.",
    "How to use this page (30 seconds)": "Bu sayfa nasıl kullanılır (30 saniye)",
    "1. **Choose a target** in the sidebar: a challenge target, any name (e.g. *M 42*), or a sky position in degrees "
    "(e.g. `269.452 4.693`). Press **Go**.\n2. **Press Play** under the image. The photos are months apart: a star "
    "that *jumps* is moving, one that *flickers* changes brightness.\n3. **Switch the view** with the buttons above: "
    "*Subtract* shows only what changed, *Spectrum* shows colour, *Motion check* compares the measured path with the "
    "predicted one.\n4. **Mark** (in the viewer's toolbar): click to mark objects, draw lines to measure sizes and "
    "distances, and mark the same object on two dates to get its speed. *Compare* puts two dates side by side, under "
    "a swipe divider, or subtracts them.\n\nHover over the small question-mark icons for short explanations; *Words "
    "used here* in the sidebar explains the astronomy terms.":
        "1. Kenar çubuğundan **bir hedef seçin**: örnek bir hedef, herhangi bir ad (örn. *M 42*) ya da derece "
        "cinsinden bir gökyüzü konumu (örn. `269.452 4.693`). **Git**'e basın.\n2. Resmin altındaki **Play**'e "
        "basın. Fotoğraflar aylarca arayla çekilmiştir: *sıçrayan* bir yıldız hareket ediyordur, *titreyen* bir "
        "yıldızın parlaklığı değişiyordur.\n3. Yukarıdaki düğmelerle **görünümü değiştirin**: *Çıkar* yalnızca "
        "değişeni, *Tayf* rengi gösterir, *Hareket kontrolü* ölçülen yolu öngörülenle karşılaştırır.\n4. **İşaretle** "
        "(görüntüleyicinin araç çubuğunda): cisimleri işaretlemek için tıklayın, boyut ve uzaklık ölçmek için çizgi "
        "çizin, aynı cismi iki tarihte işaretleyerek hızını bulun. *Karşılaştır* iki tarihi yan yana, kaydırmalı bir "
        "ayırıcının altında ya da birbirinden çıkarılmış gösterir.\n\nKısa açıklamalar için küçük soru işaretlerinin "
        "üzerine gelin; kenar çubuğundaki *Burada kullanılan terimler* astronomi terimlerini açıklar.",
    # --- loading ------------------------------------------------------------------------------
    "Find target": "Hedefi bul", "Search NASA archive": "NASA arşivinde ara", "Download photos": "Fotoğrafları indir",
    "Line up & average": "Hizala ve ortala",
    "Finding your target…": "Hedefiniz bulunuyor…",
    "Reading the position you typed, or looking the name up in the SIMBAD catalogue":
        "Yazdığınız konum okunuyor ya da ad SIMBAD kataloğunda aranıyor",
    "Asking NASA's archive which photos cover this spot…": "NASA arşivine bu noktayı hangi fotoğrafların kapsadığı soruluyor…",
    "SPHEREx has photographed every part of the sky several times since 2025":
        "SPHEREx 2025'ten beri gökyüzünün her yerini birkaç kez fotoğrafladı",
    "Flying in: downloading {} SPHEREx photos ({})…": "Yaklaşıyoruz: {} SPHEREx fotoğrafı indiriliyor ({})…",
    "Reading only the pixels around your target from NASA's archive":
        "NASA arşivinden yalnızca hedefinizin çevresindeki pikseller okunuyor",
    "Lining the photos up…": "Fotoğraflar hizalanıyor…",
    "Turning every photo north-up on one grid and averaging each epoch":
        "Her fotoğraf kuzey yukarıda olacak şekilde tek bir ızgaraya oturtuluyor ve her dönem ortalanıyor",
    "Adding the older WISE and 2MASS photos…": "Eski WISE ve 2MASS fotoğrafları ekleniyor…",
    "From 2010 and around 2000, to compare with today": "Bugünle karşılaştırmak için 2010'dan ve 2000 civarından",
    "{} of {} photos · {} s": "{} / {} fotoğraf · {} sn",
    "{} s": "{} sn",
    "NASA's servers or the connection are slow right now: it keeps trying, and whatever arrives is saved, so the next "
    "visit is fast": "NASA sunucuları veya bağlantı şu anda yavaş: denemeye devam ediyor; gelen her şey kaydediliyor, "
                     "böylece bir sonraki ziyaret hızlı olur",
    "{} photos": "{} fotoğraf", "{} photos ({} failed)": "{} fotoğraf ({} başarısız)",
    # --- errors -------------------------------------------------------------------------------
    "**{}** is not a valid sky position. Use two numbers in degrees: RA from 0 to 360, then Dec from −90 to +90, for "
    "example `269.452 4.693` or `269.452, -4.693`.":
        "**{}** geçerli bir gökyüzü konumu değil. Derece cinsinden iki sayı kullanın: 0–360 arası RA, sonra −90 ile "
        "+90 arası Dec, örneğin `269.452 4.693` veya `269.452, -4.693`.",
    "Could not find **{}** in the SIMBAD catalogue. Check the spelling, try another name (e.g. *M 42*, *HD 209458*), "
    "or type a position like `269.452 4.693`.":
        "**{}** SIMBAD kataloğunda bulunamadı. Yazımı kontrol edin, başka bir ad deneyin (örn. *M 42*, *HD 209458*) "
        "ya da `269.452 4.693` gibi bir konum yazın.",
    "Name lookup failed ({}). Coordinates like `269.452 4.693` always work.":
        "Ad araması başarısız oldu ({}). `269.452 4.693` gibi koordinatlar her zaman çalışır.",
    "The NASA/IPAC archive is busy right now. Please try again in a minute.":
        "NASA/IPAC arşivi şu anda meşgul. Lütfen bir dakika sonra yeniden deneyin.",
    "SPHEREx has no images covering {} yet. New data arrive weekly.":
        "SPHEREx'in henüz {} bölgesini kapsayan görüntüsü yok. Yeni veriler her hafta gelir.",
    "No photos could be downloaded (the archive or your connection is busy). Please try again in a minute.":
        "Hiç fotoğraf indirilemedi (arşiv veya bağlantınız meşgul). Lütfen bir dakika sonra yeniden deneyin.",
    "Older survey images unavailable ({}).": "Eski tarama görüntüleri kullanılamıyor ({}).",
    # --- Check my find -----------------------------------------------------------------------
    "The Gaia catalogue could not be reached right now.": "Gaia kataloğuna şu anda ulaşılamadı.",
    "Your mark ({}) at RA {}°, Dec {}° matches Gaia DR3 {} (G = {}), {}″ away on that date.":
        "İşaretiniz ({}), RA {}°, Dec {}°, Gaia DR3 {} (G = {}) ile eşleşiyor; o tarihte {}″ uzakta.",
    "Your find": "Bulduğunuz cisim", "Your find (Gaia DR3 {})": "Bulduğunuz cisim (Gaia DR3 {})",
    "your mark (no catalogue match)": "işaretiniz (katalogda eşleşme yok)",
    "No Gaia star within {}″ of your mark (RA {}°, Dec {}°), so there is no prediction: it is treated as standing "
    "still at the spot you marked, and what SPHEREx measures is its own.":
        "İşaretinizin {}″ yakınında Gaia yıldızı yok (RA {}°, Dec {}°), bu yüzden öngörü yok: cisim işaretlediğiniz "
        "yerde duruyor kabul edilir ve SPHEREx'in ölçtüğü yalnızca kendi ölçümüdür.",
    "Very cold or very faint objects are often missing from Gaia.":
        "Çok soğuk veya çok sönük cisimler çoğu zaman Gaia'da yoktur.",
    "**Check my find**": "**Bulduğumu kontrol et**",
    "Mark the same object on two different photos (two dates) to measure how it moved between them. Only those two "
    "photos are compared.":
        "Aradaki hareketi ölçmek için aynı cismi iki farklı fotoğrafta (iki tarihte) işaretleyin. Yalnızca bu iki "
        "fotoğraf karşılaştırılır.",
    "Spectrum of mark #1": "1 numaralı işaretin tayfı",
    "First mark (earlier photo)": "İlk işaret (önceki fotoğraf)",
    "Second mark (later photo)": "İkinci işaret (sonraki fotoğraf)",
    "The mark on the first photo.": "İlk fotoğraftaki işaret.",
    "The same object, marked on the second photo.": "Aynı cisim, ikinci fotoğrafta işaretlenmiş hâli.",
    "Compare these two marks": "Bu iki işareti karşılaştır",
    "Measures how far and how fast the object moved between the two photos you marked it on. Only those two photos "
    "are used, so fast movers (asteroids, meteors) are not mixed up with other photos. If the object is a Gaia star, "
    "its predicted move is shown too.":
        "Cismin, onu işaretlediğiniz iki fotoğraf arasında ne kadar ve ne hızla hareket ettiğini ölçer. Yalnızca bu "
        "iki fotoğraf kullanılır; böylece hızlı hareket edenler (asteroitler, meteorlar) başka fotoğraflarla "
        "karışmaz. Cisim bir Gaia yıldızıysa öngörülen hareketi de gösterilir.",
    "Spectrum of the first mark": "İlk işaretin tayfı",
    "Measures the marked object's brightness at every infrared colour SPHEREx sees.":
        "İşaretlenen cismin, SPHEREx'in gördüğü her kızılötesi renkteki parlaklığını ölçer.",
    "These two marks are on the same photo: pick marks made on two different photos.":
        "Bu iki işaret aynı fotoğrafta: iki farklı fotoğrafta yapılmış işaretleri seçin.",
    # --- spectrum -----------------------------------------------------------------------------
    "{}: SPHEREx spectrum, 0.75–5.0 µm": "{}: SPHEREx tayfı, 0,75–5,0 µm",
    "{} images loaded in {} s.": "{} görüntü {} sn'de yüklendi.",
    "← Back to {}": "← {} hedefine dön",
    "Forget this find and measure the target again.": "Bu bulguyu unut ve hedefi yeniden ölç.",
    "A Gaia star (G = {}) lies {} from the target on these dates. SPHEREx pixels are 6.15″, so its light blends into "
    "the measurement — especially at short wavelengths where a faint, cold target is dark.":
        "Bir Gaia yıldızı (G = {}) bu tarihlerde hedeften {} uzakta. SPHEREx pikselleri 6,15″ olduğundan ışığı ölçüme "
        "karışır — özellikle sönük, soğuk bir hedefin karanlık olduğu kısa dalga boylarında.",
    "Dates to show": "Gösterilecek tarihler",
    "Pick two dates to compare their spectra directly; remove one to hide it.":
        "Tayflarını doğrudan karşılaştırmak için iki tarih seçin; gizlemek için birini kaldırın.",
    "Logarithmic flux axis": "Logaritmik akı ekseni",
    "Spreads out faint and bright values; useful when the spectrum spans a big range.":
        "Sönük ve parlak değerleri yayar; tayf geniş bir aralığa yayıldığında işe yarar.",
    "Each point is one exposure; SPHEREx sees each spot at a slightly different wavelength every time, so many "
    "exposures together make a spectrum.":
        "Her nokta bir pozdur; SPHEREx her noktayı her seferinde biraz farklı bir dalga boyunda görür, bu yüzden çok "
        "sayıda poz birlikte bir tayf oluşturur.",
    "Colours are survey epochs.": "Renkler tarama dönemleridir.",
    "Colours are survey epochs (time bins merged).": "Renkler tarama dönemleridir (zaman aralıkları birleştirildi).",
    "Hollow points have flagged pixels in the aperture (often saturation): treat them with care.":
        "İçi boş noktaların açıklığında işaretli pikseller var (çoğu zaman doyma): dikkatle yorumlayın.",
    "Quick look, not calibrated; for science use the [IRSA SPHEREx Spectrophotometry Tool]({}).":
        "Hızlı bakış, kalibre edilmemiş; bilimsel kullanım için [IRSA SPHEREx Spektrofotometri Aracı]({}) kullanın.",
    "{} of {} exposures could not be measured.": "{} / {} poz ölçülemedi.",
    "{} of {} exposures could not be measured; in {} of them the target is so bright that the pixels at its centre "
    "are saturated (bands {}).":
        "{} / {} poz ölçülemedi; bunların {} tanesinde hedef o kadar parlak ki merkezindeki pikseller doymuş "
        "(bantlar {}).",
    "They are listed in the table, not plotted.": "Tabloda listelenirler, grafiğe çizilmezler.",
    "Light curve in {} (choose the band in the sidebar)": "{} bandında ışık eğrisi (bandı kenar çubuğundan seçin)",
    "**Saturated exposures: brightness index from the star's wings ({})**":
        "**Doymuş pozlar: yıldızın kanatlarından parlaklık göstergesi ({})**",
    "When the centre of a very bright star is saturated, the light 3–5 pixels from the centre still rises and falls "
    "with the star's brightness. This index shows the trend only; it is not a flux, and wavelength differences inside "
    "the band add scatter.":
        "Çok parlak bir yıldızın merkezi doyduğunda, merkezden 3–5 piksel uzaktaki ışık yine de yıldızın parlaklığıyla "
        "birlikte artıp azalır. Bu gösterge yalnızca eğilimi gösterir; bir akı değildir ve bant içindeki dalga boyu "
        "farkları saçılma ekler.",
    "**What changed?** (median of unflagged points; different wavelengths inside a band add scatter)":
        "**Ne değişti?** (işaretsiz noktaların medyanı; bant içindeki farklı dalga boyları saçılma ekler)",
    "**{}**: epoch {} is {}% {} than epoch {} — a real change is likely ({}σ){}":
        "**{0}**: {1}. dönem, {4}. dönemden %{2} {3} — gerçek bir değişim olası ({5}σ){6}",
    "**{}**: epoch {} is {}% {} than epoch {} — within the scatter ({}σ){}":
        "**{0}**: {1}. dönem, {4}. dönemden %{2} {3} — saçılma içinde ({5}σ){6}",
    "brighter": "daha parlak", "fainter": "daha sönük", "(from flagged points)": "(işaretli noktalardan)",
    "Note: the neighbouring star above moves relative to the target, so part of any change can come from it entering "
    "or leaving the aperture.":
        "Not: yukarıdaki komşu yıldız hedefe göre hareket ediyor; bu yüzden değişimin bir kısmı onun açıklığa girip "
        "çıkmasından kaynaklanabilir.",
    "Table of measurements": "Ölçüm tablosu", "Download CSV": "CSV indir",
    "Load more exposures (up to {} per band)": "Daha fazla poz yükle (bant başına en çok {})",
    "date": "tarih", "band": "bant", "wavelength": "dalga boyu", "flux": "akı", "err": "hata", "flags": "bayraklar",
    "epoch": "dönem", "wing": "kanat",
    # --- motion check -------------------------------------------------------------------------
    "Motion check: {} in {}": "Hareket kontrolü: {} ({} bandı)",
    "{} photos loaded in {} s": "{} fotoğraf {} sn'de yüklendi",
    "{} photos loaded in {} s.": "{} fotoğraf {} sn'de yüklendi.",
    "The object needs to be visible in at least two photos. Try a band where it is bright, or a larger field of view.":
        "Cismin en az iki fotoğrafta görünmesi gerekir. Parlak olduğu bir bandı ya da daha geniş bir görüş alanını "
        "deneyin.",
    "Measured motion": "Ölçülen hareket", "Catalogue motion": "Katalog hareketi", "Difference": "Fark",
    "{} ± {} ″/yr": "{} ± {} ″/yıl", "{} ″/yr": "{} ″/yıl",
    "How fast SPHEREx sees the object drift across the sky, in arcseconds per year: the slope of a straight line "
    "fitted through the measured positions after removing the parallax wobble. ± is the uncertainty (1σ).":
        "SPHEREx'in cismin gökyüzünde ne hızla kaydığını gördüğü, yılda yay saniyesi cinsinden: paralaks salınımı "
        "çıkarıldıktan sonra ölçülen konumlardan geçirilen doğrunun eğimi. ± belirsizliktir (1σ).",
    "What the catalogue says ({}): the speed the grey line is drawn with.":
        "Kataloğun söylediği ({}): gri çizginin çizildiği hız.",
    "Measured minus catalogue. σ (sigma) counts how many uncertainties apart they are: below about 3σ they agree.":
        "Ölçülen eksi katalog. σ (sigma) aralarında kaç belirsizlik olduğunu sayar: yaklaşık 3σ'nın altında uyuşurlar.",
    "**Agrees** with the catalogue: SPHEREx sees {} move {}″ per year, the catalogue says {}″.":
        "Katalogla **uyuşuyor**: SPHEREx {} cisminin yılda {}″ hareket ettiğini görüyor, katalog {}″ diyor.",
    "**Differs** from the catalogue by {}″/yr ({}σ). Saturation (an overexposed star), a blended neighbour or too few "
    "epochs can cause this; check the blink.":
        "Katalogdan yılda {}″ **farklı** ({}σ). Doyma (aşırı pozlanmış yıldız), karışan bir komşu ya da çok az dönem "
        "buna yol açabilir; blink görünümünü kontrol edin.",
    "**Moves**: {} ± {}″/yr, more than 3× the uncertainty.": "**Hareket ediyor**: yılda {} ± {}″, belirsizliğin 3 katından fazla.",
    "No motion detected: less than about {}″/yr (3σ).": "Hareket saptanmadı: yılda yaklaşık {}″'den az (3σ).",
    "**Reading the plot:** each coloured point is where the object's light is centred in one photo (east to the left, "
    "as on the sky), relative to its catalogue position on 2026-01-01 (the 0, 0 point). The grey line is where it "
    "*should* be. Error bars show the measuring uncertainty. Saturated (overexposed) stars and blended neighbours make "
    "centres less reliable.":
        "**Grafiği okumak:** her renkli nokta, cismin ışığının bir fotoğraftaki merkezidir (gökyüzündeki gibi doğu "
        "solda), 2026-01-01'deki katalog konumuna göre (0, 0 noktası). Gri çizgi olması *gereken* yerdir. Hata "
        "çubukları ölçüm belirsizliğini gösterir. Doymuş (aşırı pozlanmış) yıldızlar ve karışan komşular merkezleri "
        "daha az güvenilir yapar.",
    "How the prediction (grey line) and the measurement are calculated":
        "Öngörü (gri çizgi) ve ölçüm nasıl hesaplanır",
    "**The prediction** starts from the catalogue ({}) and adds two effects:\n\n1. **Proper motion** (the star's own "
    "drift): **{}″/yr** east and **{}″/yr** north. After *t* years the star has drifted *t* × that. *t* is counted "
    "from 2026-01-01.\n2. **Parallax** (the yearly wobble from watching on a moving Earth): **π = {}″**. Earth's "
    "position around the Sun on each date, *X, Y, Z* in AU (Earth–Sun distances), comes from the planetary ephemeris "
    "in astropy. With the star's RA α and Dec δ:\n\n   - east shift = π · (X sin α − Y cos α)\n   - north shift = π · "
    "(X cos α sin δ + Y sin α sin δ − Z cos δ)\n3. **Predicted position** = drift + wobble. Repeating this for many "
    "dates draws the grey line; for a nearby star it is a slanted line with loops (the loops are the parallax).\n\n"
    "**The measurement:** in each epoch photo the app looks within 4 display pixels (12″) of the predicted spot and "
    "takes the *centroid*, the brightness-weighted average position of the light, which can be pinned down to a "
    "fraction of a pixel. The error bar is the spread between the single photos of that epoch divided by √(number of "
    "photos). Finally the parallax part is subtracted from every point and a straight line is fitted through position "
    "versus time (weighted least squares): its slope is the **measured motion** above.\n\n**Step by step for this "
    "object** (all values in arcseconds, ″):":
        "**Öngörü** katalogdan ({}) başlar ve iki etki ekler:\n\n1. **Öz hareket** (yıldızın kendi kayması): doğuya "
        "**yılda {}″** ve kuzeye **yılda {}″**. *t* yıl sonra yıldız bunun *t* katı kadar kaymıştır. *t*, "
        "2026-01-01'den itibaren sayılır.\n2. **Paralaks** (hareket eden Dünya'dan izlediğimiz için yıllık salınım): "
        "**π = {}″**. Dünya'nın her tarihte Güneş etrafındaki konumu, AU (Dünya–Güneş uzaklığı) cinsinden *X, Y, Z*, "
        "astropy'deki gezegen efemerisinden gelir. Yıldızın RA α ve Dec δ değerleriyle:\n\n   - doğu kayması = π · "
        "(X sin α − Y cos α)\n   - kuzey kayması = π · (X cos α sin δ + Y sin α sin δ − Z cos δ)\n3. **Öngörülen "
        "konum** = kayma + salınım. Bunu birçok tarih için tekrarlamak gri çizgiyi çizer; yakın bir yıldız için "
        "ilmekli eğik bir çizgidir (ilmekler paralakstır).\n\n**Ölçüm:** her dönem fotoğrafında uygulama öngörülen "
        "noktanın 4 görüntü pikseli (12″) yakınına bakar ve ışığın parlaklık ağırlıklı ortalama konumu olan "
        "*ağırlık merkezini* alır; bu, bir pikselin kesri kadar hassas belirlenebilir. Hata çubuğu, o dönemin tek "
        "fotoğrafları arasındaki saçılmanın √(fotoğraf sayısı)'na bölünmesidir. Son olarak paralaks kısmı her "
        "noktadan çıkarılır ve konum-zaman noktalarından bir doğru geçirilir (ağırlıklı en küçük kareler): eğimi "
        "yukarıdaki **ölçülen harekettir**.\n\n**Bu cisim için adım adım** (tüm değerler yay saniyesi, ″):",
    "Photo": "Fotoğraf", "Date": "Tarih", "Years from 2026-01-01": "2026-01-01'den bu yana yıl",
    "Drift east": "Doğu kayması", "Drift north": "Kuzey kayması", "Wobble east": "Doğu salınımı",
    "Wobble north": "Kuzey salınımı", "Predicted east": "Öngörülen doğu", "Predicted north": "Öngörülen kuzey",
    "Measured east": "Ölçülen doğu", "Measured north": "Ölçülen kuzey", "± error": "± hata",
    "Miss (measured − predicted)": "Sapma (ölçülen − öngörülen)",
    "Drift = proper motion × years. Wobble = parallax shift. Predicted = drift + wobble. Miss = distance between the "
    "measured and predicted points.":
        "Kayma = öz hareket × yıl. Salınım = paralaks kayması. Öngörülen = kayma + salınım. Sapma = ölçülen ve "
        "öngörülen noktalar arasındaki uzaklık.",
    "Download this table (CSV)": "Bu tabloyu indir (CSV)", "Save the step-by-step numbers.": "Adım adım sayıları kaydet.",
    "Watch it move: the photos with the predicted position (orange ring) on each date":
        "Hareketini izle: her tarihteki öngörülen konumla (turuncu halka) fotoğraflar",
    "Motion check at {} ({})": "{} konumunda hareket kontrolü ({})",
    "A motion check compares where a **known object** is predicted to be with where SPHEREx sees it. You typed a sky "
    "position, so nothing is followed here.\n\n**Mark it yourself:** choose *Mark*, click an object on the first date, "
    "switch to a later date with Next or the timeline, and click the *same* object again. The panel shows how far it "
    "moved and its speed in ″ per year. Then **Check my find** below compares it with the Gaia catalogue.":
        "Hareket kontrolü, **bilinen bir cismin** öngörülen konumunu SPHEREx'in gördüğü konumla karşılaştırır. Bir "
        "gökyüzü konumu yazdınız, bu yüzden burada izlenen bir cisim yok.\n\n**Kendiniz işaretleyin:** *İşaretle*'yi "
        "seçin, ilk tarihte bir cisme tıklayın, Sonraki düğmesiyle ya da zaman çizelgesiyle daha sonraki bir tarihe "
        "geçin ve *aynı* cisme yeniden tıklayın. Panel ne kadar hareket ettiğini ve yılda ″ cinsinden hızını gösterir. "
        "Ardından aşağıdaki **Bulduğumu kontrol et** onu Gaia kataloğuyla karşılaştırır.",
    "{} photos loaded in {} s. Stars move very little: anything below about 3″ (half a SPHEREx pixel) between dates is "
    "within the measuring error.":
        "{} fotoğraf {} sn'de yüklendi. Yıldızlar çok az hareket eder: tarihler arasında yaklaşık 3″'den (yarım SPHEREx "
        "pikseli) küçük her şey ölçüm hatası içindedir.",
    # --- motion between two marks -------------------------------------------------------------
    "Motion between your two marks ({})": "İki işaretiniz arasındaki hareket ({})",
    "One of the photos you marked is not in this view any more (another band or field?). Go back, and mark the object "
    "again on two photos.":
        "İşaretlediğiniz fotoğraflardan biri artık bu görünümde yok (başka bir bant veya alan mı?). Geri dönün ve "
        "cismi iki fotoğrafta yeniden işaretleyin.",
    "Photo 1: {}": "Fotoğraf 1: {}", "Photo 2: {}": "Fotoğraf 2: {}",
    "Only these two photos are used.": "Yalnızca bu iki fotoğraf kullanılır.",
    "Time between them: {} days ({} years).": "Aralarındaki süre: {} gün ({} yıl).",
    "Moved": "Hareket", "Direction": "Yön", "Speed": "Hız",
    "Distance on the sky between the object's centre on photo 1 and on photo 2. ± is the measuring uncertainty (1σ) "
    "of the two centres together.":
        "Cismin 1. ve 2. fotoğraftaki merkezleri arasındaki gökyüzü uzaklığı. ± iki merkezin birlikte ölçüm "
        "belirsizliğidir (1σ).",
    "Direction of the move, in degrees from north towards east.": "Hareketin yönü, kuzeyden doğuya doğru derece cinsinden.",
    "Distance divided by the time between the two photos.": "Uzaklığın iki fotoğraf arasındaki süreye bölümü.",
    "{}″ per year": "yılda {}″", "{}″ per day": "günde {}″", "{}″ per hour": "saatte {}″",
    "{} ({}°)": "{} ({}°)",
    "north": "kuzey", "north-east": "kuzeydoğu", "east": "doğu", "south-east": "güneydoğu", "south": "güney",
    "south-west": "güneybatı", "west": "batı", "north-west": "kuzeybatı",
    "**It moved**: {}″ toward the {}, more than 3× the uncertainty ({}″).":
        "**Hareket etti**: {1} yönünde {0}″, belirsizliğin 3 katından fazla ({2}″).",
    "**No clear motion** between these two photos: the shift ({}″) is within 3× the uncertainty ({}″). One SPHEREx "
    "pixel is 6.15″.":
        "Bu iki fotoğraf arasında **belirgin hareket yok**: kayma ({}″) belirsizliğin 3 katı ({}″) içinde. Bir SPHEREx "
        "pikseli 6,15″'dir.",
    "**Agrees** with Gaia: Gaia predicts a move of {}″ between these two dates; the measured move is {}″ away from "
    "that (uncertainty {}″).":
        "Gaia ile **uyuşuyor**: Gaia bu iki tarih arasında {}″ hareket öngörüyor; ölçülen hareket bundan {}″ uzakta "
        "(belirsizlik {}″).",
    "**Differs** from Gaia: Gaia predicts a move of {}″ between these two dates; the measured move is {}″ away from "
    "that (uncertainty {}″).":
        "Gaia'dan **farklı**: Gaia bu iki tarih arasında {}″ hareket öngörüyor; ölçülen hareket bundan {}″ uzakta "
        "(belirsizlik {}″).",
    "Mark": "İşaret", "photo 1": "fotoğraf 1", "photo 2": "fotoğraf 2",
    "The centres are measured again on each photo (light-weighted centre near your mark), so a mark a little off the "
    "object does not matter.":
        "Merkezler her fotoğrafta yeniden ölçülür (işaretinizin yakınındaki ışık ağırlıklı merkez), bu yüzden cismin "
        "biraz yanına konmuş bir işaret sorun olmaz.",
    "The two photos (blink them with Play)": "İki fotoğraf (Play ile aralarında geçiş yapın)",
    "Back to {}": "{} hedefine dön", "Back": "Geri",
    "Forget these marks and show the usual motion check.": "Bu işaretleri unut ve olağan hareket kontrolünü göster.",
    # --- views --------------------------------------------------------------------------------
    "You typed a sky position, so nothing is followed here (the orange cross marks the spot). Look for anything that "
    "moves or changes yourself, then **Mark** it on two dates to measure it.":
        "Bir gökyüzü konumu yazdınız, bu yüzden burada izlenen bir cisim yok (turuncu artı noktayı gösterir). Hareket "
        "eden veya değişen bir şeyi kendiniz arayın, sonra ölçmek için onu iki tarihte **işaretleyin**.",
    "Sky viewer": "Gökyüzü görüntüleyici", "Photo list": "Fotoğraf listesi",
    "Subtract needs photos from at least two dates (epochs) in this band.":
        "Çıkarma için bu bantta en az iki tarihten (dönemden) fotoğraf gerekir.",
    "Older photo (A)": "Eski fotoğraf (A)", "The 'before' picture.": "'Önceki' resim.",
    "Newer photo (B)": "Yeni fotoğraf (B)", "The 'after' picture. The view shows B − A.": "'Sonraki' resim. Görünüm B − A'yı gösterir.",
    "Hide changes smaller than (× noise)": "Şundan küçük değişimleri gizle (× gürültü)",
    "Sets every pixel whose change is smaller than this many noise levels (σ) to 'no change' (white), so only clear "
    "changes stay coloured. 0 shows everything.":
        "Değişimi bu kadar gürültü düzeyinden (σ) küçük olan her pikseli 'değişim yok' (beyaz) yapar; böylece yalnızca "
        "belirgin değişimler renkli kalır. 0 her şeyi gösterir.",
    "How to read the colours": "Renkler nasıl okunur",
    "**Red** = brighter now (in B) · **white / grey** = no change (steady stars cancel out) · **blue** = was brighter "
    "before (in A).":
        "**Kırmızı** = şimdi daha parlak (B'de) · **beyaz / gri** = değişim yok (sabit yıldızlar birbirini götürür) · "
        "**mavi** = önceden daha parlaktı (A'da).",
    "A **moving star** leaves a blue + red pair — blue where it *was*, red where it *is now*. A star that **changed "
    "brightness** leaves a single red (brightened) or blue (faded) spot. Small rings around very bright stars are "
    "leftovers of imperfect cancelling, not real changes. The stronger the colour, the bigger the change.":
        "**Hareket eden bir yıldız** mavi + kırmızı bir çift bırakır — *olduğu* yerde mavi, *şimdi olduğu* yerde "
        "kırmızı. **Parlaklığı değişen** bir yıldız tek bir kırmızı (parladı) veya mavi (söndü) leke bırakır. Çok "
        "parlak yıldızların çevresindeki küçük halkalar kusurlu çıkarmanın artıklarıdır, gerçek değişim değildir. Renk "
        "ne kadar güçlüyse değişim o kadar büyüktür.",
    "Order": "Sıra", "time": "zaman",
    "time: oldest to newest. wavelength: from the bluest to the reddest infrared colour inside the band.":
        "zaman: en eskiden en yeniye. dalga boyu: bant içindeki en mavi kızılötesi renkten en kırmızısına.",
    "All {} photos have warnings at the target (it is probably overexposed in this band), so none are hidden. Try "
    "another band.":
        "{} fotoğrafın hepsinde hedefte uyarı var (bu bantta muhtemelen aşırı pozlanmış), bu yüzden hiçbiri gizlenmedi. "
        "Başka bir bant deneyin.",
    "Hide photos with warnings at the target ({} of {})": "Hedefte uyarı olan fotoğrafları gizle ({} / {})",
    "Leaves out single photos whose pixels at the target were marked unreliable by the NASA pipeline (overexposed, "
    "cosmic-ray hits…).":
        "Hedefteki pikselleri NASA işlem hattı tarafından güvenilmez olarak işaretlenen tek fotoğrafları dışarıda "
        "bırakır (aşırı pozlama, kozmik ışın izleri…).",
    # --- helpers ------------------------------------------------------------------------------
    "Helpers — off until you switch them on": "Yardımcılar — siz açana kadar kapalı",
    "The app does not point things out by itself: look first, then let a helper check what you saw.":
        "Uygulama kendiliğinden bir şey göstermez: önce siz bakın, sonra gördüğünüzü bir yardımcıya kontrol ettirin.",
    "Detect stars": "Yıldızları bul",
    "Finds spots well above the noise on each photo and draws small green crosses on them. Faint or crowded stars can "
    "be missed: one SPHEREx pixel is 6.15″ wide.":
        "Her fotoğrafta gürültünün belirgin şekilde üstündeki noktaları bulur ve üzerlerine küçük yeşil artılar çizer. "
        "Sönük veya kalabalık yıldızlar kaçırılabilir: bir SPHEREx pikseli 6,15″ genişliğindedir.",
    "Find what changed": "Neyin değiştiğini bul",
    "Compares two dates and lists stars that moved, brightened or faded, strongest first. Labels C1, C2… appear on the "
    "image.":
        "İki tarihi karşılaştırır ve hareket eden, parlayan veya sönen yıldızları en güçlüden başlayarak listeler. "
        "Resimde C1, C2… etiketleri görünür.",
    "Detection threshold (σ)": "Algılama eşiği (σ)",
    "How far above the noise a spot must be. Higher = fewer but surer detections. σ (sigma) is the noise level of the "
    "photo.":
        "Bir noktanın gürültünün ne kadar üstünde olması gerektiği. Daha yüksek = daha az ama daha kesin algılama. σ "
        "(sigma) fotoğrafın gürültü düzeyidir.",
    "Detected stars: {} (per photo, above {}σ).": "Bulunan yıldızlar: {} (fotoğraf başına, {}σ üstünde).",
    "Compare from (older)": "Karşılaştırma başlangıcı (eski)", "…to (newer)": "…bitiş (yeni)",
    "The 'before' photo for Find what changed.": "Neyin değiştiğini bul için 'önceki' fotoğraf.",
    "The 'after' photo.": "'Sonraki' fotoğraf.",
    "Note: one photo is from an older survey (another telescope, scaled for display): only *moved* results mean "
    "something; brightness changes between telescopes do not.":
        "Not: fotoğraflardan biri eski bir taramadan (başka bir teleskop, gösterim için ölçeklenmiş): yalnızca "
        "*hareket* sonuçları anlamlıdır; teleskoplar arası parlaklık değişimleri anlamlı değildir.",
    "label": "etiket", "what": "ne", "details": "ayrıntı", "strength (σ)": "güç (σ)",
    "moved": "hareket etti", "brightened": "parladı", "faded": "söndü",
    "How many noise levels the change is; above ~5 it is very likely real":
        "Değişimin kaç gürültü düzeyi olduğu; ~5'in üstündeyse büyük olasılıkla gerçektir",
    "moved {}″ toward the {} ({}″ per year)": "{1} yönünde {0}″ hareket etti (yılda {2}″)",
    "moved {}″ toward the {}": "{1} yönünde {0}″ hareket etti",
    "brighter by {}% ({}σ)": "%{} daha parlak ({}σ)", "fainter by {}% ({}σ)": "%{} daha sönük ({}σ)",
    "brighter by {}% ({}σ, after allowing for all stars changing by {}%)":
        "%{} daha parlak ({}σ, tüm yıldızların %{} değişimi hesaba katıldıktan sonra)",
    "fainter by {}% ({}σ, after allowing for all stars changing by {}%)":
        "%{} daha sönük ({}σ, tüm yıldızların %{} değişimi hesaba katıldıktan sonra)",
    "{} → {}.": "{} → {}.",
    "Check each one by eye: rings around very bright stars and stars at the edge can be false alarms.":
        "Her birini gözle kontrol edin: çok parlak yıldızların çevresindeki halkalar ve kenardaki yıldızlar yanlış "
        "alarm olabilir.",
    "Pick two different photos to compare.": "Karşılaştırmak için iki farklı fotoğraf seçin.",
    "No change above {}σ between {} and {}.": "{1} ile {2} arasında {0}σ üstünde değişim yok.",
    "*Note: the helpers are experimental. They can miss things or report false alarms, so always check their results "
    "by eye.*":
        "*Not: yardımcılar deneyseldir. Bir şeyleri kaçırabilir veya yanlış alarm verebilirler; sonuçlarını her zaman "
        "gözle kontrol edin.*",
    # --- buttons, footer, photo list ----------------------------------------------------------
    "Load all {} photos ({} of {})": "Tüm {} fotoğraflarını yükle ({} / {})",
    "The quick look averages a few photos per date. Loading all gives a cleaner, deeper image but takes longer.":
        "Hızlı bakış tarih başına birkaç fotoğrafın ortalamasını alır. Hepsini yüklemek daha temiz ve derin bir resim "
        "verir ama daha uzun sürer.",
    "Download GIF": "GIF indir", "Save the animation as a GIF to share.": "Paylaşmak için animasyonu GIF olarak kaydet.",
    "What am I looking at?": "Neye bakıyorum?",
    "Each frame is a SPHEREx photo of the same patch of sky, taken months apart and averaged per *epoch* (one pass of "
    "the survey).":
        "Her kare aynı gökyüzü parçasının aylarca arayla çekilmiş ve her *dönem* (taramanın bir geçişi) için "
        "ortalaması alınmış bir SPHEREx fotoğrafıdır.",
    "North is up, east is left; the white bar is 1 arcminute (1′).": "Kuzey yukarıda, doğu solda; beyaz çubuk 1 yay dakikasıdır (1′).",
    "The small orange cross marks the position you typed.": "Küçük turuncu artı yazdığınız konumu gösterir.",
    "The orange ring is where the target should be on that date, from its catalogue proper motion.":
        "Turuncu halka, katalogdaki öz hareketine göre hedefin o tarihte olması gereken yerdir.",
    "Blue circles are Gaia stars.": "Mavi halkalar Gaia yıldızlarıdır.",
    "A star that moves (like Barnard's Star) shifts against the background; a variable star changes brightness.":
        "Hareket eden bir yıldız (Barnard Yıldızı gibi) arka plana göre kayar; değişen bir yıldızın parlaklığı değişir.",
    "*Compare* in the viewer puts two dates side by side, under a swipe divider or subtracted.":
        "Görüntüleyicideki *Karşılaştır* iki tarihi yan yana, kaydırmalı bir ayırıcının altında ya da çıkarılmış gösterir.",
    "Motion between epochs is often smaller than one SPHEREx pixel (6.15″), so the image is shown 2× finer than the "
    "camera's pixels.":
        "Dönemler arasındaki hareket çoğu zaman bir SPHEREx pikselinden (6,15″) küçüktür, bu yüzden resim kameranın "
        "piksellerinden 2 kat daha ince gösterilir.",
    "Your marks stay in your browser (remembered on this computer).":
        "İşaretleriniz tarayıcınızda kalır (bu bilgisayarda hatırlanır).",
    "{} SPHEREx photos cover this position in all bands; {} in {}.":
        "Bu konumu tüm bantlarda {} SPHEREx fotoğrafı kapsıyor; {2} bandında {1}.",
    "Day the photo was taken": "Fotoğrafın çekildiği gün", "MJD": "MJD",
    "Modified Julian Date: astronomers' day count": "Değiştirilmiş Jülyen Tarihi: gökbilimcilerin gün sayımı",
    "wavelength (µm)": "dalga boyu (µm)", "Infrared colour at the target in this photo": "Bu fotoğrafta hedefteki kızılötesi renk",
    "warnings at target": "hedefteki uyarılar",
    "Pipeline flags on the pixels at the target (e.g. overexposed)": "Hedefteki piksellerdeki işlem hattı bayrakları (örn. aşırı pozlama)",
    "file": "dosya", "The original data file at NASA/IPAC": "NASA/IPAC'taki özgün veri dosyası",
    "Position RA {}, Dec {}.": "Konum RA {}, Dec {}.",
    "{}: RA {}, Dec {} on 2026-01-01 ({}).": "{}: 2026-01-01'de RA {}, Dec {} ({}).",
    "SPHEREx data through {} · processing {} · {} · loaded in {} s · Data: NASA/IPAC Infrared Science Archive (IRSA, "
    "via its AWS copy), ESA Gaia.":
        "SPHEREx verileri {} tarihine kadar · işleme {} · {} · {} sn'de yüklendi · Veri: NASA/IPAC Kızılötesi Bilim "
        "Arşivi (IRSA, AWS kopyası üzerinden), ESA Gaia.",
    "{} Gaia DR3 stars (G < {}, via {})": "{} Gaia DR3 yıldızı (G < {}, kaynak: {})",
    "Gaia star list still loading": "Gaia yıldız listesi hâlâ yükleniyor",
    "Gaia catalogue unavailable right now.": "Gaia kataloğu şu anda kullanılamıyor.",
    "{} more SPHEREx photos are still downloading (slow connection); each epoch already has at least one":
        "{} SPHEREx fotoğrafı daha hâlâ iniyor (yavaş bağlantı); her dönemde zaten en az bir tane var",
    "the older WISE (2010) / 2MASS (~2000) photos are still downloading": "eski WISE (2010) / 2MASS (~2000) fotoğrafları hâlâ iniyor",
    "the Gaia star list is still loading": "Gaia yıldız listesi hâlâ yükleniyor",
    "They are added here automatically as soon as they arrive.": "Gelir gelmez buraya otomatik olarak eklenirler.",
    # --- charts -------------------------------------------------------------------------------
    "Wavelength (µm)": "Dalga boyu (µm)", "Flux (mJy, quick look)": "Akı (mJy, hızlı bakış)",
    "{} flux (mJy)": "{} akısı (mJy)", "Wing brightness (MJy/sr, relative)": "Kanat parlaklığı (MJy/sr, göreli)",
    "Epoch {}: {}": "Dönem {}: {}", "Epoch {}": "Dönem {}", "{} (flagged)": "{} (işaretli)", "Epoch {}<br>{}": "Dönem {}<br>{}",
    "predicted (proper motion + parallax)": "öngörülen (öz hareket + paralaks)",
    "East offset (″, east is left)": "Doğu kayması (″, doğu solda)", "North offset (″)": "Kuzey kayması (″)",
    "Miss: measured − predicted (″)": "Sapma: ölçülen − öngörülen (″)", "half a SPHEREx pixel (3″)": "yarım SPHEREx pikseli (3″)",
    "Gaia prediction": "Gaia öngörüsü",
    "%{x}<br>%{y:.1f}″ from the prediction<extra></extra>": "%{x}<br>öngörüden %{y:.1f}″<extra></extra>",
    "{}<br>east %{x:.1f}″, north %{y:.1f}″ ± %{customdata[1]:.1f}″<br>%{customdata[2]:.1f}″ from prediction<extra></extra>":
        "{}<br>doğu %{x:.1f}″, kuzey %{y:.1f}″ ± %{customdata[1]:.1f}″<br>öngörüden %{customdata[2]:.1f}″<extra></extra>",
    "{}<br>east %{x:.1f}″, north %{y:.1f}″ ± {}″<extra></extra>": "{}<br>doğu %{x:.1f}″, kuzey %{y:.1f}″ ± {}″<extra></extra>",
    "%{customdata[0]} · %{customdata[4]}<br>λ %{customdata[1]:.3f} µm<br>%{y:.4g} ± %{customdata[2]:.2g} mJy<br>flags: "
    "%{customdata[3]}<extra></extra>":
        "%{customdata[0]} · %{customdata[4]}<br>λ %{customdata[1]:.3f} µm<br>%{y:.4g} ± %{customdata[2]:.2g} mJy<br>"
        "bayraklar: %{customdata[3]}<extra></extra>",
}

# --- explainers, Check my find and the motion analysis (spx/motion.py, app.py) --------------------------------------
TR.update({
    "How to use {} (30 seconds)": "{} nasıl kullanılır (30 saniye)",
    "Hover over the small question-mark icons for short explanations; *Words used here* in the sidebar explains the "
    "astronomy terms.":
        "Kısa açıklamalar için küçük soru işaretlerinin üzerine gelin; kenar çubuğundaki *Burada kullanılan sözcükler* "
        "astronomi terimlerini açıklar.",
    "about {} s left": "yaklaşık {} sn kaldı", "about {} min left": "yaklaşık {} dk kaldı",
    "almost done": "neredeyse bitti", "{} of {} photos": "{} / {} fotoğraf",
    "taking longer than usual": "her zamankinden uzun sürüyor",
    "under {}″/h": "{}″/sa altında", "under {}″/min": "{}″/dk altında",
    "No motion between the positions ({} apart): they agree within the measuring error (any motion is below {} at "
    "3σ). Over so short a time only an asteroid, a satellite or a meteor could move visibly; every star stands still.":
        "Konumlar arasında hareket yok ({} arayla): ölçüm hatası içinde uyuşuyorlar (herhangi bir hareket 3σ "
        "düzeyinde {}'den az). Bu kadar kısa sürede yalnızca bir asteroit, bir uydu veya bir meteor görünür biçimde "
        "hareket edebilir; her yıldız yerinde durur.",
    "Random teleport (/RTP): RA {}°, Dec {}°. Type /RTP again for another spot.":
        "Rastgele ışınlanma (/RTP): RA {}°, Dec {}°. Başka bir nokta için yeniden /RTP yazın.",
    # Check my find
    "Mark the same object on two or more photos (different dates) to measure how it moves. Every mark is compared "
    "with every other one, and the path is predicted into the future.":
        "Nasıl hareket ettiğini ölçmek için aynı cismi iki veya daha fazla fotoğrafta (farklı tarihlerde) işaretleyin. "
        "Her işaret diğer tüm işaretlerle karşılaştırılır ve yol geleceğe doğru tahmin edilir.",
    "Marks of one object (all of them are compared together)": "Tek bir cismin işaretleri (hepsi birlikte karşılaştırılır)",
    "Keep the marks you made on the same object, on different photos. Leave out marks of other objects. Three or more "
    "marks let the app check that they lie on one path and spot a mark that does not belong.":
        "Aynı cisim üzerinde, farklı fotoğraflarda yaptığınız işaretleri tutun; başka cisimlerin işaretlerini çıkarın. "
        "Üç veya daha fazla işaret, uygulamanın hepsinin tek bir yol üzerinde olduğunu denetlemesini ve uymayan bir "
        "işareti bulmasını sağlar.",
    "Analyse the motion of these {} marks": "Bu {} işaretin hareketini çözümle",
    "Fits one path through all the marks at once, tests whether the object stands still, drifts like a star (with or "
    "without the yearly parallax wobble) or curves like an asteroid, says what kind of object it probably is and why, "
    "and predicts where it will be in the coming years. Each mark is measured only on the photo it was made on.":
        "Tüm işaretlerden aynı anda tek bir yol geçirir; cismin durup durmadığını, bir yıldız gibi kayıp kaymadığını "
        "(yıllık paralaks salınımıyla ya da onsuz) veya bir asteroit gibi eğri çizip çizmediğini sınar; muhtemelen ne "
        "tür bir cisim olduğunu ve nedenini söyler ve önümüzdeki yıllarda nerede olacağını tahmin eder. Her işaret "
        "yalnızca yapıldığı fotoğrafta ölçülür.",
    "These marks are all on the same photo: pick marks made on at least two different photos.":
        "Bu işaretlerin hepsi aynı fotoğrafta: en az iki farklı fotoğrafta yapılmış işaretler seçin.",
    # the motion analysis page
    "Motion of your {} marks ({})": "{} işaretinizin hareketi ({})",
    "Also measure it on the other photos, at the spot your marks predict":
        "Diğer fotoğraflarda da, işaretlerinizin öngördüğü noktada ölç",
    "Off by default: the app does not look for things by itself. When on, the path through your marks predicts where "
    "the object should be on every other photo of this view; where a clear spot sits there (5σ, within the "
    "prediction's uncertainty), it is measured and added (diamonds, F1, F2…). More dates make the path, the parallax "
    "and the prediction much better.":
        "Varsayılan olarak kapalı: uygulama kendi başına bir şey aramaz. Açıkken, işaretlerinizden geçen yol cismin bu "
        "görünümdeki diğer her fotoğrafta nerede olması gerektiğini öngörür; orada belirgin bir nokta varsa (5σ, "
        "tahminin belirsizliği içinde) ölçülür ve eklenir (elmaslar, F1, F2…). Daha çok tarih; yolu, paralaksı ve "
        "tahmini çok daha iyi yapar.",
    "Found on {} more photo(s) at the predicted spot.": "Öngörülen noktada {} fotoğrafta daha bulundu.",
    "Nothing clear at the predicted spot on the other photos.": "Diğer fotoğraflarda öngörülen noktada belirgin bir şey yok.",
    "Your marks are hours apart: the other photos are months away, too far to follow an asteroid there.":
        "İşaretleriniz saatler arayla: diğer fotoğraflar aylarca uzakta, bir asteroidi orada izlemek için fazla uzak.",
    "Marks {} were made on photos that are not in this view (another band or field?), so they are left out.":
        "{} işaretleri bu görünümde olmayan fotoğraflarda yapılmış (başka bant veya alan?), bu yüzden dışarıda bırakıldı.",
    "At least two of the marks must be on different photos of this view. Go back and mark the object on two or more "
    "dates.":
        "İşaretlerden en az ikisi bu görünümün farklı fotoğraflarında olmalı. Geri dönün ve cismi iki veya daha fazla "
        "tarihte işaretleyin.",
    "{} positions on {} photos, spanning {}. Every mark is measured only on the photo it was made on, then all are "
    "fitted together. {}":
        "{1} fotoğrafta {0} konum, {2} boyunca. Her işaret yalnızca yapıldığı fotoğrafta ölçülür, sonra hepsi birlikte "
        "uydurulur. {3}",
    "Every mark matches the same Gaia DR3 star, {} (G = {}).": "Her işaret aynı Gaia DR3 yıldızıyla eşleşiyor: {} (G = {}).",
    "No Gaia star within {}″ of any mark (on each mark's date).":
        "Hiçbir işaretin {}″ yakınında (her işaretin kendi tarihinde) Gaia yıldızı yok.",
    "The Gaia catalogue could not be reached right now, so the catalogue check is left out.":
        "Gaia kataloğuna şu anda ulaşılamadı, bu yüzden katalog denetimi yapılmadı.",
    "Some marks match Gaia stars and some do not (or different ones), so the catalogue check is left out. If the "
    "object moves, your marks may be on neighbouring stars.":
        "Bazı işaretler Gaia yıldızlarıyla eşleşiyor, bazıları eşleşmiyor (veya farklı yıldızlarla), bu yüzden katalog "
        "denetimi yapılmadı. Cisim hareket ediyorsa işaretleriniz komşu yıldızlarda olabilir.",
    "What it probably is: {} ({}%)": "Muhtemelen ne: {} (%{})",
    "#### What it probably is: {} ({}%)": "#### Muhtemelen ne: {} (%{})",
    "#### Not clear yet. Most likely: {} ({}%)": "#### Henüz net değil. En olası: {} (%{})",
    "#### Where it goes next: predicted path": "#### Sonra nereye gidecek: tahmini yol",
    "#### What SPHEREx's own measurements say about {}": "#### SPHEREx'in kendi ölçümleri {} hakkında ne diyor",
    "**The evidence, step by step**": "**Kanıtlar, adım adım**",
    "{}″/yr": "{}″/yıl", "{}″/h": "{}″/sa", "under {}″/yr": "{}″/yıl altında", "{}° ({})": "{}° ({})",
    "Not clear yet. Most likely: {} ({}%)": "Henüz net değil. En olası: {} (%{})",
    "The evidence, step by step": "Kanıtlar, adım adım",
    "For": "Lehine", "Against": "Aleyhine",
    "An algorithmic estimate from the evidence above, not a certainty: always look at the photos too. The percentages "
    "say how strongly the evidence points to each kind of object.":
        "Yukarıdaki kanıtlardan algoritmik bir tahmin, kesinlik değil: fotoğraflara da mutlaka bakın. Yüzdeler, "
        "kanıtların her cisim türünü ne kadar güçlü gösterdiğini söyler.",
    "How strongly the evidence points to it (%)": "Kanıtın onu ne kadar güçlü gösterdiği (%)",
    "Standing still: a background star or galaxy": "Duruyor: bir arka plan yıldızı veya gökada",
    "Moving star: a nearby star or brown dwarf with a steady proper motion":
        "Hareketli yıldız: düzgün öz hareketi olan yakın bir yıldız veya kahverengi cüce",
    "Solar System object: an asteroid or comet": "Güneş Sistemi cismi: bir asteroit veya kuyruklu yıldız",
    "Not one object: the marks do not belong together, or an artefact":
        "Tek bir cisim değil: işaretler birbirine ait değil ya da bir yapay iz",
    "Standing still": "Duruyor", "Moving star": "Hareketli yıldız", "Asteroid / comet": "Asteroit / kuyruklu yıldız",
    "Not one object": "Tek cisim değil",
    "Speed": "Hız", "Direction": "Yön", "Path model": "Yol modeli", "Distance (from parallax)": "Uzaklık (paralakstan)",
    "Consistency (χ²ν)": "Tutarlılık (χ²ν)",
    "No motion was found; it is slower than this (3σ).": "Hareket bulunmadı; bundan daha yavaş (3σ).",
    "How fast it moves across the sky, from one path fitted through all the marks at once (not from two marks only). "
    "± is the uncertainty (1σ).":
        "Gökyüzünde ne kadar hızlı hareket ettiği; tüm işaretlerden aynı anda geçirilen tek bir yoldan (yalnızca iki "
        "işaretten değil). ± belirsizliktir (1σ).",
    "Direction of motion, in degrees from north towards east.": "Hareketin yönü, kuzeyden doğuya doğru derece cinsinden.",
    "still": "duruyor", "drift": "kayma", "drift + parallax": "kayma + paralaks", "curved": "eğri",
    "standing still": "duruyor", "steady drift (proper motion)": "düzgün kayma (öz hareket)",
    "steady drift + yearly parallax wobble": "düzgün kayma + yıllık paralaks salınımı", "curved path": "eğri yol",
    "Parallax π is the yearly wobble; distance in parsecs = 1 / π (1 pc = 3.26 light-years).":
        "Paralaks π yıllık salınımdır; parsek cinsinden uzaklık = 1 / π (1 pc = 3,26 ışık yılı).",
    "How well the marks follow one path: about 1 = within the measuring errors; much more than 1 = the marks scatter "
    "more than they should (a wrong mark, a blend, or different objects).":
        "İşaretlerin tek bir yolu ne kadar iyi izlediği: yaklaşık 1 = ölçüm hataları içinde; 1'den çok büyük = işaretler "
        "olması gerekenden fazla saçılıyor (yanlış bir işaret, karışma veya farklı cisimler).",
    "With two marks any path fits exactly: mark a third photo to test it.":
        "İki işaretle her yol tam uyar: sınamak için üçüncü bir fotoğrafı işaretleyin.",
    "Where it goes next: predicted path": "Sonra nereye gidecek: tahmini yol",
    "Predict ahead (years)": "İleriye tahmin (yıl)", "Predict ahead (hours)": "İleriye tahmin (saat)",
    "How many years past today to continue the path.": "Yolun bugünden kaç yıl sonrasına kadar sürdürüleceği.",
    "How far past the last mark to continue the path.": "Yolun son işaretten ne kadar sonrasına kadar sürdürüleceği.",
    "The marks probably do not belong to one object, so the prediction below is not reliable. Leave out the mark that "
    "does not fit (see *Every mark against the others* below) and try again.":
        "İşaretler büyük olasılıkla tek bir cisme ait değil, bu yüzden aşağıdaki tahmin güvenilir değil. Uymayan işareti "
        "çıkarın (aşağıda *Her işaret diğerlerine karşı*) ve yeniden deneyin.",
    "Asteroids and comets follow curved orbits, so this straight-line guess holds for hours to a few days only. To "
    "follow a new one for longer, its positions go to the Minor Planet Center, which computes an orbit.":
        "Asteroitler ve kuyruklu yıldızlar eğri yörüngeler izler, bu yüzden bu düz çizgi tahmini yalnızca saatlerden "
        "birkaç güne kadar geçerlidir. Yeni birini daha uzun izlemek için konumları, yörüngeyi hesaplayan Minor Planet "
        "Center'a gönderilir.",
    "**Reading the plot:** coloured points are the marks (east to the left, as on the sky), with their measuring "
    "error. The solid line is the one path fitted through all of them; the dashed line continues it into the future, "
    "with the date of each predicted point. The shaded band is the uncertainty (1σ): it widens the further ahead we "
    "look, because a small error in the speed grows with time.":
        "**Grafiği okumak:** renkli noktalar işaretlerdir (gökyüzündeki gibi doğu solda), ölçüm hatalarıyla. Düz çizgi "
        "hepsinden geçirilen tek yoldur; kesikli çizgi onu, her tahmini noktanın tarihiyle geleceğe sürdürür. Gölgeli "
        "bant belirsizliktir (1σ): ileriye baktıkça genişler, çünkü hızdaki küçük bir hata zamanla büyür.",
    "The dotted grey line is the catalogue's own prediction.": "Noktalı gri çizgi kataloğun kendi tahminidir.",
    "The same path against time: east and north position separately. A steady drift is a straight line; the yearly "
    "parallax wobble makes it wave; an orbit makes it bend. The dotted line is today.":
        "Aynı yol zamana karşı: doğu ve kuzey konumu ayrı ayrı. Düzgün kayma düz bir çizgidir; yıllık paralaks salınımı "
        "onu dalgalandırır; bir yörünge onu büker. Noktalı çizgi bugündür.",
    "prediction uncertainty (1σ)": "tahmin belirsizliği (1σ)", "catalogue prediction": "katalog tahmini",
    "path fitted through the marks": "işaretlerden geçirilen yol", "predicted path (future)": "tahmini yol (gelecek)",
    "predicted position each year": "her yıl tahmini konum", "predicted position each hour step": "her saat adımında tahmini konum",
    "uncertainty (1σ)": "belirsizlik (1σ)", "fitted path": "uydurulan yol", "prediction": "tahmin",
    "East offset (″)": "Doğu kayması (″)", "North offset (″)": "Kuzey kayması (″)",
    "When": "Ne zaman", "± east (″)": "± doğu (″)", "± north (″)": "± kuzey (″)",
    "From the last mark (″)": "Son işaretten (″)", "Catalogue minus measured (″)": "Katalog eksi ölçülen (″)",
    "Download the prediction (CSV)": "Tahmini indir (CSV)",
    "Predicted positions with their uncertainty.": "Belirsizlikleriyle tahmini konumlar.",
    "Every pair of marks compared": "Her işaret çifti karşılaştırıldı", "Pair": "Çift", "Time apart": "Aradaki süre",
    "Moved (″)": "Hareket (″)", "± (″)": "± (″)", "Real move? (σ)": "Gerçek hareket mi? (σ)",
    "Off the fitted path (″)": "Uydurulan yoldan sapma (″)",
    "Each pair on its own: how far the object moved between the two photos, in which direction and how fast. 'Real "
    "move?' counts measuring errors: above 3σ the move is real. 'Off the fitted path' compares the pair's move with "
    "the path through all marks: pairs that agree have small values.":
        "Her çift kendi başına: cismin iki fotoğraf arasında ne kadar, hangi yöne ve ne hızla hareket ettiği. 'Gerçek "
        "hareket mi?' ölçüm hatalarını sayar: 3σ üstünde hareket gerçektir. 'Uydurulan yoldan sapma' çiftin hareketini "
        "tüm işaretlerden geçen yolla karşılaştırır: uyuşan çiftlerin değerleri küçüktür.",
    "Every mark against the others": "Her işaret diğerlerine karşı", "Mark": "İşaret", "Date": "Tarih",
    "Off the path of the other marks (″)": "Diğer işaretlerin yolundan sapma (″)", "Verdict": "Karar",
    "fits": "uyuyor", "slightly off": "biraz sapık", "does not fit": "uymuyor",
    "Each mark is left out in turn; the other marks draw a path, and the table says how far the left-out mark is from "
    "it. A mark that 'does not fit' is probably on another object or badly placed: leave it out in *Check my find* and "
    "analyse again.":
        "Her işaret sırayla dışarıda bırakılır; diğer işaretler bir yol çizer ve tablo, dışarıda bırakılan işaretin bu "
        "yoldan ne kadar uzak olduğunu söyler. 'Uymuyor' denen işaret büyük olasılıkla başka bir cisimde veya kötü "
        "konmuştur: *Bulduğumu denetle*'de çıkarın ve yeniden çözümleyin.",
    "With two marks, each one is needed to draw the path, so they cannot check each other. Mark the object on a third "
    "photo to test them.":
        "İki işaretle her biri yolu çizmek için gereklidir, bu yüzden birbirlerini denetleyemezler. Sınamak için cismi "
        "üçüncü bir fotoğrafta işaretleyin.",
    "The models that were tried": "Denenen modeller", "Degrees of freedom": "Serbestlik derecesi",
    "Improvement": "İyileşme", "Chosen": "Seçilen", "yes": "evet",
    "χ² adds up how far each mark is from the path, in measuring errors (squared). A richer model is chosen only when "
    "it lowers χ² by much more than chance would (improvement > 11.8 for motion, > 9 for parallax, > 16 for a curve), "
    "after scaling by the scatter that is left.":
        "χ², her işaretin yoldan ne kadar uzak olduğunu ölçüm hatası cinsinden (karesi alınarak) toplar. Daha zengin bir "
        "model ancak χ²'yi şansın yapabileceğinden çok daha fazla düşürürse seçilir (hareket için iyileşme > 11,8, "
        "paralaks için > 9, eğri için > 16), kalan saçılmaya göre ölçeklendikten sonra.",
    "How the motion analysis works": "Hareket çözümlemesi nasıl çalışır",
    "Seen in single exposures": "Tek pozlarda görülme", "Photo": "Fotoğraf",
    "Watch it move: every photo with the predicted spot (orange ring)":
        "Hareketini izle: tahmini noktayla her fotoğraf (turuncu halka)",
    "The orange ring follows the path fitted through your marks, also on the photos you did not mark: if the object "
    "sits in the ring there too, the prediction works.":
        "Turuncu halka, işaretlerinizden geçirilen yolu işaretlemediğiniz fotoğraflarda da izler: cisim oralarda da "
        "halkanın içindeyse tahmin çalışıyor demektir.",
    "What SPHEREx's own measurements say about {}": "SPHEREx'in kendi ölçümleri {} hakkında ne diyor",
    "The same analysis as for your own finds: all measured positions fitted together, the kind of object estimated "
    "from the evidence, and the path predicted into the future, next to the catalogue's prediction.":
        "Kendi bulduklarınızla aynı çözümleme: tüm ölçülen konumlar birlikte uydurulur, cisim türü kanıtlardan tahmin "
        "edilir ve yol, kataloğun tahminiyle yan yana geleceğe doğru tahmin edilir.",
    "The same analysis as for your own finds: all measured positions fitted together, the kind of object estimated "
    "from the evidence, and the path predicted into the future.":
        "Kendi bulduklarınızla aynı çözümleme: tüm ölçülen konumlar birlikte uydurulur, cisim türü kanıtlardan tahmin "
        "edilir ve yol geleceğe doğru tahmin edilir.",
    # the evidence lines (spx/motion.py)
    "No motion between the positions ({} apart): they agree within the measuring error (any motion is below {}″ per "
    "year at 3σ).":
        "Konumlar arasında hareket yok ({} arayla): ölçüm hatası içinde uyuşuyorlar (herhangi bir hareket 3σ düzeyinde "
        "yılda {}″'den az).",
    "But the dates are too close together (or the errors too big) to rule out a drift of up to {}″ per year, which a "
    "nearby star could have; there is even a hint of motion ({}σ). More dates, or the older 2MASS / WISE photos "
    "(sidebar), settle it.":
        "Ama tarihler birbirine çok yakın (veya hatalar çok büyük); yakın bir yıldızda olabilecek yılda {}″'ye kadar bir "
        "kayma dışlanamaz; hatta bir hareket belirtisi var ({}σ). Daha çok tarih ya da eski 2MASS / WISE fotoğrafları "
        "(kenar çubuğu) bunu netleştirir.",
    "But the dates are too close together (or the errors too big) to rule out a drift of up to {}″ per year, which a "
    "nearby star could have. More dates, or the older 2MASS / WISE photos (sidebar), settle it.":
        "Ama tarihler birbirine çok yakın (veya hatalar çok büyük); yakın bir yıldızda olabilecek yılda {}″'ye kadar bir "
        "kayma dışlanamaz. Daha çok tarih ya da eski 2MASS / WISE fotoğrafları (kenar çubuğu) bunu netleştirir.",
    "But the positions scatter {}× more than the measuring error, and not along one path: they are probably on "
    "different objects.":
        "Ama konumlar ölçüm hatasından {} kat fazla ve tek bir yol boyunca olmadan saçılıyor: büyük olasılıkla farklı "
        "cisimler üzerindeler.",
    "It moved: {}σ, far more than the measuring error.": "Hareket etti: {}σ, ölçüm hatasından çok daha fazla.",
    "It moved {} within {}. No star moves measurably in days (the fastest, Barnard's Star, drifts {}″ per day).":
        "{} içinde {} hareket etti. Hiçbir yıldız günler içinde ölçülebilir biçimde hareket etmez (en hızlısı Barnard "
        "Yıldızı günde {}″ kayar).".replace("{} içinde {}", "{1} içinde {0}").replace("{}″ kayar", "{2}″ kayar"),
    "{}″ per year is faster than any star (the fastest, Barnard's Star, moves {}″ per year).":
        "Yılda {}″ her yıldızdan hızlıdır (en hızlısı Barnard Yıldızı yılda {}″ hareket eder).",
    "An asteroid crosses a field this size in hours to days, so marks {} apart are probably on two different objects.":
        "Bir asteroit bu büyüklükteki bir alanı saatler ile günler içinde geçer, bu yüzden {} arayla yapılmış işaretler "
        "büyük olasılıkla iki farklı cisim üzerindedir.",
    "Steady drift of {}″ per year: typical of a nearby star or brown dwarf.":
        "Yılda {}″ düzgün kayma: yakın bir yıldız veya kahverengi cüce için tipik.",
    "An asteroid would have left this patch of sky within days, yet it is still here after {}.":
        "Bir asteroit bu gökyüzü parçasını günler içinde terk ederdi, oysa {} sonra hâlâ burada.",
    "It shows the yearly parallax wobble (π = {} ± {}″), so it is close: about {} parsecs ({} light-years).":
        "Yıllık paralaks salınımı gösteriyor (π = {} ± {}″), yani yakın: yaklaşık {} parsek ({} ışık yılı).",
    "The path curves within weeks, as an orbit seen from the moving Earth does.":
        "Yol haftalar içinde eğriliyor; hareket eden Dünya'dan görülen bir yörünge gibi.",
    "The path bends, which a star's steady drift does not do.": "Yol bükülüyor; bir yıldızın düzgün kayması bunu yapmaz.",
    "The positions do not lie on one path: the scatter is {}× the measuring error.":
        "Konumlar tek bir yol üzerinde değil: saçılma ölçüm hatasının {} katı.",
    "All {} positions lie on one path within the errors (χ²ν = {}).":
        "{} konumun hepsi hatalar içinde tek bir yol üzerinde (χ²ν = {}).",
    "Mark {} is {}″ ({}σ) from where the other marks put it: leave it out and try again, or check it on its photo.":
        "{} işareti, diğer işaretlerin koyduğu yerden {}″ ({}σ) uzakta: çıkarıp yeniden deneyin ya da fotoğrafında "
        "denetleyin.",
    "It is near the ecliptic (latitude {}°), where most asteroids are.":
        "Ekliptiğe yakın (enlem {}°); asteroitlerin çoğu burada.",
    "It is far from the ecliptic (latitude {}°), where asteroids are rare.":
        "Ekliptikten uzak (enlem {}°); burada asteroitler seyrektir.",
    "No Gaia star at the marks: not a catalogued star (very cold brown dwarfs, asteroids and faint galaxies are "
    "missing from Gaia).":
        "İşaretlerde Gaia yıldızı yok: kataloglanmış bir yıldız değil (çok soğuk kahverengi cüceler, asteroitler ve "
        "sönük gökadalar Gaia'da yoktur).",
    "A Gaia DR3 star (G = {}) sits at the marks: a catalogued star, not an asteroid.":
        "İşaretlerde bir Gaia DR3 yıldızı (G = {}) var: kataloglanmış bir yıldız, asteroit değil.",
    "{} is in the catalogue ({}): a catalogued star, not an asteroid.":
        "{} katalogda var ({}): kataloglanmış bir yıldız, asteroit değil.",
    "{} gives its proper motion as {}″ per year.": "{} öz hareketini yılda {}″ olarak veriyor.",
    "{} gives it almost no motion ({}″ per year).": "{} ona neredeyse hiç hareket vermiyor (yılda {}″).",
    "The catalogue": "Katalog",
    "{} hours": "{} saat", "{} days": "{} gün", "{} years": "{} yıl",
    "{} minutes": "{} dakika", "{}″ per minute": "dakikada {}″", "{}″/min": "{}″/dk",
    "Predict ahead (minutes)": "İleriye tahmin (dakika)", "predicted position each step": "her adımda tahmini konum",
    "{}″ per hour": "saatte {}″", "{}″ per year": "yılda {}″",
    "On the photo of mark {}, something {}% as bright still sits where mark {} was: the marks may be on two different "
    "objects.":
        "{} işaretinin fotoğrafında, {} işaretinin olduğu yerde hâlâ %{} parlaklıkta bir şey duruyor: işaretler iki "
        "farklı cisim üzerinde olabilir.".replace("{} işaretinin fotoğrafında, {} işaretinin olduğu yerde hâlâ %{}",
                                                  "{0} işaretinin fotoğrafında, {2} işaretinin olduğu yerde hâlâ %{1}"),
    "The old spots are empty on the later photos: the object really left them.":
        "Eski noktalar sonraki fotoğraflarda boş: cisim gerçekten oradan ayrılmış.",
    "Mark {} is seen in only {}% of the single exposures behind its photo: a cosmic-ray hit or hot pixel, or "
    "something that moved during the exposures.":
        "{} işareti, fotoğrafının arkasındaki tek pozların yalnızca %{} kadarında görülüyor: bir kozmik ışın çarpması "
        "veya sıcak piksel ya da pozlar sırasında hareket eden bir şey.",
    "Its brightness changed: {}× brighter on {} than on {} ({}σ, allowing 20% for the different colours each photo "
    "averages). A variable star, a flare, or a different object.":
        "Parlaklığı değişti: {1}'de {2}'ye göre {0} kat daha parlak ({3}σ; her fotoğrafın farklı renkleri ortalaması "
        "için %20 pay bırakılarak). Değişen bir yıldız, bir parlama ya da farklı bir cisim.",
})

GLOSSARY_TR = """
- **RA, Dec** (sağ açıklık, dik açıklık): gökyüzünde bir konum; Dünya'daki boylam ve enlem gibi, derece (°)
  cinsinden. RA 0–360, Dec −90 (güney kutbu) ile +90 (kuzey kutbu) arasında değişir.
- **″ yay saniyesi, ′ yay dakikası**: küçük açılar. 60″ = 1′, 60′ = 1°. Dolunay yaklaşık 30′ (1.800″) genişliğindedir.
- **Piksel**: kameranın bir noktası. Bir SPHEREx pikseli gökyüzünün 6,15″'ını kapsar.
- **Bant / µm**: SPHEREx gözümüzün göremediği kızılötesi ışığı görür ve onu D1–D6 olmak üzere altı banda ayırır.
  µm (mikrometre) = metrenin milyonda biri; görünür ışık 0,4–0,7 µm'dir.
- **Dönem**: taramanın bu noktadan bir geçişi. SPHEREx yaklaşık altı ayda bir geri gelir.
- **Öz hareket**: bir yıldızın gökyüzündeki kendi yavaş kayması, yılda ″ cinsinden (″/yıl).
- **Paralaks**: yakın yıldızların küçük yıllık ileri-geri salınımı; çünkü hareket eden Dünya'dan izliyoruz.
- **MJy/sr, mJy**: parlaklık birimleri (gökyüzü için steradyan başına megajansky, yıldız için millijansky).
- **İşaretli pikseller**: NASA işlem hattının güvenilmez olarak işaretlediği pikseller (aşırı pozlama, kozmik ışın…).
- **σ (sigma)**: ölçüm belirsizliği. 3σ'dan büyük bir değişim büyük olasılıkla gerçektir.
- **Gaia, 2MASS, WISE, SIMBAD**: bir yıldız haritalama uydusu (ESA), iki eski kızılötesi gökyüzü taraması
  (~2000 ve 2010) ve adlandırılmış cisimlerin büyük bir kataloğu.
"""

_PH = re.compile(r"\{(\d*)\}")
_TEMPLATES: list[tuple[re.Pattern, str]] = []


def _compile():
    """Templates (keys with '{}') as regular expressions, the most specific (most literal text) first."""
    items = [(k, v) for k, v in TR.items() if "{}" in k]
    items.sort(key=lambda kv: -len(kv[0].replace("{}", "")))
    for k, v in items:
        parts = [re.escape(p) for p in k.split("{}")]
        _TEMPLATES.append((re.compile("^" + "(.+?)".join(parts) + "$", re.S), v))


def _fill(template: str, groups) -> str:
    """Put the caught values into the Turkish template: '{}' in order, or '{0}', '{1}' by number."""
    it = iter(groups)

    def put(m):
        return groups[int(m.group(1))] if m.group(1) else next(it, "")
    return _PH.sub(put, template)


def _one(text: str) -> str | None:
    """Exact or template translation of one piece (also without a final full stop), or None."""
    key = text.strip()
    if not key:
        return None
    out = _one_key(key)
    if out is None and key.endswith("."):
        inner = _one_key(key[:-1])
        out = None if inner is None else inner + "."
    return None if out is None else (text.replace(key, out) if key != text else out)


def _one_key(key: str) -> str | None:
    if key in TR:
        return TR[key]
    if not _TEMPLATES:
        _compile()
    for rx, tr_t in _TEMPLATES:
        m = rx.match(key)
        if m:
            return _fill(tr_t, [_group(g) for g in m.groups()])
    return None


def _group(g: str) -> str:
    """A value caught by a template: translated when it is itself known ('349 days', 'north'), else as it is.
    Each value is shorter than the text it came from, so this always ends."""
    k = g.strip()
    if not k or k[0].isdigit() and " " not in k:
        return g
    out = _one_key(k)
    return g if out is None else g.replace(k, out)


_BULLET = re.compile(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", re.S)
_SENT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z*(])")


def tr(text, lang: str = "tr"):
    """Turkish for text when lang is 'tr' (see the module notes); anything else is returned unchanged."""
    if lang != "tr" or not isinstance(text, str) or not text.strip() or text.lstrip().startswith("<"):
        return text
    if GLOSSARY_EN and text.strip() == GLOSSARY_EN.strip():
        return GLOSSARY_TR
    whole = _one(text)
    if whole is not None:
        return whole
    if "\n" in text.strip():   # line by line, keeping bullets and numbering
        out = []
        for ln in text.split("\n"):
            m = _BULLET.match(ln)
            head, body = (m.group(1), m.group(2)) if m else ("", ln)
            out.append(head + _piece(body))
        return "\n".join(out)
    return _piece(text)


def _piece(text: str) -> str:
    """One line: whole, else sentence by sentence, else by '; ' and ' · '."""
    if not text.strip():
        return text
    whole = _one(text)
    if whole is not None:
        return whole
    sents = _SENT.split(text)
    if len(sents) > 1:
        return " ".join(_piece(s) for s in sents)
    for sep in ("; ", " · "):
        if sep in text:
            return sep.join(_piece(p) for p in text.split(sep))
    return text


GLOSSARY_EN = ""   # set by the app (its English glossary), so the whole block is swapped at once


# --- Streamlit: send what is shown through tr() ---------------------------------------------------------
_INSTALLED = {"done": False}


def install(get_lang):
    """Wrap Streamlit's display functions once per process. get_lang() returns 'en' or 'tr' for this session."""
    if _INSTALLED["done"]:
        return
    _INSTALLED["done"] = True
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator as DG

    def T(x):
        return tr(x, get_lang())

    def fmt(kw):
        f = kw.get("format_func")
        kw["format_func"] = (lambda o, f=f: T(f(o))) if f is not None else (lambda o: T(str(o)))

    def simple(name, with_format=False):
        orig = getattr(DG, name)

        def wrapper(self, *args, **kw):
            if get_lang() == "tr":
                if args and isinstance(args[0], str):
                    args = (T(args[0]),) + args[1:]
                for k in ("body", "label", "help", "placeholder"):
                    if isinstance(kw.get(k), str):
                        kw[k] = T(kw[k])
                if with_format:
                    fmt(kw)
                if name == "tabs" and args and isinstance(args[0], (list, tuple)):
                    args = ([T(t) for t in args[0]],) + args[1:]
                if name == "metric" and len(args) > 1 and isinstance(args[1], str):
                    args = (args[0], T(args[1])) + args[2:]
            return orig(self, *args, **kw)
        wrapper.__wrapped__ = orig
        setattr(DG, name, wrapper)

    for n in ("markdown", "caption", "title", "header", "subheader", "info", "warning", "error", "success",
              "button", "download_button", "toggle", "checkbox", "text_input", "slider", "expander", "tabs", "metric"):
        simple(n)
    for n in ("selectbox", "multiselect", "radio", "segmented_control"):
        simple(n, with_format=True)

    orig_df = DG.dataframe

    def dataframe(self, data=None, *args, **kw):
        if get_lang() == "tr" and hasattr(data, "columns"):
            data = data.copy()
            for c in data.columns:
                if data[c].dtype == object and data[c].nunique() < 300:
                    data[c] = data[c].map(lambda v: T(v) if isinstance(v, str) else v)
            data.columns = [T(c) if isinstance(c, str) else c for c in data.columns]
            cc = kw.get("column_config")
            if cc:
                new = {}
                for k, v in cc.items():
                    if isinstance(v, dict):
                        v = dict(v)
                        for f in ("label", "help"):
                            if isinstance(v.get(f), str):
                                v[f] = T(v[f])
                    new[T(k)] = v
                kw["column_config"] = new
        return orig_df(self, data, *args, **kw)
    DG.dataframe = dataframe

    orig_pl = DG.plotly_chart

    def plotly_chart(self, fig=None, *args, **kw):
        if get_lang() == "tr" and hasattr(fig, "layout"):
            lay = fig.layout
            for ax in ("xaxis", "yaxis"):
                t = getattr(lay, ax).title
                if t is not None and t.text:
                    t.text = T(t.text)
            if lay.title is not None and lay.title.text:
                lay.title.text = T(lay.title.text)
            for a in lay.annotations or []:
                if a.text:
                    a.text = T(a.text)
            for sh in fig.data:
                if getattr(sh, "name", None):
                    sh.name = T(sh.name)
                if getattr(sh, "hovertemplate", None):
                    sh.hovertemplate = T(sh.hovertemplate)
                xs = getattr(sh, "x", None)
                if xs is not None and len(xs) and all(isinstance(v, str) for v in xs):
                    sh.x = [T(v) for v in xs]
        return orig_pl(self, fig, *args, **kw)
    DG.plotly_chart = plotly_chart

    for n in ("markdown", "caption", "title", "header", "subheader", "info", "warning", "error", "success", "button",
              "download_button", "toggle", "checkbox", "text_input", "slider", "expander", "tabs", "metric", "selectbox",
              "multiselect", "radio", "segmented_control", "dataframe", "plotly_chart"):
        setattr(st, n, getattr(st._main, n))   # st.markdown & co. are bound at import: bind them to the wrappers
