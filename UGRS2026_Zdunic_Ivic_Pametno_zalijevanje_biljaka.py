"""
classifier.py
--------------
HSV-bazirana klasifikacija zdravlja biljke iz slike, sa dodatnim
"leaf confidence" korakom koji mora potvrditi da je detektirani objekt
stvarno dovoljno slican listu prije nego sto se odredi Healthy/Dry/Critical.
"""

import cv2
import numpy as np

#OpenCV hue ide 0-179. Ovo su "raspon boja" koje uopće mogu biti dio biljke.
GREEN_HUE_RANGE  = (25, 110) #probat od 40 do 70  bilo: 25, 110
YELLOW_HUE_RANGE = (10, 25)  #probat od 15 do 30  bilo: 10, 25

#Value = Saturation = max 255
#Maska siluete
SILHOUETTE_MIN_SATURATION = 10  # ispod ovoga je preblijedo/sivo da bude list  bilo 20
SILHOUETTE_MIN_VALUE = 30       # ispod ovoga je pretamno da razaznamo boju
SILHOUETTE_MAX_VALUE = 180      # iznad ovoga je preblizu bijele (zid/strop)

#Maska
MIN_SATURATION = 20   
MIN_VALUE      = 30   
MAX_VALUE      = 255  

#Tamno/smeđe razlikovanje
DARK_VALUE_THRESHOLD = 60        # nekroza mora biti tamna
DARK_SATURATION_THRESHOLD = 55   # i obezbojena
BROWN_MAX_VALUE = 110            # unutar zutog huea: ispod ovoga = smede, iznad = zuto

#Odluka o zdravstvenom stanju
DARK_CRITICAL_THRESHOLD = 0.15   # >15% (tamno+smedje) piksela -> Critical
YELLOW_DRY_THRESHOLD    = 0.25   # >25% zutih piksela -> Dry

#Min povrsina lista
MIN_LEAF_AREA_RATIO = 0.25       

#Popunjavanje sitnih rupa (11×11)
CLOSING_KERNEL_SIZE = 11

# Namjerno nisu ekstremno strogi: cilj je prvenstveno sprijeciti da
# stolnjak/zid/strop postanu Healthy samo zato sto imaju slican hue.
MIN_LEAF_COLOR_RATIO = 0.75       # barem (bilo 35%) siluete mora biti green/yellow
MIN_LEAF_HUE_STD = 15.0            # gotovo jednobojna povrsina -> Unknown, provjerit jel te vrijednosti valjaju
MIN_LEAF_SAT_STD = 10.0
MIN_LEAF_VALUE_STD = 15.0
MIN_LEAF_SOLIDITY = 0.55
MIN_LEAF_EXTENT = 0.18 
MAX_LEAF_ASPECT_RATIO = 8.0

# Ako je kandidat ogromna ploha koja dodiruje rub slike - vrlo cesto pozadina (stol, zid, zavjesa)
BORDER_TOUCH_PENALTY = True


def _find_leaf_candidate(hsv: np.ndarray, img_shape: tuple):
    """Vraca (silueta, kontura, coarse_mask) ili (None, None, None)."""
    h, s, v = cv2.split(hsv)

    #Rezultat coarse_mask je 2D niz bool vrijednosti
    #numpy zahtijeva &/| za array operacije
    coarse_mask = (((h >= GREEN_HUE_RANGE[0]) & (h <= GREEN_HUE_RANGE[1])) | ((h >= YELLOW_HUE_RANGE[0]) & (h < YELLOW_HUE_RANGE[1]))) \
      & (s >= SILHOUETTE_MIN_SATURATION) \
      & (v >= SILHOUETTE_MIN_VALUE) \
      & (v <= SILHOUETTE_MAX_VALUE)

    #pretvara True/False u 1/0, a * 255 to skalira u 255/0
    coarse_u8 = coarse_mask.astype(np.uint8) * 255 

    kernel = np.ones((CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE), np.uint8)
    closed = cv2.morphologyEx(coarse_u8, cv2.MORPH_CLOSE, kernel)   #sve rupe/pukotine unutar oblika manje od kernel-veličine se popune
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   #uklanja sitni šum — izolirane bijele mrljice raspršene po pozadini nestaju, dok veće, stvarne plohe (list) ostaju netaknute.
    #Zatvaranje pa otvaranje - Prvo se popune rupe unutar lista (zatvaranje), zatim se očisti sitni šum oko lista (otvaranje) — da otvaranje ne izgriza rupe koje smo upravo popunili unutar lista.

    #findContours pretražuje binarnu sliku (opened) i vraća popis svih odvojenih bijelih "otoka" kao liste točaka koje opisuju njihov rub. 
    #RETR_EXTERNAL — zanimaju nas samo vanjski rubovi (ako bi unutar bijele plohe postojala potpuno okružena crna rupa, ignoriramo tu unutarnju konturu — želimo samo vanjski obris cijelog oblika).
    #CHAIN_APPROX_SIMPLE — način kompresije točaka konture. Umjesto da sprema svaku pojedinu graničnu točku (puno redundantnih točaka), sprema samo ključne točke (npr. krajeve ravnih segmenata), štedeći memoriju.
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if not contours:                                                                   
        return None, None, None

    
    largest = max(contours, key=cv2.contourArea) #Odabir najveće konture 
    area = cv2.contourArea(largest)              #Izračun površine
    total_area = img_shape[0] * img_shape[1] 

    if area / total_area < MIN_LEAF_AREA_RATIO:  #provjera površine
        return None, None, None

    #largest - samo popis točaka na rubu oblika, stvara granicu 
    #FILLED - boja sve piksele unutar granice
    #vrijednost 255 - predstavlja True, odnosno svaka druga vrijednost osim 0
    #return: gotova puna maska, sama kontura i coarse_mask(treba za coarse_inside_ratio izračun u _leaf_confidence)
    silhouette = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(silhouette, [largest], -1, 255, thickness=cv2.FILLED)
    return silhouette.astype(bool), largest, coarse_mask


def _leaf_confidence(hsv, silhouette, contour, coarse_mask, img_shape):
    """Vraca score 0..1 i dijagnostiku."""
    h, s, v = cv2.split(hsv)
    ys, xs = np.where(silhouette) #vraća koordinate svih piksela koji su True u maski. 
    if len(xs) == 0:
        return 0.0, {"reason": "empty_silhouette"}

    leaf_count = len(xs) #koliko ukupno ima True piksela u maski
    total = img_shape[0] * img_shape[1] 
    area_ratio = leaf_count / total

    #Provjera boje - 2D bool niz (zelena/žuta boja - True)
    #[silhouette] - numpy boolean indeksiranje - uzimaju se samo oni zeleni/žuti pikseli koji su untar siluete
    green = ((h >= GREEN_HUE_RANGE[0]) & (h <= GREEN_HUE_RANGE[1]) &
             (s >= MIN_SATURATION) & (v >= MIN_VALUE))[silhouette]
    yellow = ((h >= YELLOW_HUE_RANGE[0]) & (h < YELLOW_HUE_RANGE[1]) &
              (s >= MIN_SATURATION) & (v >= MIN_VALUE))[silhouette]
    color_ratio = float(np.count_nonzero(green | yellow)) / leaf_count #postotak koliko je piksela unutar siulete zeleno ili žuto 

    #Standardne devijacije (nizak std - vrijednosti slične, ujdednačena vrijednost, visok std - vrijednosti variraju, šarolika površina)
    h_std = float(np.std(h[silhouette])) #boolean indekisranje izravno na hue nizu, rezultat je niz stvarnih hue vrijednosti svih piksela unutar lista
    s_std = float(np.std(s[silhouette])) 
    v_std = float(np.std(v[silhouette]))


    x, y, w, hbox = cv2.boundingRect(contour) #koordinate gornjeg lijevog kuta najmanjeg pravokutnika koji cijelu konturu obuhvaća, w je širina tog pravokutnika, hbox je visina
    aspect = max(w, hbox) / max(1, min(w, hbox)) #omjer dužina/širina bounding pravokutnika (Ukoliko je nešto izrazito usko, a dugačko, vjerojatno nije list)
    contour_area = cv2.contourArea(contour) #Površina (u pikselima) unutar same konture
    perimeter = cv2.arcLength(contour, True) #Duljina ruba konture (opseg)
    hull_area = cv2.contourArea(cv2.convexHull(contour)) #izračun konveksne ljuske
    bbox_area = max(1, w * hbox)  #Površina bounding pravokutnika
    solidity = contour_area / max(1.0, hull_area) #prava površina konture podijeljena s površinom njene konveksne ljuske, List (relativno gladak, konveksan oblik) ima solidity blizu 1.0
    extent = contour_area / bbox_area #prava površina konture podijeljena s površinom njenog bounding pravokutnika 
    circularity = (4.0 * np.pi * contour_area / (perimeter * perimeter)) if perimeter > 0 else 0.0 

    # Koliki dio kandidata zaista pripada originalnoj coarse maski
    coarse_inside_ratio = float(np.count_nonzero(coarse_mask[silhouette])) / leaf_count

    touches_border = x <= 0 or y <= 0 or (x + w) >= img_shape[1] - 1 or (y + hbox) >= img_shape[0] - 1

    #Score
    score = 0.0
    reasons = []

    if area_ratio >= MIN_LEAF_AREA_RATIO:
        score += 1.0
    if color_ratio >= MIN_LEAF_COLOR_RATIO:
        score += 1.5
    if coarse_inside_ratio >= 0.45:
        score += 1.0
    if h_std >= MIN_LEAF_HUE_STD:
        score += 0.75
    if s_std >= MIN_LEAF_SAT_STD:
        score += 0.75
    if v_std >= MIN_LEAF_VALUE_STD:
        score += 0.75
    if solidity >= MIN_LEAF_SOLIDITY:
        score += 0.5
    if extent >= MIN_LEAF_EXTENT:
        score += 0.5
    if aspect <= MAX_LEAF_ASPECT_RATIO:
        score += 0.5

    if touches_border and BORDER_TOUCH_PENALTY:
        score -= 1.5
        reasons.append("touches_border")

    # Skoro jednobojne velike povrsine nisu list.
    if h_std < MIN_LEAF_HUE_STD and s_std < MIN_LEAF_SAT_STD and v_std < MIN_LEAF_VALUE_STD:
        score -= 2.0
        reasons.append("too_uniform")

    # Ne treba zahtijevati zeleno jer Dry/Critical mogu imati malo zelenog,
    # ali mora postojati dovoljno pixela koji su u plant-like hue maski.
    if color_ratio < MIN_LEAF_COLOR_RATIO:
        reasons.append("too_little_leaf_color")

    # 4.0 je namjerno sigurnosni prag: radije Unknown nego pogresan Healthy.
    confidence = max(0.0, min(1.0, score / 6.5))
    is_leaf = score >= 4.0

    debug = {
        "leaf_confidence": round(confidence, 3),
        "leaf_score": round(score, 2),
        "leaf_area_ratio": round(area_ratio, 3),
        "leaf_color_ratio": round(color_ratio, 3),
        "coarse_inside_ratio": round(coarse_inside_ratio, 3),
        "h_std": round(h_std, 2),
        "s_std": round(s_std, 2),
        "v_std": round(v_std, 2),
        "solidity": round(float(solidity), 3),
        "extent": round(float(extent), 3),
        "aspect_ratio": round(float(aspect), 3),
        "circularity": round(float(circularity), 3),
        "touches_border": bool(touches_border),
        "is_leaf": bool(is_leaf),
        "reason": ",".join(reasons) if reasons else "ok",
    }
    return confidence, debug


def classify_plant_health(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Ne mogu ucitati sliku: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total_pixels = img.shape[0] * img.shape[1]

    #Pronalazak kandidata
    silhouette, contour, coarse_mask = _find_leaf_candidate(hsv, img.shape)

    
    if silhouette is None:
        return {
            "classification": "Unknown",
            "green_ratio": 0.0,
            "yellow_ratio": 0.0,
            "brown_ratio": 0.0,
            "dark_ratio": 0.0,
            "leaf_pixel_ratio": 0.0,
            "leaf_confidence": 0.0,
            "leaf_score": 0.0,
            "unknown_reason": "no_candidate",
        }

    #Provjera 
    leaf_confidence, leaf_debug = _leaf_confidence(
        hsv, silhouette, contour, coarse_mask, img.shape
    )

    #PRIVREMENO
    if not leaf_debug["is_leaf"]:
        print("\n=== UNKNOWN DIAGNOSTICS ===")
        print(leaf_debug)
    
        return {
            "classification": "Unknown",
            "green_ratio": 0.0,
            "yellow_ratio": 0.0,
            "brown_ratio": 0.0,
            "dark_ratio": 0.0,
            "leaf_pixel_ratio": round(float(np.count_nonzero(silhouette) / total_pixels), 3),
            **leaf_debug,
            "unknown_reason": leaf_debug["reason"],
        }
    ##


    if not leaf_debug["is_leaf"]:
        return {
            "classification": "Unknown",
            "green_ratio": 0.0,
            "yellow_ratio": 0.0,
            "brown_ratio": 0.0,
            "dark_ratio": 0.0,
            "leaf_pixel_ratio": round(float(np.count_nonzero(silhouette) / total_pixels), 3),
            **leaf_debug,
            "unknown_reason": leaf_debug["reason"],
        }

    #Klasifikacija 
    leaf_pixel_count = int(np.count_nonzero(silhouette))

    #2D bool niz veličine cijele slike — True samo gdje je piksel i unutar lista i zelene boje
    green_mask = silhouette & (h >= GREEN_HUE_RANGE[0]) & (h <= GREEN_HUE_RANGE[1]) & (s >= MIN_SATURATION) & (v >= MIN_VALUE) & (v <= MAX_VALUE)

    yellow_hue_mask = silhouette & (h >= YELLOW_HUE_RANGE[0]) & (h < YELLOW_HUE_RANGE[1]) & (s >= MIN_SATURATION) & (v >= MIN_VALUE) & (v <= MAX_VALUE)

    #Razdvajanje žutog tona na svjetliji žuti i smeđi   
    yellow_mask = yellow_hue_mask & (v >= BROWN_MAX_VALUE)
    brown_mask = yellow_hue_mask & (v < BROWN_MAX_VALUE)
    dark_mask = silhouette & (v < DARK_VALUE_THRESHOLD) & (s < DARK_SATURATION_THRESHOLD)

    #Postatak piksela svake boje 
    green_ratio = np.count_nonzero(green_mask) / leaf_pixel_count
    yellow_ratio = np.count_nonzero(yellow_mask) / leaf_pixel_count
    brown_ratio = np.count_nonzero(brown_mask) / leaf_pixel_count
    dark_ratio = np.count_nonzero(dark_mask) / leaf_pixel_count
    leaf_pixel_ratio = leaf_pixel_count / total_pixels

    #Klasifikacija
    if (dark_ratio + brown_ratio) > DARK_CRITICAL_THRESHOLD:
        classification = "Critical"
    elif yellow_ratio > YELLOW_DRY_THRESHOLD:
        classification = "Dry"
    else:
        classification = "Healthy"

    return {
        "classification": classification,
        "green_ratio": round(float(green_ratio), 3),
        "yellow_ratio": round(float(yellow_ratio), 3),
        "brown_ratio": round(float(brown_ratio), 3),
        "dark_ratio": round(float(dark_ratio), 3),
        "leaf_pixel_ratio": round(float(leaf_pixel_ratio), 3),
        **leaf_debug,
    }


def debug_save_silhouette(image_path: str, output_path: str = "silhouette_debug.jpg") -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Ne mogu ucitati sliku: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    silhouette, contour, _ = _find_leaf_candidate(hsv, img.shape)

    overlay = img.copy()
    if silhouette is None:
        print("Silueta kandidata NIJE pronadena.")
    else:
        _, debug = _leaf_confidence(hsv, silhouette, contour, _, img.shape)
        overlay[~silhouette] = (overlay[~silhouette] * 0.25).astype(np.uint8)
        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 3)
        cv2.putText(
            overlay,
            f"leaf={debug['is_leaf']} score={debug['leaf_score']:.1f} conf={debug['leaf_confidence']:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )
        #cv2.putText(
        #    overlay,
        #    f"reason={debug['reason']}",
        #    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        #)

    cv2.imwrite(output_path, overlay)
    print(f"Spremljeno: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Upotreba: python classifier.py <putanja_do_slike>")
        sys.exit(1)

    result = classify_plant_health(sys.argv[1])
    print(result)
    debug_save_silhouette(sys.argv[1], "silhouette_debug.jpg")
