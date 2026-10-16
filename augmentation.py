"""
================================================================================
Problème #1 — Augmentation de données : Images ultrasonores
================================================================================

3 catégories d'images :
  - simple      : une seule fenêtre, aucune contrainte.
  - doppler     : deux fenêtres côte à côte de même largeur.
                  Gauche = niveaux de gris | Droite = même image + couleurs Doppler.
  - independent : deux fenêtres côte à côte de même largeur.
                  Gauche et droite = deux vues différentes et indépendantes.
 
Chaque transformation doit conserver la définition de la catégorie.
 
structure du code: 
  Section 1 : Fonctions génériques       : communes aux 3 catégories
  Section 2 : Fonctions photométriques   : gestion de la luminosité et des couleurs
  
  Section 3 : Fonctions d'augmentation par catégorie    : appliquer les fonctions de transformations aux images selon les catégories. 
  Section 4 : Pipeline principal         : lecture, augmentation, sauvegarde
 
structure des dossiers attendus: 
    dataset/
       simple/
       doppler/
       independent/
================================================================================
"""
 
import os
import cv2
import numpy as np
from pathlib import Path
 
 
# ==============================================================================
# Section 1 : Fonctions génériques 
# Transformations applicables à toutes les catégories: 
# flip vertical + rotation + zoom in + gaussian noise + speckle noise + salt pepper noise + gaussian blur + sharpen + elastic_deformation
# ==============================================================================
 
def flip_vertical(image):
    """
    Retourne l'image de haut en bas.
    valide pour les 3 catégories -> Le retournement haut/bas ne touche pas à la séparation gauche/droite (structure doppler préservée).
    """
    return cv2.flip(image, 0) # 0: flip horizontal , 1: vertical , -1: horizontal et vertical.
 
 
def rotate(image, angle):
    """
    pivote l'image autour du centre d'un angle donné (°) (< 0 sens horaire, > 0 anti-horaire) .
    Les bords laissés vides par la rotation sont remplis en noir.
    il faut que j'utilise des petits angles (exple: 5°, 10°, 15°), pour ne pas perdre la structure gauche/droit de la catégorie doppler. 
    """
    h, w = image.shape[:2] # height & weight: image.shape : (hauteur, largeur) pour le niveaux gris, et (hauteur, largeur, canaux) pour image couleur
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0) # scale 1: ne pas changer la taille de l image après rotation. 
    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) # BORDER_CONSTANT: remplacer les zones vides par un pixel constant. borderValue=0 : couleur noire. 
 
 
def zoom_in(image, factor=1.2):
    """
    zoom simulé: recadre la zone centrale de l'image : on coupe de façon sym des deux côtés. 
    puis en la redimensionnant à la taille d'origine.
    exemple: factor=1.2 -> on garde 1/1.2 ≈ 83% du centre de l'image.
    """
    h, w = image.shape[:2]
    new_h = int(h / factor)
    new_w = int(w / factor)
    top  = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = image[top : top + new_h, left : left + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
 
 
def add_gaussian_noise(image, std=15): # exemple std = 15 -> détermine le noveau du bruit (bruit modéré)
    """
    simule le bruit électronique d'un appareil
    bruit gaussien : j ajoute à chaque pixel une valeur aléatoire -> image granuleuse
    tirée d'une distribution normale (moyenne=0, écart-type=std). (std faible: bruit léger, std élevé: bruit fort)
    """
    noise = np.random.normal(0, std, image.shape).astype(np.float32) # donne une image (matrice) du bruit qu'on ajoutera à l'image réelle. .astype(np.float32): conversion en float pour faire l'addition
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
 
 
def add_speckle_noise(image, intensity=0.05):
    """
    Ajoute du bruit "speckle" (bruit multiplicatif).
    bruit gaussien : ajoute une valeur fixe, le speckle multiplie chaque pixel par un facteur légèrement aléatoire : pixel_sortie = pixel_entrée x (1 + valeur aléatoire)
    """
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    speckled = image.astype(np.float32) * (1.0 + noise)
    return np.clip(speckled, 0, 255).astype(np.uint8)
 
 
def add_salt_pepper_noise(image, prob=0.01):
    """
    bruit sel et poivre: certains pixels sont mis à blanc (sel=255) ou à noir (poivre=0) de façon aléatoire.
    prob=0.01 → 1% des pixels sont affectés.
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    salt_mask   = np.random.rand(h, w) < prob
    pepper_mask = np.random.rand(h, w) < prob
    noisy[salt_mask]   = 255
    noisy[pepper_mask] = 0
    return noisy
 
 
def gaussian_blur(image, kernel_size=(3, 3)):
    """
    Applique un flou gaussien : chaque pixel est remplacé par la moyenne pondérée de ses voisins (pondération suit la loi normal, chaque pixel a son poid). L'image devient floue.
    exple: kernel_size=(3,3) -> flou léger . ça doit être impair pour avoir un pixel au centre. 
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=0) # sigmaX = 0 : auto calcul du std a partir du kernel size
 
 
def sharpen(image):
    """
    Accentue les contours et les détails de l'image (effet inverse du flou).
    On utilise un filtre de convolution qui renforce les différences entre un pixel et ses voisins.
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32) # filtre appliqué à chaque pixel : nouveau_pixel = 5 × pixel central - voisins (haut, bas, gauche, droite) -> accentue les contours. 
    sharpened = cv2.filter2D(image, -1, kernel) # -1 : garder le même type de l'image
    return np.clip(sharpened, 0, 255).astype(np.uint8)
 
 
def elastic_deformation(image, alpha=30, sigma=4):
    """
    Déforme l'image de façon élastique.
    alpha : intensité de la déformation (plus grand = plus déformé)
    sigma : douceur de la déformation (plus grand = plus lisse)
    le médecin appuie plus ou moins fort avec la sonde ça déforme légèrement les tissus ..
    """
    h, w = image.shape[:2]
    # On génère deux champs de déplacement aléatoires (un horizontal, un vertical)
    # et on les lisse avec un flou gaussien pour obtenir une déformation douce
    dx = cv2.GaussianBlur( np.random.randn(h, w).astype(np.float32), (0, 0), sigma ) * alpha # np.random.randn(h, w): matrice aléatoire qui suit une loi normal.(0, 0):kernel automatique (dépend de sigma). sigma: controle de lissage
    dy = cv2.GaussianBlur( np.random.randn(h, w).astype(np.float32), (0, 0), sigma ) * alpha # déplacement vertical . alpha: force de la déformation 
    # On crée une grille de coordonnées et on lui applique le déplacement
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h)) # np.arange(w): array qui part de 0 à w-1 / x_grid : coord x de chaque pixel. 
    map_x = (x_grid + dx).astype(np.float32) # nouvelles positions des pixels
    map_y = (y_grid + dy).astype(np.float32)
 
    return cv2.remap(image, map_x, map_y,interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) # remap: fonction qui déforme l’image , BORDER_REFLECT : OpenCV réflechit l’image dans les bords
 
 
def is_grayscale(image):
    """
    Détecte si une image à 3 canaux est en réalité en niveaux de gris.
    cv2.imread renvoie toujours 3 canaux, même pour un fichier gris :
    on compare donc les canaux entre eux.
    """
    if len(image.shape) == 2:
        return True
    b, g, r = cv2.split(image)
    return np.array_equal(b, g) and np.array_equal(g, r)


# ==============================================================================
# Section 2 — Fonctions photométriques
# Modifications de luminosité, contraste, et couleur.
# ==============================================================================
 
def adjust_brightness(image, factor):
    """
    Modifier la luminosité en multipliant chaque pixel par "factor".
    factor < 1 -> image plus sombre  (ex : 0.7 = 30% plus sombre)
    factor > 1 -> image plus claire  (ex : 1.3 = 30% plus claire)
    """
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)
 

def adjust_contrast(image, alpha):
    mean = np.mean(image)
    result = alpha * (image.astype(np.float32) - mean) + mean
    return np.clip(result, 0, 255).astype(np.uint8)
 
 
def gamma_correction(image, gamma):
    """
    Applique une correction gamma (nn linéaire - plus réaliste p/r à luminosité) : pixel_sortie = (pixel_entrée / 255)^(1/gamma) x 255
    gamma < 1.0 → image plus claire (éclaircit les zones sombres)
    gamma > 1.0 → image plus sombre (assombrit les zones claires)
    # appliqué sur les images en niveau de gris
    """
    inv_gamma = 1.0 / gamma
    # Table de correspondance : pour chaque valeur de 0 à 255, on précalcule la sortie
    lut = np.array( [(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8) 
    return cv2.LUT(image, lut)
 
 
def adjust_hsv(image, s_factor=1.0, v_factor=1.0):
    """
    Modifie la saturation (S) et/ou la luminosité (V) d'une image colorée,
    ne pas toucher à la teinte (H = la couleur elle-même).
    s_factor : facteur pour la saturation  (ex : 0.7 = moins saturé)
    v_factor : facteur pour la luminosité  (ex : 1.3 = plus clair)
    """
    # convertir l'image de BGR (format OpenCV) vers HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32) #en BGR, modifier la luminosité peut changer la couleur. en HSV, je peux modifier la lumière sans casser la couleur
 
    # modifier S et V 
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)  # Saturation  # [h, w, canaux] = [:, :, 1]: canal 1 de HSV qui est S.
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_factor, 0, 255)  # Luminosité
 
    # reconvertir vers BGR
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
 
 
# ==============================================================================
# Section 3 : Fonctions d'augmentation par catégorie 
# ==============================================================================
 
def apply_common_transforms(image):
    """
    Applique toutes les transformations communes aux 3 catégories.
    Retourne une liste de tuples : (nom, image_transformée)

    """
    results = []
 
    # géo
    results.append(("flip_vertical",      flip_vertical(image)))
    results.append(("rotate_m15",         rotate(image, -15)))
    results.append(("rotate_m10",         rotate(image, -10)))
    results.append(("rotate_m5",          rotate(image,  -5)))
    results.append(("rotate_p5",          rotate(image,   5)))
    results.append(("rotate_p10",         rotate(image,  10)))
    results.append(("rotate_p15",         rotate(image,  15)))
    results.append(("zoom_in",            zoom_in(image, factor=1.2))) # 83%
 
    # bruit
    results.append(("gaussian_noise",     add_gaussian_noise(image, std=15))) # bruit additif
    results.append(("speckle_noise",      add_speckle_noise(image, intensity=0.05))) # bruit multiplicatif
    results.append(("salt_pepper",        add_salt_pepper_noise(image, prob=0.01)))
 
    # filtre
    results.append(("blur",               gaussian_blur(image, (3, 3))))
    results.append(("sharpen",            sharpen(image))) # image plus accentué
 
    # déformation
    results.append(("elastic",            elastic_deformation(image)))
 
    return results 
 


def apply_half_transforms(half):
    """
    Transformations valides sur une moitié d'image.
    On exclut rotation, zoom, déformation élastique car elles cassent la cohérence visuelle quand on recolle les deux moitiés : voir en bas 
    """
    return [
        ("noise_gauss",   add_gaussian_noise(half, std=15)),
        ("noise_speckle", add_speckle_noise(half, intensity=0.05)),
        ("noise_sp",      add_salt_pepper_noise(half, prob=0.01)),
        ("blur",          gaussian_blur(half, (3, 3))),
        ("sharpen",       sharpen(half)),
        ("flip_v",        flip_vertical(half)),
    ]
 


def augment_simple(image):
    results = []

    # --- transformations communes ---
    results += apply_common_transforms(image)

    # --- flip horizontal ---
    results.append(("flip_horizontal", cv2.flip(image, 1)))

    # --- Photométrie --- 
    is_color = not is_grayscale(image)

    if is_color:
        results.append(("val_dark",    adjust_hsv(image, v_factor=0.7)))
        results.append(("val_bright",  adjust_hsv(image, v_factor=1.3)))
        results.append(("sat_low",     adjust_hsv(image, s_factor=0.6)))
        results.append(("sat_high",    adjust_hsv(image, s_factor=1.4)))
    else:
        results.append(("brightness_dark",   adjust_brightness(image, 0.7)))
        results.append(("brightness_bright", adjust_brightness(image, 1.3)))
        results.append(("contrast_low",      adjust_contrast(image, 0.8)))
        results.append(("contrast_high",     adjust_contrast(image, 1.3)))
        results.append(("gamma_dark",        gamma_correction(image, 2.0)))
        results.append(("gamma_bright",      gamma_correction(image, 0.5)))
    
    return results




def augment_doppler(image):
    results = []
    h, w = image.shape[:2]
    mid = w // 2  # index de la colonne centrale = ligne de séparation
 
    left  = image[:, :mid]   # moitié gauche (niveaux de gris)
    right = image[:, mid:]   # moitié droite (couleurs Doppler)
 
    # --- Transformations communes (14) ---
    results += apply_common_transforms(image)
 
    # --- Flip horizontal par moitié ---
    # flip gauche : on retourne la moitié grise, on recolle avec droite intacte
    left_flipped = cv2.flip(left, 1)
    results.append(("flip_left_half",
                    np.concatenate([left_flipped, right], axis=1)))
 
    # flip droite : on retourne la moitié colorée, on recolle avec gauche intacte
    right_flipped = cv2.flip(right, 1)
    results.append(("flip_right_half",
                    np.concatenate([left, right_flipped], axis=1)))
    
    # ajouter des images ou les deux côtés sont flippés.
    results.append(("flip_both_halves",
                    np.concatenate([left_flipped, right_flipped], axis=1)))
 
 
    # --- Photométrie ---
    left_transforms = [ # left : mode gris
        ("normal",            left),   # inchangé
        ("brightness_dark",   adjust_brightness(left, 0.7)),
        ("brightness_bright", adjust_brightness(left, 1.3)),
        ("contrast_low",      adjust_contrast(left, 0.8)),
        ("contrast_high",     adjust_contrast(left, 1.3)),
        ("gamma_dark",        gamma_correction(left, 2.0)),
        ("gamma_bright",      gamma_correction(left, 0.5)),  ] + [(name, t) for name, t in apply_half_transforms(left)]

    right_transforms = [ # doppler : code couleur
        ("normal",     right),  # inchangé
        ("sat_low",    adjust_hsv(right, s_factor=0.6)),
        ("sat_high",   adjust_hsv(right, s_factor=1.4)),
        ("val_dark",   adjust_hsv(right, v_factor=0.7)),
        ("val_bright", adjust_hsv(right, v_factor=1.3)),  ] + [(name, t) for name, t in apply_half_transforms(right)]

    # combiner toutes les transformations
    for left_name, left_t in left_transforms:
        for right_name, right_t in right_transforms:
            if left_name == "normal" and right_name == "normal":    # pour éviter un doublon normal-normal.
                continue
            name = f"{left_name}_{right_name}"
            results.append((name, np.concatenate([left_t, right_t], axis=1)))
    
    return results 



def augment_independent(image):
    results = []
    h, w = image.shape[:2]
    mid = w // 2

    left  = image[:, :mid]
    right = image[:, mid:]

    # --- transformations communes ---
    results += apply_common_transforms(image)

    # --- flip horizontal (image entière) ---
    results.append(("flip_horizontal", cv2.flip(image, 1)))

    # --- Swap des moitiés ---
    results.append(("swap_halves", np.concatenate([right, left], axis=1)))


    is_color_left  = not is_grayscale(left)
    is_color_right = not is_grayscale(right)
    # --- left_transforms selon couleur ---
    if is_color_left:
        left_transforms = [
            ("normal",         left),
            ("flip_left_half", cv2.flip(left, 1)),
            ("val_dark",       adjust_hsv(left, v_factor=0.7)),
            ("val_bright",     adjust_hsv(left, v_factor=1.3)),
            ("sat_low",        adjust_hsv(left, s_factor=0.6)),
            ("sat_high",       adjust_hsv(left, s_factor=1.4)),
        ] + [(name, t) for name, t in apply_half_transforms(left)]
    else:
        left_transforms = [
            ("normal",            left),
            ("flip_left_half",    cv2.flip(left, 1)),
            ("brightness_dark",   adjust_brightness(left, 0.7)),
            ("brightness_bright", adjust_brightness(left, 1.3)),
            ("contrast_low",      adjust_contrast(left, 0.8)),
            ("contrast_high",     adjust_contrast(left, 1.3)),
            ("gamma_dark",        gamma_correction(left, 2.0)),
            ("gamma_bright",      gamma_correction(left, 0.5)),
        ] + [(name, t) for name, t in apply_half_transforms(left)]

    # --- right_transforms selon couleur ---
    if is_color_right:
        right_transforms = [
            ("normal",          right),
            ("flip_right_half", cv2.flip(right, 1)),
            ("val_dark",        adjust_hsv(right, v_factor=0.7)),
            ("val_bright",      adjust_hsv(right, v_factor=1.3)),
            ("sat_low",         adjust_hsv(right, s_factor=0.6)),
            ("sat_high",        adjust_hsv(right, s_factor=1.4)),
        ] + [(name, t) for name, t in apply_half_transforms(right)]
    else:
        right_transforms = [
            ("normal",            right),
            ("flip_right_half",   cv2.flip(right, 1)),
            ("brightness_dark",   adjust_brightness(right, 0.7)),
            ("brightness_bright", adjust_brightness(right, 1.3)),
            ("contrast_low",      adjust_contrast(right, 0.8)),
            ("contrast_high",     adjust_contrast(right, 1.3)),
            ("gamma_dark",        gamma_correction(right, 2.0)),
            ("gamma_bright",      gamma_correction(right, 0.5)),
        ] + [(name, t) for name, t in apply_half_transforms(right)]

    # --- Combinaisons ---
    for left_name, left_t in left_transforms:
        for right_name, right_t in right_transforms:
            if left_name == "normal" and right_name == "normal":
                continue
            name = f"{left_name}_{right_name}"
            results.append((name, np.concatenate([left_t, right_t], axis=1)))

    return results




# ==============================================================================
# Section 4 — Pipeline principal : Lecture des images → augmentation → sauvegarde
# ==============================================================================
 
def augment_dataset(input_dir, output_dir, max_per_image=None, seed=0):
    """
    Fonction principale
    Pour chaque catégorie :
      1. Lit toutes les images du dossier correspondant
      2. Applique la fonction d'augmentation adaptée
      3. Sauvegarde l'image originale + toutes les images augmentées
    """
 
    # Association : nom du dossier -> fonction d'augmentation
    category_config = {
        "simple":      augment_simple,
        "doppler":     augment_doppler,
        "independent": augment_independent,
    }
 
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("  AUGMENTATION DU DATASET — IMAGES ULTRASONORES")
    print("=" * 60)
 
    total_original  = 0
    total_generated = 0
 
    for category, augment_fn in category_config.items():
 
        input_path  = os.path.join(input_dir,  category)
        output_path = os.path.join(output_dir, category)
        os.makedirs(output_path, exist_ok=True)
 
        # Récupérer toutes les images du dossier
        image_files = (
            list(Path(input_path).glob("*.png"))  +
            list(Path(input_path).glob("*.jpg"))  +
            list(Path(input_path).glob("*.jpeg"))
        )
 
        if not image_files:
            print(f"\n  [{category}]. Aucune image trouvée dans : {input_path}")
            continue
 
        count_generated = 0
 
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                print(f" Impossible de lire : {img_file.name}")
                continue
 
            base_name = img_file.stem  # nom du fichier sans extension
 
            # Sauvegarder l'image originale
            cv2.imwrite(
                os.path.join(output_path, f"{base_name}_original.png"),
                image
            )
 
            # Appliquer les augmentations et sauvegarder chaque résultat
            augmented_list = augment_fn(image)

            # Les catégories doppler et independent produisent un grand nombre
            # de combinaisons : on peut en tirer un sous-ensemble aléatoire.
            if max_per_image is not None and len(augmented_list) > max_per_image:
                idx = rng.choice(len(augmented_list), size=max_per_image, replace=False)
                augmented_list = [augmented_list[i] for i in sorted(idx)]

            for aug_name, aug_image in augmented_list:
                out_path = os.path.join(output_path, f"{base_name}_{aug_name}.png")
                cv2.imwrite(out_path, aug_image)
                count_generated += 1
 
        n_original = len(image_files)
        n_total    = n_original + count_generated
        total_original  += n_original
        total_generated += count_generated
 
        print(f"\n  [{category}]")
        print(f"    Images originales          : {n_original}")
        print(f"    Transformations par image  : {count_generated // n_original if n_original > 0 else 0}")
        print(f"    Images générées            : {count_generated}")
        print(f"    Total dans le dossier      : {n_total}")
 
    print("\n" + "=" * 60)
    print(f"  RÉSUMÉ FINAL")
    print(f"    Images originales  : {total_original}")
    print(f"    Images générées    : {total_generated}")
    print(f"    TOTAL dataset      : {total_original + total_generated}")
    print("=" * 60)
 
 



if __name__ == "__main__":
 
    INPUT_DIR  = "dataset"            # dossier source
    OUTPUT_DIR = "dataset_augmented"  # dossier de sortie
 
    # max_per_image limite le nombre d'images générées par image source.
    # Mettre None pour générer toutes les combinaisons possibles.
    augment_dataset(INPUT_DIR, OUTPUT_DIR, max_per_image=30)