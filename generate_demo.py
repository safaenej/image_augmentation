"""
================================================================================
Génération d'un jeu de démonstration synthétique
================================================================================

Aucune donnée médicale réelle n'est utilisée dans ce dépôt.

Ce script fabrique, avec NumPy et OpenCV uniquement, trois images qui
reproduisent la structure des trois catégories traitées par augmentation.py :

  - simple      : une fenêtre unique, en niveaux de gris
  - doppler     : deux fenêtres de même largeur ; à gauche l'image en gris,
                  à droite la même image avec une zone colorée superposée
  - independent : deux fenêtres de même largeur, montrant deux vues différentes

Il produit ensuite une planche de résultats illustrant quelques augmentations.

Usage :
    python generate_demo.py
================================================================================
"""

import os
import cv2
import numpy as np

import augmentation as aug


RNG = np.random.default_rng(0)
H, W = 256, 256


# ==============================================================================
# Fabrication d'une fausse image d'échographie
# ==============================================================================

def make_ultrasound_like(h=H, w=W, n_blobs=5):
    """
    Fabrique une image en niveaux de gris qui imite l'aspect d'une échographie :
    un cône d'acquisition clair sur fond sombre, quelques structures diffuses,
    et un bruit speckle multiplicatif caractéristique.
    """
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Cône d'acquisition : la sonde est au centre du bord supérieur
    cx = w / 2.0
    angle = np.abs(np.arctan2(x - cx, y + 1e-3))
    cone = np.clip(1.0 - (angle / 0.8) ** 2, 0.0, 1.0)
    depth = np.clip(1.0 - y / h, 0.15, 1.0)
    image = 90.0 * cone * depth

    # Quelques structures internes diffuses
    for _ in range(n_blobs):
        bx = RNG.uniform(0.25 * w, 0.75 * w)
        by = RNG.uniform(0.25 * h, 0.85 * h)
        sx = RNG.uniform(10, 30)
        sy = RNG.uniform(10, 30)
        amp = RNG.uniform(40, 90)
        image += amp * np.exp(-(((x - bx) ** 2) / (2 * sx ** 2)
                                + ((y - by) ** 2) / (2 * sy ** 2)))

    # Bruit speckle : multiplicatif, typique de l'échographie
    image *= (1.0 + RNG.normal(0.0, 0.25, image.shape))

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def add_doppler_overlay(gray_bgr):
    """
    Superpose une zone colorée de type Doppler sur une image en gris.
    La teinte code une vitesse : rouge d'un côté, bleu de l'autre.
    C'est exactement l'information que l'augmentation ne doit pas altérer.
    """
    h, w = gray_bgr.shape[:2]
    out = gray_bgr.copy()

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.55
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = r < min(h, w) * 0.22

    # Vitesse fictive : positive à gauche, négative à droite
    velocity = (cx - x) / (min(h, w) * 0.22)
    velocity = np.clip(velocity, -1.0, 1.0)

    overlay = np.zeros_like(out)
    overlay[..., 2] = np.clip(velocity, 0, 1) * 255      # rouge
    overlay[..., 0] = np.clip(-velocity, 0, 1) * 255     # bleu

    out[mask] = cv2.addWeighted(out, 0.35, overlay, 0.65, 0)[mask]
    return out


def make_simple():
    return make_ultrasound_like()


def make_doppler():
    """Gauche = gris, droite = la MÊME image avec surcouche Doppler."""
    left = make_ultrasound_like()
    right = add_doppler_overlay(left)
    return np.concatenate([left, right], axis=1)


def make_independent():
    """Deux vues différentes et indépendantes."""
    left = make_ultrasound_like(n_blobs=4)
    right = add_doppler_overlay(make_ultrasound_like(n_blobs=7))
    return np.concatenate([left, right], axis=1)


def pick_showcase(results, category, n=7):
    """
    Sélectionne des augmentations variées et représentatives plutôt que
    les premières de la liste, afin que la planche montre réellement
    la diversité des transformations.
    """
    wanted = {
        "simple": ["flip_horizontal", "rotate_p10", "speckle_noise",
                   "elastic", "blur", "sharpen", "gamma_bright"],
        "doppler": ["flip_left_half", "flip_right_half", "flip_both_halves",
                    "speckle_noise", "elastic",
                    "brightness_bright_sat_high", "gamma_dark_val_bright"],
        "independent": ["swap_halves", "flip_horizontal", "speckle_noise",
                        "elastic", "flip_left_half_normal",
                        "contrast_high_sat_low", "gamma_bright_val_dark"],
    }[category]

    by_name = dict(results)
    chosen = [(w, by_name[w]) for w in wanted if w in by_name]

    # Complète si certains noms n'existent pas dans cette configuration
    if len(chosen) < n:
        taken = {name for name, _ in chosen}
        step = max(1, len(results) // n)
        for name, img in results[::step]:
            if name not in taken:
                chosen.append((name, img))
                taken.add(name)
            if len(chosen) >= n:
                break

    return chosen[:n]


# ==============================================================================
# Planche de résultats
# ==============================================================================

def make_montage(pairs, cell_h=200, cols=4, margin=8):
    """Assemble une liste de (titre, image) en une planche unique."""
    cells = []
    for name, img in pairs:
        scale = cell_h / img.shape[0]
        thumb = cv2.resize(img, (int(img.shape[1] * scale), cell_h))
        canvas = np.zeros((cell_h + 26, thumb.shape[1], 3), dtype=np.uint8)
        canvas[26:, :] = thumb
        cv2.putText(canvas, name[:30], (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(canvas)

    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        height = max(c.shape[0] for c in row)
        width = max(c.shape[1] for c in row)
        padded = []
        for c in row:
            p = np.zeros((height, width, 3), dtype=np.uint8)
            p[:c.shape[0], :c.shape[1]] = c
            padded.append(p)
            padded.append(np.zeros((height, margin, 3), dtype=np.uint8))
        rows.append(np.concatenate(padded[:-1], axis=1))

    width = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        p = np.zeros((r.shape[0], width, 3), dtype=np.uint8)
        p[:, :r.shape[1]] = r
        padded_rows.append(p)
        padded_rows.append(np.zeros((margin, width, 3), dtype=np.uint8))

    return np.concatenate(padded_rows[:-1], axis=0)


# ==============================================================================
# Programme principal
# ==============================================================================

def main():
    builders = {
        "simple": (make_simple, aug.augment_simple),
        "doppler": (make_doppler, aug.augment_doppler),
        "independent": (make_independent, aug.augment_independent),
    }

    os.makedirs("dataset", exist_ok=True)
    os.makedirs("examples", exist_ok=True)

    for category, (build, augment_fn) in builders.items():
        os.makedirs(os.path.join("dataset", category), exist_ok=True)

        image = build()
        cv2.imwrite(os.path.join("dataset", category, f"{category}_000.png"), image)

        results = augment_fn(image)
        selection = [("original", image)] + pick_showcase(results, category)

        montage = make_montage(selection)
        out = os.path.join("examples", f"{category}.png")
        cv2.imwrite(out, montage)

        print(f"[{category}] {len(results)} augmentations générées → {out}")

    print("\nJeu de démonstration créé dans dataset/, planches dans examples/")


if __name__ == "__main__":
    main()
