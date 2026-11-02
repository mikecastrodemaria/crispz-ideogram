# crispz-ideogram : fork Ideogram 4.0 de crispz-flux3

- Base : crispz-flux3 0.1.0 (`main` 8117fb2), historique complet importé.
- Remotes : `origin` = crispz-ideogram, `flux3` = crispz-flux3.
- Notes des forks précédents : [FORK-flux3.md](FORK-flux3.md),
  [FORK-flux3-CLAUDE.md](FORK-flux3-CLAUDE.md) (pièges d'environnement, à lire avant tout
  test GPU), [FORK-klein.md](FORK-klein.md).
- Journal complet, lot par lot, avec les mesures : [CLAUDE.md](CLAUDE.md) §8.

## Pourquoi crispz-flux3 comme base

| | crispz-flux3 | crispz-klein | crispz-krea2 |
|---|---|---|---|
| VAE | **le même** (`AutoencoderKLFlux2`, identique bit pour bit, stats `bn` comprises) | idem | autre |
| Encodeur texte | **même famille** (Qwen3-VL) | Qwen3 | autre |
| Pile | **diffusers 0.42.1 + transformers 5 déjà réglée** | diffusers figé, transformers 4 | ancienne |
| img2img / inpaint émulés | **écrits et testés** (SDEdit, mélange latent) | non | renonce |

Le choix a tenu : l'aperçu live, l'encodage VAE, l'émulation d'img2img et d'inpaint et la
garde par famille de modèle se sont transposés, avec deux corrections propres à
Ideogram (voir plus bas).

## Ce qui a changé par rapport à crispz-flux3

| | crispz-flux3 | crispz-ideogram |
|---|---|---|
| Moteur | `cz_flux3.Flux3Pipe` (ModularPipeline FLUX.3) | `cz_ideogram.IdeogramPipe` (`Ideogram4Pipeline`, même convention d'appel) |
| Poids | bf16 / fp8r / nvfp4, fichiers uniques acceptés | nf4 (bitsandbytes, CUDA), dépôt diffusers seulement, fichiers uniques refusés avec la raison |
| Réglages par défaut | 16 steps, guidance 1,0 (distillé) | 48 steps, guidance 7 = pic du planning officiel par step |
| Négatif | sur le checkpoint base | aucun : la branche non conditionnelle ne prend pas de texte |
| Édition par références | native (même pipeline) | **déléguée à crispz-flux3** (onglet Edit in crispz-flux3, protocole v1) |
| img2img / inpaint | natifs dans l'adaptateur | **émulés** (fin du planning logit-normal d'Ideogram, temps inversé respecté) |
| Prompt | texte libre | **légendes JSON** (éditeur, boîtes, presets de design, magic prompt, Describe) |
| Offload | `model` de diffusers | `model` propre à Ideogram : encodeur seul (le `model` de diffusers renvoyait un transformer en RAM à chaque step) |
| Nouveau | | vectorisation locale (vtracer), fonctions cloud Ideogram cachées, test e2e Ideogram |

Le code FLUX.3 reste dans `cz_pipeline` (non atteint, testé par `test_flux3_wiring`) pour
que les correctifs de crispz-flux3 fusionnent sans conflit inutile.

## Ce qu'Ideogram a demandé en plus

- **Aperçu live** : même VAE, mais Ideogram range ses tokens en (p, p, c) et les normalise
  avec les stats `bn` indexées dans l'ordre FLUX. Permutation plus correction affine,
  repliées dans la matrice (erreur au dernier step 45 contre 78 sur 255).
- **Attention et normalisation** : le pipeline réservait 2048 places de texte et passait
  un masque entièrement vrai (noyau lent) ; la RMSNorm du texte tournait sur toute la
  séquence en float32. Zone texte à la taille du prompt, masque retiré, norme sur les
  seules lignes de texte : x1,32 et pic à 2048 de 29,9 à 23,4 Go.
- **Filtre de sécurité** intégré aux poids : faux positifs fréquents en texte libre, rares
  en JSON. Le contourner est interdit par la licence (§4), et refusé ici.

## Mesures (RTX 5090, nf4, offload auto, 48 steps)

| | |
|---|---|
| Chargement depuis le cache | 20 à 23 s |
| 1024x1024 | 61 s, pic 19,2 Go |
| 2048x2048 | 394 s, pic 23,4 Go |
| Magic prompt local | 10,7 à 18,5 s |
| img2img 0,3 / inpaint / outpaint (0,7 à 1 Mpx) | 29 s / 40 s / 57 s |

## Pièges à connaître

- Accepter la licence sur `ideogram-ai/ideogram-4-nf4-diffusers` lui-même.
- Ne jamais lancer `pip install -r requirements.txt` dans le `.venv` installé (Pillow 13
  rétrogradé en 11.3 par gradio).
- Un verdict d'offload est mémorisé dans `cache/hw_profile.json` : le supprimer après un
  changement de carte ou de réglage pour re-mesurer.
- L'op `edit` du protocole de crispz-flux3 ne garde que l'image d'entrée comme référence :
  avec des références, la passerelle envoie `gen` + `refs`.
