# crispz-flux3 : fork FLUX.3 Image Klein de crispz-klein

Fork **texte vers image + édition multi-référence** basé sur
[`black-forest-labs/flux-3-image-klein`](https://huggingface.co/black-forest-labs/flux-3-image-klein)
(5,6B, distillé 16 steps, sans CFG), sorti le 15 octobre 2026.

- Remotes : `origin` = crispz-flux3, `klein` = crispz-klein.
- Base du fork : `3316ef8` (crispz-klein/main, 1.36.6).
- Historique du fork précédent (crispz-qwen-edit vers crispz-klein) : [FORK-klein.md](FORK-klein.md).

> **État : étape 1 sur 5 (squelette).** Le code est encore celui de crispz-klein,
> renommé. Rien ne charge FLUX.3 pour l'instant : voir la feuille de route.

## Pourquoi un fork et pas une 3e variante dans crispz-klein

FLUX.3 n'est pas un FLUX.2 plus gros, c'est une autre mécanique :

| | FLUX.2 klein (crispz-klein) | FLUX.3 Image Klein (ce fork) |
|---|---|---|
| Pipeline diffusers | `Flux2KleinPipeline` + `Flux2KleinInpaintPipeline` | `ModularPipeline` (`Flux3AutoBlocks`), pas d'inpaint |
| Transformer | `Flux2Transformer2DModel` (3072 / 4096) | `Flux3Transformer2DModel`, 5,6B |
| Encodeur texte | Qwen3-4B / Qwen3-8B (texte seul) | Qwen3-VL-4B-Instruct (vision-langage) |
| Layout du repo HF | sous-dossiers diffusers | `model.safetensors`, `ae.safetensors`, `text_encoder/`, `variants/{fp8r,nvfp4}` |
| Steps / CFG | 4 / aucun | 16 / aucun (Base : 50 / CFG) |
| Références | liste `image=[...]` | liste `image=[...]` + `<ref_image_N>` dans le prompt, jusqu'à 10 |
| Extras | | bounding boxes (grille 0-1000, y d'abord), sortie 4K native |
| diffusers | commit figé `de6b049` | diffusers récent (ModularPipeline) |
| Licence | Apache 2.0 (4B) | **FLUX Kommunity License v1.0** |

Mélanger les deux dans un même venv imposait de monter diffusers pour tout le monde,
au risque de casser crispz-klein, validé de bout en bout sur son commit. Ici, le fork
a son propre `.venv` et sa propre pile.

## Licence : à lire avant tout usage commercial

`flux-3-image-klein` est sous **FLUX Kommunity License v1.0** (repo HF à accès
conditionnel : accepter les conditions avec son compte HF, puis un token READ dans
la config `hf_token`).

- Usage non commercial : libre.
- Usage commercial des sorties : gratuit pour un "Qualifying User" dont le chiffre
  d'affaires annualisé (sociétés affiliées comprises) est inférieur à 5 M$.
- Conditions : filtrer ou relire les sorties avant publication, signaler le contenu
  généré par IA quand la loi l'exige, ne pas utiliser les sorties pour entraîner un
  modèle d'image concurrent.
- Au-delà de 5 M$ : licence payante ou API BFL.

L'encodeur Qwen3-VL-4B-Instruct est sous Apache 2.0. Le prompt upsampler
`flux-3-image-klein-pup` (9B) a sa propre licence, à vérifier avant de l'activer.

## Mesures sur RTX 5090 (32 Go, driver 610.60, torch 2.11 cu128, 28/10/2026)

`tests/test_flux3_e2e.py`, repo distillé en bf16, offload `auto` résolu en `none`
(30,3 Go libres pour 22,9 Go estimés), 1024x1024, 16 steps, seed 7 :

| Étape | Temps | VRAM résidente | Pic réservé |
|---|---|---|---|
| Chargement (transformer + Qwen3-VL + VAE) | 50,6 s à froid, 21,8 s à chaud | 18,8 Go | 18,8 Go |
| txt2img (première image, noyaux CUDA compris) | 7,1 s | 18,8 Go | 23,7 Go |
| Édition, 1 référence | 13,4 s | 18,8 Go | 21,7 Go |
| Édition, 2 références | 23,5 s | 18,8 Go | 22,6 Go |
| Inpaint (force 0,9, 14 steps) | 3,6 à 5,7 s | 18,8 Go | 21,7 Go |
| Refine img2img (force 0,35, 6 steps) | 2,3 s | 18,8 Go | 21,7 Go |
| txt2img, Stop au 3e step | 1,4 s | 18,8 Go | 21,7 Go |

**2K et 4K natifs** (même machine, 16 steps, offload `model` : le verdict en cache venait
d'un plantage à deux modèles, les temps en `none` seront un peu meilleurs) :

| Format | Pixels | Tokens | Temps | Pic réservé |
|---|---|---|---|---|
| 2048 x 2048 | 4,2 MP | 16384 | 52 s | 15,0 Go |
| 2560 x 1440 | 3,7 MP | 14400 | 34 s | 14,1 Go |
| 3840 x 2160 | 8,3 MP | 32400 | 113 s | 19,6 Go |

Le 3840x2160 est aussi propre que le 1 MP (pas de sujet dupliqué, composition tenue) :
FLUX.3 rend le 4K sans tuilage ni upscale. Les presets 2K et 4K de Settings en découlent
(curseurs jusqu'à 4096 par côté, la borne que diffusers met à une image de référence).

**Fichiers CivitAI** (page "Flux 3 Klein INT8/INT6/INT4", 28/10) : le bf16 single-file
charge et rend (23,7 Go de pic) ; les INT8 comfy-quant sont déquantifiés en bf16 par le
chargeur de klein (11 Go en VRAM, le gain est sur le disque) ; les INT6 `w6a8_int8` sont
refusés dès le menu, aucun chargeur ne les reconstruit ; le GGUF Q4_K_M charge via
GGUFQuantizationConfig une fois le dtype de calcul épinglé (diffusers annonce uint8 comme
dtype d'un modèle GGUF et les blocs modulaires castaient les latents dessus).

**Outpaint LoRA** (CivitAI 3003865, le premier LoRA d'édition FLUX.3, 29/10) : vérifié sur
la 5090 par la CLI (`--expand right --expand-lora`, 1024 vers 1344x1024) : bf16 16 steps, et
GGUF Q4_K_M en offload model 30,9 s ; la bande continue la pluie et le sol sans couture ni
gris résiduel (`assets/sample-outpaint.jpg`). Le premier essai de Mike dans l'app échouait
en offload model : le chargeur LoRA de diffusers retire les hooks accelerate puis appelle
`enable_model_cpu_offload()` sur le pipeline modulaire, qui ne l'a pas (AttributeError, le
pipe laissé sur CPU). `Flux3Pipe.load_lora_weights` retire les hooks avant et les remet
après. Mesure annexe : un premier pas à 5 min 27 quand l'app de Mike rendait en même temps
(carte à 100 %), 1,3 s par pas ensuite ; la même commande seule, 6,2 s en tout à 4 steps.

Bench `tools/bench_models.py` du 28/10 au soir (5090, bf16, offload none sauf GGUF,
1024x1024, seed 12345, 3 images par modèle ; "stable" = moyenne des deux dernières) :

| Fichier | Variante | Steps | Chargement | Stable | Pic VRAM |
|---|---|---|---|---|---|
| repo distillé (cache HF) | distilled | 4 (voir note) | 15 à 26 s | 1,6 s | 21,2 Go |
| flux3Klein_klein_bf16.safetensors (11 Go) | distilled | 16 | 131 s | 5,6 s | 21,2 Go |
| flux3KleinSDINT8.safetensors (comfy-quant) | distilled | 16 | 85 s dont 64 s de dequant | 5,6 s | 21,2 Go |
| flux3KleinGGUF_v10_q4KM.gguf | distilled | 15 | 45 s | 11,4 s | 8,4 Go |
| flux3KleinINT8.safetensors (comfy-quant) | base | 50, CFG 3,5 | 84 s dont 64 s de dequant | 33,2 s | 21,2 Go |
| flux3KleinINT6 / SDINT6 (w6a8_int8) | | | refusés dès le menu | | |

Lecture : les single files bf16 et INT8 déquantifié rendent au même rythme que le repo
(0,35 s/step) ; le GGUF tourne en offload `model` (forcé par l'app, un GGUF ne va pas sur
le GPU en none/sequential) à 0,76 s/step pour 8,4 Go de pic ; la colonne chargement est
le disque (le même transformer : 15 s depuis le cache HF, 131 s en fichier de 11 Go sur
le F:) plus la minute de dequant des comfy-quant, refaite à chaque chargement (pas de
cache disque, choix du lot A). Note : ce bench tournait encore avec les profils klein
(4 steps sur les repos, et l'entrée `-base` chargeait le repo distillé) ; corrigé le
soir même (variantes lues dans les tenseurs, profils 16 / 50, refus affichés dans le
plan), le repo base reste à mesurer. Le bench a aussi révélé que `from_single_file`
(diffusers 0.42.1) casts `norm_q` / `norm_k` en bf16 là où `from_pretrained` les garde
en float32 (`_keep_in_fp32_modules`) : rms_norm sortait du noyau fusionné à chaque pas
(avertissement "Mismatch dtype between input and weight") ; cz_flux3 les remet en
float32 après tout chargement single-file non quantifié.

Les 18,8 Go résidents sont exactement le chiffre BFL pour la H100 en bf16 ; le pic à
23,7 Go sur la première génération inclut l'échauffement des noyaux. Une 5090 fait donc
tourner FLUX.3 bf16 entier, sans offload ; fp8r (14,4 Go annoncés) et nvfp4 (13,4 Go)
restent à mesurer ici (sélecteur Precision).

Ce que le e2e a appris, au-delà des chiffres :
- **L'édition est timide sans le prompt upsampler.** "make it daytime, sunny, blue sky"
  sur la scène de pluie éclaircit le ciel mais garde la pluie, et le chemin diffusers
  brut (bloc text_encoder officiel, même seed) donne un résultat identique au pixel
  près : c'est le modèle, pas l'adaptateur. Une description complète de l'image cible
  aide un peu ; le pup 9B de BFL (v0.2, ou son `system_prompts.json` via Ollama) est fait
  pour cette réécriture.
- **L'inpaint veut le prompt de toute la scène**, pas celui du patch : "a large full moon"
  seul ne donne rien, "the same panel ... a large bright full moon behind heavy clouds
  in the top right corner" peint la lune, sans couture.
- **La force du img2img est une fraction des steps** (règle diffusers, celle de klein) :
  lue comme un sigma, avec le shift 6,93, un refine à 0,35 ne faisait qu'un seul step.

Le prompt upsampler ajoute environ 28 Go : sur une 5090 (32 Go), il ne cohabite pas
avec le générateur. Schéma prévu : upsampler, puis déchargement, puis génération
(ou upsampler confié à Ollama, déjà intégré via `cz_ollama.py`).

## Feuille de route

1. **Squelette** (fait) : fork, renommage `crispz-klein` vers `crispz-flux3`,
   version `0.1.0-dev`, ce document.
2. **Cœur** (fait, lot A) : `cz_flux3.Flux3Pipe` enveloppe le pipeline modulaire de
   diffusers 0.42.1 derrière la convention Flux2Klein ; img2img et inpaint ajoutés dans
   l'adaptateur ; défauts 16 steps / sans CFG ; références `<ref_image_N>`.
3. **Nettoyage** (fait, lots A à C) : gardes par famille de modèle au lieu de la largeur
   3072/4096 (FLUX.3 est aussi large que klein-4B), sélecteur distillé / base puis
   bf16 / fp8r / nvfp4, encodeur de remplacement et sampler non-Euler grisés avec leur
   raison, dequant ComfyUI non porté.
4. **Dépendances et tests** (fait, lots D et E) : diffusers PyPI, transformers 5, Python
   3.11+, suite verte sur CPU, e2e GPU ci-dessus.
5. **Reste** : documentation finale et fusion dans `main` (lot F), launcher
   `crispz-flux3.pinokio`. Puis v0.2 : prompt upsampler, bounding boxes, troncature de
   l'encodeur Qwen3-VL, mesures fp8r / nvfp4, qualité du CFG sur la variante base.

## Sources

- Model card : https://huggingface.co/black-forest-labs/flux-3-image-klein
- Diffusers : https://huggingface.co/blog/flux-3
- Docs BFL : https://docs.bfl.ml/flux_3/flux3_image_klein
- Code de référence : https://github.com/black-forest-labs/flux-3-image-klein
