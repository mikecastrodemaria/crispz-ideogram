# crispz-moss : cahier des charges pour Claude Code

Futur `CLAUDE.md` du repo `mikecastrodemaria/crispz-moss` (à créer par Mike). Rédigé le
03/11/2026 depuis crispz-ideogram 0.1.0, à partir du code publié de MOSS-Image (pipeline
`pipeline_moss_image.py` lu en entier), de son guide utilisateur, du rapport technique
(arXiv 2610.23374) et de trois tests indépendants. Même méthode que `CLAUDE.md` de
crispz-ideogram : lots courts, validation de Mike entre chaque lot, mesures sur la 5090.

## 1. Objectif

Un studio local autour de **MOSS-Image** (OpenMOSS, Fudan, sorti le 11/10/2026) : le
frère **commercialisable** de crispz-ideogram. Même positionnement (texte dans l'image,
affiches, logos, packaging, slides, typographie dense, contrôle par légende JSON avec
boîtes), même interface, mais des poids **Apache 2.0** : un livrable client produit en
local est permis, là où Ideogram 4 l'interdit sans licence négociée.

Ce que MOSS apporte en plus d'Ideogram 4 : licence libre, modèle deux fois plus petit
(5B contre 2 x 9,3B), prompt négatif réel, un seul transformer, LoRAs entraînables dès
6 Go (KohakuMossTrainer, DiffSynth), un écosystème déjà vivant (distillations 4 et 8
steps, LoRA d'édition avec références, LoRA anti-banding, repack ComfyUI, GGUF, MLX).
Ce qu'il n'apporte pas, ou pas encore : la preuve, sur nos affiches, que le texte est
aussi fiable que celui d'Ideogram 4 (les tests indépendants se contredisent, voir §2).
**Le lot 1 tranche cette question par un A/B avant tout autre travail.**

## 2. Le modèle : ce qui est vérifié (03/11/2026)

| | MOSS-Image (local) |
|---|---|
| Poids | `OpenMOSS-Team/MOSS-Image-SFT` (usage général) et `MOSS-Image-Base` (base des LoRAs), non gated |
| Licence | Apache 2.0, poids et code de pipeline ; encodeur Qwen3.5-9B Apache 2.0 ; VAE repris de FLUX.2 klein 4B (Apache 2.0, à confirmer au lot 6) |
| DiT | 5B, 42 blocs, dim 3072, 24 têtes (head_dim 128), FFN 8192, SiTU-GLU, RoPE 4D, modulation partagée |
| Encodeur | Qwen3.5-9B gelé (`Qwen3_5Model`, 32 couches, hidden 4096), `last_hidden_state` seulement, tour vision incluse dans les poids (`qwen3_5_vision`) |
| VAE | `AutoencoderKLFlux2` (32 canaux, normalisation `bn`), le même que FLUX.2, FLUX.3 et Ideogram 4 |
| Latents | 128 canaux sur grille /16 (2 x 2 patches de 32 canaux), rangés dans l'ordre FLUX `(c, p, p)` |
| Taille des fichiers (bf16) | DiT 10,2 Go + encodeur 16,8 Go + VAE 0,25 Go = **27,2 Go** |
| Réglages officiels | 50 steps, CFG 3,0 (3 à 6 conseillé), shift 5,0, Euler, `uncond_skip_layers=[22]`, négatif vide ou `nsfw, worst quality, bad anatomy, watermark, jpeg artifact` |
| Résolution | multiples de 16 ; recommandé 1, 1,5 ou 2 Mpx ; extrapole de 0,25 à 4 Mpx ; ratio libre |
| Prompts | texte libre (anglais conseillé, chinois ok), séquences de tags, layout JSON ; `\n` pour forcer les retours à la ligne du texte rendu |

Le pipeline, tel que publié (`custom_pipeline="pipeline_moss_image"`, `trust_remote_code`,
440 lignes, Apache 2.0) :
- Classes : `MOSSImageTransformer`, `MOSSImageScheduler` (sous-classe de
  `FlowMatchEulerDiscreteScheduler` : sigmas décalées par `shift` puis **alignées sur la
  grille d'entraînement**, doublons retirés ; `set_timesteps(num_inference_steps)` seul,
  pas de paramètre `sigmas`), `MOSSImagePipeline(DiffusionPipeline)`.
- **Absents** : `callback_on_step_end`, `_interrupt`, mixin LoRA, paramètre `sigmas` ou
  `timesteps`. Présents : `latents`, `prompt_embeds` + `prompt_attention_mask`,
  `output_type="latent"`, `num_images_per_prompt`, `uncond_skip_layers`, SLG optionnel
  (`slg_layers`, `slg_scale`, `slg_start`, `slg_stop`).
- Le négatif passe par l'encodeur (deux encodages par image quand il change) ; par
  défaut CFG = deux passes de transformer par step, la branche négative sautant le bloc 22.
- Le transformer **boucle sur le batch** (une passe par image, à la vraie longueur de
  texte, aucun masque dans SDPA) : batch N = N passes, pas d'attention gonflée par le
  rembourrage (le problème corrigé sur Ideogram n'existe pas ici). head_dim 128 : le
  noyau flash de SDPA est éligible (Ideogram, à 256, n'avait que `efficient`).
- Encodage du prompt : gabarit de chat Qwen (`add_generation_prompt`,
  `enable_thinking=True`), 4096 tokens max, un prompt à la fois sans rembourrage.
- Bruit initial et pas d'Euler en fp32, DiT et encodeur en bf16, VAE en fp32.
- Décodage : dénormalisation `bn` sur les 128 canaux tels que stockés, puis
  `(b, c//4, 2, 2, h, w)` vers `(b, c//4, 2h, 2w)`.
- `model_cpu_offload_seq = text_encoder->transformer->vae`, `maybe_free_model_hooks()`
  en fin d'appel.

Mesures publiées par des tiers, à refaire chez nous : RTX 5090, ComfyUI, DiT et encodeur
en int8, 1088x1920, 50 steps : **37 s par image, 15,5 Go** ; checkpoint DMD 8 steps à
CFG 1 : 3,9 s ; LoRA 4 steps : 2,2 s. RTX 4070 12 Go, int8, 832x1216 : 52 s. Diffusers
bf16 sans quantification à 2048x1152 : plus de 32 Go. Chez OpenMOSS, 1024x1024 en
50 steps avec leur profil `minimal` (DiT en RAM) : 3,65 Go de VRAM, 14 Go de RAM.

Qualité du texte, état des sources : le rapport annonce 96,3 sur son propre benchmark
LLTB (Qwen-Image 2512 : 64,9) et un mIoU de boîtes de 0,88 contre 0,44 pour Ideogram 4 ;
son classement humain (2e derrière GPT Image 2) compte un prompt enhancer GPT-6. Tests
indépendants : MindStudio juge le texte et les boîtes « hésitants, pas prêts pour la
prod » et le photo-réalisme excellent ; ai-muninn obtient 4 titres chinois sur 4 contre
0 sur 4 pour Qwen-Image 2.1, mais rate les petits textes fixes, et constate que les
variantes 4 et 8 steps ajoutent de faux textes anglais en fond. Défaut connu du SFT :
**banding vertical** (LoRA communautaire DeBand). Variante FP16 corrigée existante,
inutile en bf16.

À vérifier au lot 1 (non confirmé) : quantification nf4 bitsandbytes de l'encodeur
Qwen3.5 (couches GDN hybrides, noyaux `flash-linear-attention` absents sous Windows :
repli torch, vitesse d'encodage à mesurer) ; RAM nécessaire à l'offload « encodeur
seul » (16,8 Go en RAM) ; vitesse réelle en bf16 sans quantification du DiT.

## 3. Base : crispz-ideogram, pas crispz-flux3

| Candidat | Pour | Contre |
|---|---|---|
| **crispz-ideogram 0.1.0** (recommandé) | éditeur de légende avec boîtes à la souris, presets de design, magic prompt (local / Ollama), Describe local, SDEdit et inpaint émulés sur ce même VAE, offload « encodeur seul » déjà écrit pour un encodeur Qwen de 8 à 9B, garde VRAM, handoff vers crispz-flux3, pile diffusers 0.42.1 + transformers 5.21 déjà compatible avec MOSS | schéma JSON à traduire (0..1000 contre pixels centrés) ; reste de code FLUX.3 et Ideogram à retirer |
| crispz-flux3 | même VAE, même pile | aucun éditeur de légende, aucun preset de design : tout le lot 2 d'Ideogram serait à refaire |

**Décision : forker crispz-ideogram `main` (0.1.0).** Remotes : `origin` = crispz-moss,
`ideogram` = crispz-ideogram (pour reprendre les correctifs communs).

Ce qu'on garde : toute l'app, `cz_caption` / `cz_caption_ui` / `cz_design` / `cz_magic`,
`cz_vector`, `cz_handoff`, le détailleur, les tuiles, l'ESRGAN, la queue, le XYZ, le
protocole v1, les tests. Ce qu'on remplace : le moteur (`cz_ideogram.py` devient
`cz_moss.py`), les profils, le schéma de légende, les presets de steps. Ce qu'on retire :
`cz_ideogram_api.py` et `cz_cloud_ui.py` (MOSS n'a aucun cloud, tant mieux),
`third_party/ideogram4`, le code FLUX.3 de `cz_pipeline.py` resté pour les fusions.

## 4. Règles de travail

- Commits au nom de Mike (`Mike Castro Demaria <supersoniquestudio@gmail.com>`), auteur
  et committer, sans aucune ligne d'attribution à Claude.
- Ni tiret cadratin ni emoji, nulle part.
- Étape par étape : à la fin de chaque lot, s'arrêter, résumer, attendre la validation.
- Aucune fonction n'envoie de données hors de la machine ; Ollama compris (serveur local
  seulement, URL affichée dans le log à chaque appel).
- Toute dégradation est annoncée dans le log (`_log`), jamais silencieuse.
- Le pipeline MOSS est **vendorisé** dans `third_party/moss_image/` (fichier, LICENSE
  Apache 2.0, commit d'origine, hash SHA-256) : jamais de `trust_remote_code` à
  l'exécution, et une diff lisible si OpenMOSS publie une nouvelle version.
- Avant toute mesure GPU : carte libre (`nvidia-smi`, 0 à 5 %), apps crispz fermées.

## 5. Lots

### Lot 1 : moteur local (cz_moss.py) et A/B texte contre Ideogram

1. `MossPipe` autour d'une sous-classe du `MOSSImagePipeline` vendorisé (même convention
   d'appel que `IdeogramPipe`). La sous-classe ajoute : `callback_on_step_end` (tenseur
   `latents`, step joué, total), `_interrupt` (Stop qui rend l'image partielle),
   paramètre `sigmas` (planning tronqué pour le lot 3, en gardant l'alignement sur la
   grille d'entraînement du scheduler d'origine). Rien d'autre n'est modifié dans le
   fichier vendorisé.
2. Chargement depuis `moss_model` (défaut `OpenMOSS-Team/MOSS-Image-SFT`, `Base` au
   choix), sans token. Précision : DiT bf16 (10,2 Go), encodeur `moss_encoder_precision`
   = bf16 ou nf4 (bitsandbytes, à vérifier sur Qwen3.5), VAE fp32. Offload none / model
   (« encodeur seul », repris d'`IdeogramPipe.enable_model_cpu_offload`) / sequential /
   auto. Les hooks de diffusers ne sont pas utilisés (même raison qu'Ideogram : le
   transformer ne doit pas faire d'aller-retour par step).
3. Txt2img : steps (défaut 50, curseur 2 à 80), taille (multiples de 16, 512 à 2048 par
   côté, 4 Mpx max, préréglages à 1 / 1,5 / 2 Mpx), seed, batch (boucle interne du
   transformer : coût linéaire, dit dans l'aide), `guidance_scale` constant (défaut 3,0,
   curseur 0 à 10), **prompt négatif réel** (`supports.negative = True`, preset de tags
   officiel proposé), `uncond_skip_layers` (défaut 22, case « réalisme brut » qui le
   coupe, aide : « plus réaliste, détails qui peuvent s'effondrer »), SLG avancé masqué
   par défaut. `encode_prompt` exposé au cache d'embeddings de `cz_pipeline` (clé :
   prompt + négatif + longueur max).
4. Aperçu live : les tokens sont dans l'ordre FLUX, la matrice FLUX.2 de
   `cz_pipeline` doit s'appliquer sans la correction Ideogram. Test : inverse exact du
   décodage MOSS sur stats `bn` aléatoires, puis mesure sur la carte (erreur sur 0-255
   au dernier step, seuil 40 du script).
5. Garde par famille (`family_from_keys`) : marqueurs forts `video_embedder`,
   `text_embedder.proj`, `shared_block_modulation`, `final_linear_video`,
   `layers.N.proj.q` ; largeur 3072 ; FLUX.2 / FLUX.3 / Ideogram / Z-Image refusés en
   nommant leur studio. LoRAs : lot 4 (pas de mixin dans le pipeline).
6. `cz_protocol` : `negative` True, `img2img` / `inpaint` False jusqu'au lot 3, `edit`
   et `refs` False, `cloud` absent.
7. **`tools/gpu_lot1_check.py`** sur la 5090, carte libre, pour chaque offload utile :
   chargement, VRAM au repos et en pic, 1024x1024, 1456x1456 (2 Mpx), 2048x1152, 50 steps,
   Stop, aperçu live, encodeur bf16 contre nf4 (temps d'encodage et écart). Puis l'**A/B
   texte** : les trois affiches du lot 2 d'Ideogram (`bench/lot2/`, dont « GRAND OPENING »
   et le logo « BEAN THERE ») rendues par MOSS en texte libre et en layout JSON, 4 seeds
   chacune, mêmes tailles ; rapport `bench/lot1/ab.md` : texte exact, texte fantôme en
   fond, banding, temps. Verdict de Mike sur images.

Acceptation : tests CPU sur un MOSS miniature aléatoire (`tests/moss_tiny.py`, vrai
tokenizer Qwen3.5, mêmes tests que `test_ideogram_engine.py` : forme, Stop, aperçu,
négatif, garde), une 2 Mpx sur la 5090, et un A/B jugé « au moins aussi bon qu'Ideogram
sur le texte » par Mike. **Si l'A/B déçoit, on s'arrête là** et crispz-moss reste un
banc d'essai.

### Lot 2 : légendes layout JSON

1. Schéma MOSS dans `cz_caption` : `summary {meta[], canvas, caption}`, `items [{label,
   caption, text?, bbox}]`, `bbox_format: "yxyx"`, `bbox_coord_space: "abs_centralized"`
   (pixels, origine au centre, y vers le bas, x vers la droite ; donc **une légende vaut
   pour une taille** : la taille du canevas est écrite dans la légende et l'UI force
   largeur et hauteur à la suivre). Représentation interne gardée en 0..1000 (celle de
   l'éditeur et du plateau de boîtes) et sérialisation MOSS à la taille choisie ; import
   inverse d'un JSON MOSS collé. Validation par un vérificateur maison (le modèle n'en
   publie pas), `\n` autorisés dans `text`.
2. Presets de design repris (affiche, logo, carte, livre, slide, packaging, miniature,
   story, bannière) aux tailles natives 1 à 2 Mpx, `meta` rempli (`poster`, `logo`...).
3. Mode « tags » (ordre officiel Qualité, Score, Meta, Année, Personnage, Série, Artiste,
   Général) et preset de négatif.
4. Magic prompt, deux moteurs locaux : **Ollama** (`cz_ollama`, modèle `qwen3.5` ou au
   choix) avec le prompt du skill officiel `skills/moss-image-prompt-enhancer` (Apache
   2.0, conçu pour GPT-6, à adapter à un modèle local, schéma passé en `format`) ;
   **encodeur réutilisé** (Qwen3.5-9B déjà chargé + `lm_head` téléchargé seul depuis
   `Qwen/Qwen3.5-9B`, même montage que la tête d'Ideogram, `outlines` pour le JSON) :
   à étudier, défaut Ollama si la tête ne tient pas en VRAM.
5. Describe local : tour vision de Qwen3.5 (présente dans les poids) ; image vers légende
   MOSS avec boîtes ; à vérifier en nf4 et en offload encodeur seul.
6. `tools/gpu_lot2_check.py` : légende à la main, magic prompt, Describe, détecteur de
   texte fantôme (OCR léger si disponible, sinon jugement sur images).

Acceptation : une légende à la main et une légende du magic prompt produisent un JSON
valide et une image dont les textes sont aux boîtes demandées (à l'œil, puis mIoU
approximatif si l'OCR est là).

### Lot 3 : img2img, inpaint, outpaint émulés, et variantes rapides

1. SDEdit sur la fin du planning MOSS (sigmas du scheduler d'origine, alignement sur la
   grille conservé, `strength` = fraction des steps, au moins un), image encodée en tokens
   FLUX (inverse exact du décodage, normalisation `bn`), `scheduler.set_timesteps`
   enveloppé le temps de l'appel. Masque : mélange latent à chaque step, max-pool sur la
   grille de 16 px ; masque plein = img2img simple. Reprise directe de
   `IdeogramPipe._prepare_emulated` avec l'ordre de tokens changé.
2. Débloque refine, tuiles, inpaint, outpaint, reframe, détailleur. Presets de refine et
   d'upscale à vérifier sur la carte (12 à 16 steps à denoise 0,3, hérités ; sur MOSS à
   50 steps nominaux, la règle « 3 ou 4 steps joués suffisent » est à remesurer).
3. **Preset « Draft »** : LoRA 4 steps ou checkpoint DMD 8 steps communautaires
   (`lodestones/MOSS-Image-SFT-8-steps-dmd`), CFG 1, `uncond_skip_layers` coupé ; aide :
   « brouillon, textes de fond à relire ». Dépend du chargeur de LoRA du lot 4 pour la
   LoRA, le checkpoint DMD passe par `moss_transformer`.
4. `FAMILY_CAPS` : img2img et inpaint True, « émulé » dans `caps`.

Acceptation : tests CPU repris de `test_ideogram_emulated.py` ; `tools/gpu_lot3_check.py`
(img2img 0,3 / 0,6, inpaint, outpaint, reframe, Draft) sur la 5090.

### Lot 4 : LoRAs

1. Chargeur de LoRA maison (le pipeline n'a pas de mixin) sur le dialecte de `cz_pipeline`
   : cibles `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`, `text_embedder`, `video_embedder`.
   **Piège relevé** : les poids ComfyUI (Comfy-Org) fusionnent q/k/v et les LoRAs
   communautaires sont souvent à ce format ; un chargeur naïf saute 378 clés en silence.
   Table de correspondance fusionné vers séparé, clés non reconnues **comptées et dites**.
2. Garde par famille (lot 1), refus nommés pour FLUX.2 / FLUX.3 / Ideogram / Z-Image.
3. LoRAs communautaires à essayer, chacune opt-in avec sa licence affichée : DeBand
   (banding du SFT), 4 steps (Draft), couleur / profondeur (Comfy-Org `loras/`), et
   **`chinoll/Moss-Image-Edit`** (édition par instruction avec 1 à 4 images de référence ;
   code GPLv3, licence des poids non dite : étudier d'abord comment les références
   entrent dans la séquence dans ses nœuds ComfyUI, et les blocs modulaires
   `OzzyGT/moss_image_custom_blocks` qui font passer des références par la tour vision).
   Si une voie propre existe : onglet « Edit (references) » réactivé, `supports.edit` et
   `refs` à True. Sinon : lot 5.

Acceptation : un LoRA Ideogram refusé, un LoRA MOSS au format ComfyUI chargé sans clé
perdue (test CPU sur fixtures), DeBand et Draft mesurés sur la carte.

### Lot 5 : outils locaux et passerelles

- Remove BG (rembg) et Vectorize (vtracer) : repris tels quels.
- Handoff : « Edit in crispz-flux3 » repris ; ajout de « Compare in crispz-ideogram »
  (même légende traduite dans le schéma Ideogram, même taille, route CLI du protocole v1)
  pour les A/B à la demande. Ports : crispz-flux3 7860, crispz-ideogram et crispz-moss
  à régler dans les deux configs.

### Lot 6 : tests, doc, livraison

- Suite CPU verte, e2e GPU (`tests/test_moss_e2e.py` : chargement, 1024, 2 Mpx, img2img,
  inpaint, Stop) sur la 5090.
- README : licence Apache 2.0 en tête et **chaîne des licences** (DiT, Qwen3.5-9B, VAE
  FLUX.2 klein 4B, LoRAs tierces) avec ce que ça permet ; tableau des mesures ; A/B
  contre Ideogram ; limites (texte fantôme, banding, RAM de l'offload).
- FORK.md : base crispz-ideogram, différences, mesures.
- Launcher Pinokio `crispz-moss.pinokio` (repo à créer par Mike), icône générée avec
  l'app (preset Logo, enfin commercialisable).

## 6. Licence : à afficher dans l'app et le README

Apache 2.0 sur les poids MOSS-Image et le code du pipeline (fichier LICENSE du repo HF).
Usage commercial permis, attribution et mention des modifications à conserver. Les
composants tiers suivent leur propre licence : Qwen3.5-9B (Apache 2.0), VAE FLUX.2 klein
4B (Apache 2.0, à confirmer sur le repo black-forest-labs), LoRAs communautaires (à lire
une par une ; `Moss-Image-Edit` : code GPLv3, poids non précisés). Pour Supersonique
Studio : livrables clients possibles avec le modèle de base ; les LoRAs tierces sont
vérifiées avant tout usage client.

## 7. Hors périmètre v0.1

- Entraînement de LoRAs (KohakuMossTrainer et DiffSynth font ça très bien à côté).
- Variantes non publiées du rapport (MoE, distillée CFG, vidéo / GIF, référence native).
- SGLang, vLLM-Omni, MLX, GGUF (ComfyUI les couvre).
- Tout cloud : il n'en existe pas pour ce modèle, et on n'en veut pas.

## 8. Risques connus avant de commencer

- Texte : résultats tiers contradictoires ; l'A/B du lot 1 décide.
- RAM : l'offload « encodeur seul » garde 16,8 Go d'encodeur en RAM ; sous 48 Go de RAM
  système, préférer l'encodeur nf4 (à valider) ou l'offload sequential.
- Qwen3.5 sous Windows : couches GDN sans noyaux Triton, repli torch ; l'encodage d'un
  prompt long (4096 tokens max) est à chronométrer.
- Banding du SFT : montrer la case DeBand dès le lot 4 ; Base en secours.
- Un `config.txt` repris de crispz-ideogram garde 48 steps / guidance 7,0 : à écraser au
  premier lancement (MOSS : 50 / 3,0), avec un message.

## 9. Sources

- https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT (pipeline, configs, LICENSE)
- https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
- https://github.com/OpenMOSS/MOSS-Image (README, USERGUIDE.md, skill du prompt enhancer)
- https://arxiv.org/html/2610.23374 (rapport technique)
- https://huggingface.co/Comfy-Org/MOSS-Image (repack, int8, LoRAs tierces)
- https://huggingface.co/chinoll/Moss-Image-Edit
- https://huggingface.co/OzzyGT/moss_image_custom_blocks
- https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd
- https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand
- https://ai-muninn.com/en/blog/moss-image-traditional-chinese-comic-cards (5090, int8)
- https://www.mindstudio.ai/blog/moss-image-5b-tested-realism-text
