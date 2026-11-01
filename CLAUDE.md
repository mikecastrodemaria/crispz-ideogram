# crispz-ideogram : cahier des charges pour Claude Code

`CLAUDE.md` du repo `mikecastrodemaria/crispz-ideogram`. Claude Code le lit à
l'ouverture du projet.
Rédigé le 28/10/2026, à partir du code de diffusers 0.42.1 et des fiches Hugging Face.

## 1. Objectif

Un studio local "Ideogram" dans la famille crispz, qui offre le maximum de possibilités
autour du seul modèle Ideogram à poids ouverts, **Ideogram 4.0**, complété en option
par les fonctions cloud d'**Ideogram 4.5** (fermé, API seulement).

Positionnement dans la gamme : **le studio du texte dans l'image et du design**
(affiches, logos, packaging, slides, couvertures, typographie dense). Les autres
crispz restent meilleurs en photo et en édition ; Ideogram 4 est loin devant sur le
rendu de texte (environ 97 % de précision OCR annoncée sur du texte dense).

## 2. Le modèle : ce qui est vérifié

| | Ideogram 4.0 (local) | Ideogram 4.5 (cloud) |
|---|---|---|
| Poids | ouverts, `ideogram-ai/ideogram-4-nf4-diffusers` (gated) | **fermés**, API Ideogram uniquement |
| Licence | Ideogram 4 Non-Commercial | conditions de l'API Ideogram (payant à l'image) |
| Rôle | génération texte vers image | édition précise, masque, références |
| Taille | DiT 9,3B, flow-matching | non publiée |

Ideogram 4.0 dans diffusers 0.42.1 (lu dans le code source) :
- Pipeline classique `Ideogram4Pipeline(DiffusionPipeline, Ideogram4LoraLoaderMixin)`.
  Pas de ModularPipeline à envelopper, contrairement à FLUX.3.
- **Deux transformers** : `transformer` (conditionnel) et `unconditional_transformer`
  (branche non conditionnelle du CFG, sans texte). Conséquence : **pas de prompt
  négatif** possible (la branche négative a des entrées mises à zéro).
- Encodeur texte : **Qwen3-VL-8B-Instruct**, 13 couches cachées concaténées.
- VAE : **`AutoencoderKLFlux2`** avec normalisation `bn`, le même que FLUX.2 et FLUX.3.
- Planning : sigmas logit-normaux (`mu`, `std=1.5`), 48 steps par défaut, guidance
  **par step** (`guidance_schedule` = 7,0 sur 45 steps puis 3,0 sur 3). Le temps du
  modèle est inversé (`t_model = 1 - t/1000`) : à respecter pour l'img2img.
- Résolution : multiples de 16 de 256 à 2048 par côté, ratio jusqu'à 6:1, 2048x2048 par
  défaut (natif 2K).
- `callback_on_step_end` (tenseur `latents`) et `_interrupt` disponibles : le Stop,
  l'aperçu live et la garde VRAM de crispz se branchent sans adaptateur.
- **Magic prompt local intégré** : `prompt_upsampling=True` réutilise le Qwen3-VL-8B
  avec une tête `prompt_enhancer_head` et produit la légende JSON. Génération
  contrainte au schéma si `outlines` est installé (sinon JSON non garanti).
- Pas d'img2img, pas d'inpaint, pas d'édition dans le pipeline.
- Poids nf4 (fichiers) : encodeur 5,48 Go + 2 transformers de 5,22 Go + VAE 0,17 Go,
  soit environ **16,1 Go**. Tient sur la RTX 5090 (32 Go) sans offload, à mesurer.
- Le modèle a été entraîné **uniquement sur des légendes JSON** (boîtes `bbox` pour le
  sujet, le texte et le fond, `compositional_deconstruction`, `colour_palette` en hex).
  Un prompt en texte libre fonctionne mais moins bien.

À vérifier au lot 1 (non confirmé) : la dépendance de quantification nf4
(bitsandbytes sous Windows ?), la modération Hive annoncée dans la fiche (aucune trace
dans le code diffusers), la VRAM réelle et le temps par image à 1024 et 2048.

## 3. Quel crispz prendre comme base : crispz-flux3

| Candidat | Pour | Contre |
|---|---|---|
| **crispz-flux3** (recommandé) | même VAE (aperçu live, encodage, normalisation `bn`) ; même famille d'encodeur (Qwen3-VL) ; pile diffusers 0.42.1 + transformers 5 déjà réglée ; helpers img2img SDEdit et inpaint par mélange latent déjà écrits et testés ; garde par famille de modèle | n'est pas encore fini (v0.1 en cours) |
| crispz-klein | pipeline classique comme Ideogram4Pipeline ; app la plus mûre (1.36.6) | diffusers figé sur un commit ancien, transformers 4 : incompatible avec Ideogram 4 sans refaire tout le travail de montée de version de flux3 ; pas d'img2img émulé |
| crispz-krea2 | profil "texte vers image seulement", proche d'Ideogram natif | pile ancienne aussi ; renonce à l'img2img et à l'inpaint, alors qu'on veut le maximum |
| crispz-studio / crispz-qwen-edit | complets | autre VAE, autre encodeur, plus éloignés |

**Décision : forker crispz-flux3, une fois ses lots A à D terminés** (moteur branché,
config, UI, dépendances). Démarrer avant figerait un état à moitié porté dans deux
repos. Remotes du nouveau repo : `origin` = crispz-ideogram, `flux3` = crispz-flux3.

Ce qu'on garde de crispz-flux3 : toute l'app (UI, CLI, protocole v1, détailleur, tuiles,
upscaler ESRGAN, queue, XYZ, Asset Browser, CivitAI), la matrice d'aperçu live, les
fonctions d'img2img et d'inpaint de `cz_flux3.py` (à transposer, voir §5 lot 3).
Ce qu'on remplace : le moteur (`cz_flux3.py` devient `cz_ideogram.py`), les profils,
l'onglet Edit (local, voir lot 3).

## 4. Règles de travail

- Commits au nom de Mike (`Mike Castro Demaria <supersoniquestudio@gmail.com>`),
  auteur et committer, sans aucune ligne d'attribution à Claude.
- Ni tiret cadratin ni emoji, nulle part.
- Étape par étape : à la fin de chaque lot, s'arrêter, résumer, attendre la validation.
- Toute fonction qui envoie des données hors de la machine est **désactivée par
  défaut**, signalée dans l'UI, et loggée à chaque appel (préférence de Mike pour
  l'auto-hébergement ; données potentiellement clients).
- Toute dégradation est annoncée dans le log (`_log`), jamais silencieuse.

## 5. Lots

### Lot 1 : moteur local (cz_ideogram.py)

1. Chargement de `Ideogram4Pipeline` depuis `ideogram_model` (défaut
   `ideogram-ai/ideogram-4-nf4-diffusers`), token HF depuis `hf_token`, offload
   none/model/sequential (le `model_cpu_offload_seq` du pipeline existe déjà).
2. Txt2img : steps, taille, seed, batch ; `guidance_schedule` exposé (préréglage officiel
   7,0 x 45 + 3,0 x 3) et `guidance_scale` constant en option.
3. `supports.negative = False` dans le protocole, champ négatif masqué dans l'UI avec
   la raison (branche non conditionnelle sans texte).
4. LoRAs via `Ideogram4LoraLoaderMixin`, avec une garde par famille (noms de tenseurs
   du transformer Ideogram) sur le modèle de `cz_flux3.family_from_keys`.
5. Mesures sur la 5090 : chargement, VRAM, temps à 1024 et 2048, 48 steps.

Acceptation : test CPU sur composants miniatures (comme `tests/test_flux3_adapter.py`),
puis une image 2048x2048 sur la 5090.

### Lot 2 : légendes JSON (le cœur de la valeur ajoutée)

1. **Éditeur de légende structurée** dans l'onglet Generate : sujet, liste d'éléments de
   texte (contenu exact, police ou style, boîte), palette (pastilles hex), style, fond.
   Produit le JSON attendu ; mode "JSON brut" pour l'éditer à la main.
2. **Boîtes dessinées à la souris** sur un canevas au ratio choisi, converties en `bbox`.
3. **Magic prompt, trois moteurs**, au choix dans les réglages :
   - local intégré (`prompt_upsampling=True`, Qwen3-VL-8B déjà chargé, `outlines` en
     dépendance pour garantir un JSON valide) : par défaut ;
   - Ollama (`cz_ollama.py`) avec le `CAPTION_SYSTEM_MESSAGE` publié par diffusers ;
   - API Ideogram (`/v1/ideogram-v4/magic-prompt`), opt-in, avertissement données.
4. Bouton "Describe" : image vers légende JSON (local via Qwen3-VL, cloud via l'API
   Describe 4.0 en option).
5. Presets de design : affiche, logo, carte de visite, couverture, slide, packaging,
   miniature YouTube, story verticale 9:16, bannière 6:1.

Acceptation : une légende construite à la main et une légende du magic prompt local
produisent toutes deux un JSON valide (validé contre le schéma) et une image.

### Lot 3 : img2img, inpaint, outpaint en local (émulés)

Ideogram 4 n'a ni img2img ni inpaint. On réutilise les techniques de crispz-flux3 :
1. **img2img (SDEdit)** : encoder l'image avec le VAE (même code que
   `Flux3Pipe._encode_image`), la bruiter au premier sigma conservé, ne parcourir que la
   queue du planning. Le pipeline calcule ses sigmas en interne (pas de paramètre
   `sigmas`) : sous-classer `Ideogram4Pipeline` ou envelopper son scheduler pour
   tronquer le planning selon `strength`. Attention au temps inversé du modèle.
2. **inpaint** : mélange latent à chaque step via `callback_on_step_end` (hors masque,
   remettre l'image d'origine bruitée au sigma atteint), masque max-poolé sur la grille.
3. Débloque automatiquement : refine, upscale par tuiles, détailleur visages et mains,
   reframe et outpaint de l'app.
4. `FAMILY_CAPS` : img2img et inpaint à True, avec mention "émulé" dans `caps`.

Acceptation : tests CPU équivalents à ceux de flux3 (étapes réduites, zone hors masque
conservée, Stop qui rend l'image partielle).

### Lot 4 : fonctions cloud Ideogram 4.5 (opt-in)

Module séparé `cz_ideogram_api.py`, désactivé tant que `ideogram_api_key` est vide et
que `ideogram_cloud_enabled` n'est pas à true. Chaque bouton cloud porte un badge
"Cloud" et un rappel que l'image part chez Ideogram (États-Unis).
Fonctions, d'après la documentation de l'API au 28/10/2026 (chemins exacts à relever) :
- **Edit précis 4.5** : instruction en langage naturel ou JSON, masque optionnel,
  jusqu'à 4 images de référence (3 avec masque), sortie à la taille de la source ;
- **Replace background 4.5** ;
- **Remix 4.0** ;
- **Remove background** ;
- **Vectorisation / génération SVG** ;
- **Upscale** (modèles Topaz côté Ideogram) ;
- **Transparent** (Ideogram 4.0 Transparent : fond alpha natif).
Chaque appel : estimation de prix avant envoi quand l'API la propose (`dry_run`),
journal des coûts, résultat rangé dans le dossier de sortie avec ses métadonnées.

Acceptation : tout fonctionne sans clé (boutons grisés avec explication) ; avec une clé,
un edit et un remove background aboutissent.

### Lot 5 : alternatives locales aux fonctions cloud

Pour rester utile sans cloud :
- fond transparent : bouton "Remove background" local déjà présent dans la famille ;
- vectorisation : vtracer ou potrace en local sur les logos à aplats ;
- édition : envoyer l'image vers crispz-flux3 (édition multi-références locale) par la
  route CLI du protocole v1 ; bouton "Ouvrir dans crispz-flux3".

### Lot 6 : tests, doc, livraison

- Suite CPU verte, test GPU e2e (gen 1024, gen 2048, img2img, inpaint) pour la 5090.
- README : licence non commerciale en tête, en clair ; ce qui est local et ce qui est
  cloud ; tableau des mesures.
- FORK.md : choix de crispz-flux3 comme base, différences, mesures.
- Launcher Pinokio `crispz-ideogram.pinokio` (Mike crée le repo vide).

## 6. Licence : à afficher dans l'app et le README

Ideogram 4 Non-Commercial : usage personnel et recherche autorisés ; **tout usage
commercial des poids auto-hébergés demande une licence négociée** avec Ideogram. Pour
Supersonique Studio : pas de livrable client produit en local avec ce modèle sans cette
licence. Les images faites via l'API Ideogram (lot 4) suivent les conditions de l'API,
qui autorisent l'usage commercial selon l'offre souscrite (à vérifier).

## 7. Hors périmètre v0.1

- Entraînement de LoRAs Ideogram.
- fp8 (32 Go et plus, ComfyUI seulement à ce jour).
- Ideogram 4.5 en local : impossible tant que les poids ne sont pas publiés ; à
  surveiller (une rumeur parle d'une sortie ouverte, non confirmée).

## 8. État au 31/10/2026

- `main` (2272b08) : crispz-flux3 0.1.0 importé avec son historique (8117fb2), renommé
  crispz-ideogram, `APP_VERSION` 0.1.0-dev. Notes de flux3 gardées dans `FORK-flux3.md`
  et `FORK-flux3-CLAUDE.md` (pièges d'environnement : à lire avant tout test GPU).
- **Lot 1 livré sur `wip/ideogram-engine`, validé le 01/11/2026 (bilan plus bas)** :
  - `cz_ideogram.py` : `IdeogramPipe` autour d'`Ideogram4Pipeline` (même convention
    d'appel que `Flux3Pipe`), planning de guidance officiel conservé en forme pour tout
    nombre de steps (le curseur = le pic, 7,0 par défaut), taille ramenée à la grille
    (16 px, 256 à 2048, ratio 6:1), négatif ignoré et dit une fois, `encode_prompt` masqué
    (le cache d'embeddings de cz_pipeline ne doit pas l'appeler), refus NOMMÉS pour
    img2img / inpaint (lot 3) et édition (lot 4), détection de famille (marqueurs forts
    `llm_cond_proj`... ; pour un LoRA, la largeur 4608 tranche avec Z-Image).
  - `cz_pipeline.py` : moteur `engine = cz_ideogram`, clés `ideogram_model`,
    `ideogram_transformer`, `ideogram_precision` (nf4), env `IDEOGRAM_*`, guidance = pic
    du planning, pas de négatif, sampler laissé tel quel, VRAM estimée 15 Go (fichiers),
    single-file refusé avec raison, LoRA FLUX.2 / FLUX.3 refusé en nommant leur studio.
    Le code FLUX.3 reste, non atteint, pour fusionner depuis crispz-flux3.
  - `cz_protocol` : `negative`, `img2img`, `inpaint`, `edit`, `refs` à False.
  - `cz_core` / `config-sample.txt` : profil `ideogram` 48 / 7,0, presets Ideogram (48)
    et Draft (24), aides réécrites. `cz_ui` : un seul repo, sélecteurs Distilled/Base et
    Precision masqués, champ négatif grisé avec la raison.
  - `requirements.txt` : `bitsandbytes>=0.45` (nf4), à confirmer au premier chargement.
  - Tests : `tests/test_ideogram_engine.py` (14) et `tests/test_ideogram_wiring.py` (10)
    sur un Ideogram 4 miniature aléatoire (`tests/ideogram_tiny.py`, vrai tokenizer
    Qwen3-VL-8B). Encodeur : `Qwen3VLModel`, pas `...ForConditionalGeneration`.
- Suite CPU : **518 passent, 44 échouent, 1 skip** ; smoke test 22/22. Les 44 échecs
  testent le comportement FLUX.3 (profils 16 / 50, Distilled/Base, dequant ComfyUI FLUX.3,
  fixtures de LoRA aux noms FLUX.3 désormais refusées) : à réécrire avec des fixtures
  Ideogram ou à retirer au lot 6, jamais à désactiver en silence.
- **À faire sur la 5090 avant de valider le lot 1** (app de Mike fermée, voir
  FORK-flux3-CLAUDE.md §3) : premier chargement du repo gated (quantizer nf4,
  bitsandbytes), une image 1024 et une 2048 à 48 steps, VRAM au repos et en pic, temps.
  Vérifier aussi l'aperçu live : la matrice latent vers RGB de flux3 suppose l'ordre de
  canaux FLUX (c, p, p), Ideogram range ses tokens en (p, p, c).

### Validation du lot 1, partie sans GPU (31/10/2026, session cloud, ni GPU ni token HF)

- **Quantizer** : `bitsandbytes`, nf4, `compute_dtype` bfloat16, sans double quant, sur
  les TROIS composants (les deux transformers ET l'encodeur Qwen3VLModel). Lu sur le
  miroir public `cocktailpeanut/ideogram-4-nf4` (le repo officiel répond 401 sans
  token). `requirements.txt` (`bitsandbytes>=0.45`) est donc juste, inchangé. La fiche
  officielle annonce « CUDA » comme seul matériel supporté.
- **Accès au repo gated** : non vérifiable sans le token de Mike ;
  `tools/gpu_lot1_check.py` le vérifie en premier (`auth_check`).
- **Lot 2, à noter** : le magic prompt local demande un composant de plus,
  `Ideogram4PromptEnhancerHead` depuis `diffusers/qwen3-vl-8b-instruct-lm-head` (déjà
  prévu dans `cz_ideogram.load(prompt_enhancer=True)`), et `outlines` pour forcer le JSON.
- **Aperçu live : couleurs fausses, corrigé.** Le VAE d'Ideogram 4 est identique bit pour
  bit à celui de FLUX.2 (les 251 tenseurs, stats `bn` comprises ; seul
  `num_batches_tracked` diffère). Mais Ideogram range ses tokens en (p, p, c) ET les
  normalise avec les stats `bn` telles que stockées, donc indexées dans l'ordre FLUX
  (c, p, p) : la position j reçoit la moyenne et l'écart type du canal FLUX j, pas les
  siens. Une permutation seule ne suffit pas ; il faut aussi une correction affine par
  entrée. Les deux se replient dans la matrice (`cz_pipeline._ideogram_preview_proj`,
  dérivée du VAE chargé, en cache), l'aperçu reste un seul matmul. Mesuré avec le vrai VAE
  sur CPU (3 images d'`assets/`, 512 px), erreur moyenne sur 0-255 : matrice FLUX telle
  quelle 52 à 61, matrice corrigée 13 à 24, soit exactement les chiffres de FLUX.2.
  Tests : `test_latent_preview.py` (algèbre contre les helpers FLUX.2 de diffusers et
  l'inverse exact du décodage Ideogram, stats bn aléatoires) et
  `test_ideogram_engine.py::test_live_preview_uses_the_ideogram_projection` (vrai callback
  de l'app sur le pipeline miniature).
- **Valeurs par défaut FLUX.3 restées dans l'app, corrigées** : `config-sample.txt`
  démarrait sur `Distilled (16 steps)`, 16 steps et guidance 1,0 (soit un pic de planning
  à 1,0, presque pas de CFG, tant qu'aucun preset n'était cliqué) ; le curseur de steps
  plafonnait à 40 (48 inatteignable). Désormais `Ideogram (48 steps)`, 48, 7,0 ; curseur
  steps 2 à 80, guidance 0 à 10 ; `GUIDANCE` vaut 7,0 si la config ne dit rien ;
  largeur / hauteur 256 à 2048 ; formats au-delà de 2048 retirés du menu (lot 2 refera la
  liste avec les presets de design) ; aides Performance, Aspect, Guidance réécrites.
  ATTENTION : un `config.txt` local déjà écrit garde ses anciennes valeurs, à vérifier.
- Suite CPU : **522 passent, 44 échouent (les mêmes 44 FLUX.3), 1 skip** ; smoke 22/22.
- **Mesure GPU : `tools/gpu_lot1_check.py`**, à lancer par Mike, app crispz-flux3
  FERMÉE (le script refuse de tourner si un autre processus tient la carte, `--force`
  pour passer outre). Passe par le vrai chemin de l'app (`get_pipe`, `generate`,
  `request_stop`, callback d'aperçu). Mesure : chargement, VRAM au repos (torch et
  pilote), 1024 à froid puis à chaud, 2048, 48 steps, prompt d'affiche « GRAND OPENING »,
  Stop depuis un autre thread (verdict : image rendue et arrêt avant le dernier step),
  aperçu live (images aux steps 1, 12, 24, 36, 48 avec les deux matrices, erreur contre
  l'image finale). Sortie : `bench/lot1/` (`report.md`, `results.json`, images).
  Validé à blanc sur CPU avec un Ideogram miniature sauvé sur disque.

### Premier passage sur la 5090 (31/10/2026, Mike, app fermée, offload auto)

- Accès gated : OK après acceptation de la licence sur le repo `-diffusers` lui-même
  (accepter sur `ideogram-4-nf4` ne suffit pas) et `hf auth login`.
- Chargement : 139,6 s au premier lancement, dont 2 min 06 de téléchargement (14,8 Go à
  117 Mo/s) ; 18,1 s depuis le cache. Au repos : 15,03 Go alloués par torch, 18,56 Go
  vus par le pilote (sur 31,84).
- 1024x1024, 48 steps, offload none : 82,6 s à froid (1,72 s/step), 86,1 s à chaud ;
  pic 20,21 Go réservés. Lent : environ 5 Go d'attention pour 6144 tokens.
- Aperçu live, dernier step contre l'image finale : erreur 43,1 avec la projection
  Ideogram contre 73,5 avec la matrice FLUX telle quelle (sur 0-255). Correctif confirmé
  dans le sens attendu, écart encore à regarder sur les images.
- 2048x2048 : premier step 65 s en offload none, la garde VRAM a basculé en 'model' et
  rechargé, puis 29,9 Go en VRAM et 90 s/step (débordement en mémoire partagée).
  Mesure arrêtée, deux causes trouvées et corrigées :
  1. **Fuite au repli 'model'** : `generate()` gardait la référence locale à l'ancien
     pipe pendant `free_vram()` puis le rechargement : deux copies de 15 Go. Le local
     est lâché avant (`test_vram_downgrade_frees_the_old_pipe_before_reloading`, qui
     échoue sur l'ancien code). Même motif dans `_refine_whole` : à revoir au lot 3,
     l'appelant y tient aussi le pipe.
  2. **Attention gonflée** : le pipeline brut réserve 2048 places de texte (rembourrage
     à gauche pour un prompt d'une centaine de tokens) : 6144 tokens à 1024 px, 18432 à
     2048 px, et l'encodeur lit aussi 2048 tokens. Il passe en plus un masque booléen à
     chaque attention, même entièrement vrai (toujours sur la branche non
     conditionnelle) : SDPA ne peut pas prendre le noyau flash et peut retomber sur le
     noyau math, en heads x L² (18 x 18432² en bf16 = 12 Go par appel à 2048 px).
     `cz_ideogram.FAST_ATTENTION` (défaut ; `IDEOGRAM_FAST_ATTENTION=0` pour couper)
     dimensionne la zone texte au nombre exact de tokens du prompt et retire un masque
     entièrement vrai. Même image : écart max 1,2e-6 sur le miniature en float32.
- `gpu_lot1_check.py` complété : diagnostic des noyaux SDPA à head_dim 256 avec et sans
  masque, et un rendu 1024 sans l'accélération (A/B : gain et écart en pixels).
- Suite CPU : 525 passent, les mêmes 44 échecs FLUX.3, 1 skip ; smoke 22/22.
- À relancer : la mesure complète, 2048 compris.

### Deuxième passage sur la 5090 (31/10/2026) et offload « encodeur seul »

- Le run est parti en 'model' : le repli du premier passage était mémorisé dans
  `cache/hw_profile.json` (« cached verdict »). Le script le signale désormais en tête.
- Mesures ('model' de diffusers, accélération d'attention active) : chargement 23,5 s ;
  1024 en 134 s à chaud (2,8 s/step), pic 13,1 Go ; le même sans l'accélération
  177 s (3,7 s/step), soit x1,32, écart moyen 8,7 sur 255 (images à juger) ; 2048 en
  464 s (9,7 s/step), pic 29,85 Go sur 31,8 (probablement le décodage VAE plein format,
  pas de débordement visible) ; Stop au step 15 sur 48, image rendue 4,8 s après ;
  aperçu : erreur 45 contre 79. Noyaux SDPA à head_dim 256 : `efficient` seulement
  (ni flash ni cuDNN), avec ou sans masque.
- **Offload 'model' de diffusers inadapté, remplacé** (`IdeogramPipe.enable_model_cpu_offload`) :
  1. sa chaîne `text_encoder -> transformer -> unconditional_transformer -> vae` renvoie
     le transformer conditionnel en RAM à chaque step (5,2 Go dans chaque sens) ;
  2. son crochet sur l'encodeur ne se déclenche jamais : le pipeline appelle les couches
     de l'encodeur une par une (`_get_text_encoder_hidden_states`), jamais son
     `forward()`. L'encodeur 8B tournait donc sur le CPU.
  Désormais : transformers et VAE restent sur la carte, l'encodeur seul y entre avant
  l'appel et repart au premier transformer (ou en fin d'appel, Stop compris). Aucun
  crochet accelerate, donc le chargeur de LoRA de diffusers ne réinstalle pas son propre
  offload (s'il revenait, il est remplacé et c'est dit une fois). Test :
  `test_model_offload_moves_only_the_text_encoder` (un aller-retour par image, jamais par
  step, transformers immobiles, image identique).
- Suite CPU : 526 passent, les mêmes 44 échecs FLUX.3 (vérifié nom par nom), 1 skip.
- À mesurer : 1024 et 2048 en 'model' nouvelle formule, puis 1024 en 'none' pour comparer.
  Pistes si le pic à 2048 reste à 30 Go : décodage VAE en tuiles au-delà d'un seuil.

### Troisième passage sur la 5090 : 'model' encodeur seul (31/10/2026)

- `--offload model --skip-raw`, `cache/hw_profile.json` supprimé avant. Chargement 20,2 s ;
  au repos 9,92 Go alloués (transformers + VAE), 17,05 Go vus par le pilote.
- 1024, 48 steps : **61,1 s à chaud (1,27 s/step)**, 62,1 s à froid ; pic 19,15 Go
  réservés. Contre 134 s en 'model' de diffusers et 86 s en 'none' brut du 1er passage.
- 2048, 48 steps : **400,9 s (8,35 s/step)**, pic 29,95 Go réservés sur 31,84 : trop
  près du plafond (un navigateur ouvert suffirait à faire déborder). Le script sépare
  désormais le pic du débruitage de celui du décodage VAE (`--only-2048`) pour
  décider d'un décodage en tuiles.
- Stop : arrêt au step 15 sur 48, image rendue 1,4 s après la demande.
- Aperçu : erreur 44,6 corrigé contre 78,2 (sur 0-255) ; verdict CHECK (seuil 40 du
  script), images à regarder avant de conclure.
- **Critère d'acceptation du lot 1 atteint** (une 2048x2048 sur la 5090). Restent à juger
  sur images : texte rendu, écart 8,7 accéléré contre brut, aperçu.

### Pic à 2048 : la normalisation du texte, pas le VAE (01/11/2026)

- `--only-2048` : 462 s (9,6 s/step, 2 Go déjà occupés par d'autres programmes au
  départ) ; pic du débruitage 29,88 Go, décodage VAE 2,4 s sans rien demander de plus.
  Le décodage en tuiles est donc inutile, abandonné.
- Cause : `llm_cond_norm` (RMSNorm sur 53248 = 13 couches de l'encodeur) passe sur TOUTE
  la séquence, environ 150 lignes de texte et 16384 lignes d'image nulles à 2048 px, et
  l'RMSNorm de diffusers convertit en float32 : 9 à 10 Go de temporaires par appel de
  transformer, deux appels par step.
- Correctif `cz_ideogram.install_sparse_text_norm` : la norme ne tourne que sur les
  lignes non nulles (une ligne nulle se normalise en ligne nulle), zéros ailleurs. Seule
  la norme est interceptée, pas `llm_cond_proj` que les LoRA peuvent viser. Suit
  `FAST_ATTENTION`. Test : égalité exacte en bf16. Les processeurs d'attention suivent
  aussi le drapeau à l'appel : le rendu « raw » de l'A/B du 2e passage gardait en partie
  l'accélération (écart 8,7 à re-mesurer).
- Suite CPU : 527 passent, les mêmes 44 échecs, 1 skip ; smoke 22/22.
- Mesuré (`--only-2048`, 'model' encodeur seul) : **pic du débruitage 23,35 Go** (contre
  29,88), 393,7 s soit **8,2 s/step** (contre 9,6 au run précédent), décodage VAE 1,1 s.
  Marge de 8,5 Go sur 31,84.
- **Offload 'auto'** (décision de Mike) : quand le test VRAM conclut 'none' sur une carte
  de moins de 40 Go (`cz_ideogram.AUTO_NONE_MIN_VRAM_GB`), Ideogram passe en 'model'
  encodeur seul, avec la raison dans le log. Un 'none' explicite n'est jamais touché.
  Test : `test_auto_offload_prefers_encoder_only_under_40_gb`. Suite : 528 passent, les
  mêmes 44 échecs, 1 skip ; smoke 22/22.

### Bilan du lot 1 (01/11/2026) : VALIDÉ par Mike

Acceptation remplie : tests CPU sur Ideogram miniature, et une 2048x2048 sur la 5090 avec
un texte correctement rendu (« GRAND OPENING », « SATURDAY 9 AM », vu par Mike).

Mesures finales (RTX 5090, Windows, torch 2.11 cu128, bitsandbytes 0.50.2, nf4,
offload 'model' encodeur seul, 48 steps, guidance 7,0) :

| | valeur |
|---|---|
| Chargement depuis le cache | 20 à 23 s (premier lancement : + 2 min de téléchargement, 14,8 Go) |
| VRAM au repos | 9,9 Go alloués (transformers + VAE), ~17 Go vus par le pilote |
| 1024x1024 | 61 s (1,27 s/step), pic 19,2 Go |
| 2048x2048 | 394 s (8,2 s/step), pic 23,4 Go sur 31,8, décodage VAE 1,1 s |
| Stop | image partielle rendue 1,4 s après la demande |

Gains obtenus pendant la validation, tous testés sur CPU et mesurés sur la carte :
- attention : zone texte à la taille du prompt, masques entièrement vrais retirés (x1,32) ;
- normalisation du texte sur les seules lignes de texte (pic 2048 : 29,9 -> 23,4 Go) ;
- offload 'model' propre à Ideogram, encodeur seul (1024 : 134 -> 61 s), et 'auto' qui
  le choisit sous 40 Go ;
- fuite de VRAM au repli 'model' corrigée ; aperçu live aux bonnes couleurs ;
- valeurs par défaut FLUX.3 retirées de l'app ; bitsandbytes ajouté au lock ;
  `ideogram_base_repos` documenté (job CI `check` vert).

Points ouverts, aucun bloquant, à reprendre dans les lots indiqués :
- aperçu live : erreur 45 sur 255 au dernier step (seuil du script 40, verdict CHECK) ;
  nettement mieux que la matrice FLUX (78), mais l'image d'aperçu n'a pas été regardée.
  Si les couleurs déçoivent dans l'app, réajuster la matrice sur des rendus Ideogram.
- A/B accéléré contre brut : l'écart de 8,7 mesuré au 2e passage n'est pas fiable (le
  rendu « brut » gardait une partie de l'accélération, corrigé depuis). Texte correct en
  mode accéléré ; un A/B propre reste à faire si un doute apparaît.
- LoRA Ideogram : garde par famille testée sur CPU seulement, aucun LoRA chargé sur la
  carte (à faire quand un LoRA Ideogram sera utilisé).
- Modération Hive annoncée par la fiche : aucune trace dans le code diffusers, rien dans
  le chemin local.
- `_refine_whole` : l'appelant tient aussi le pipe au repli 'model' (lot 3).
- Les 44 tests FLUX.3 et le job CI `tests` rouge : lot 6.
- Un `config.txt` local écrit avant le lot 1 garde 16 steps / guidance 1,0 : à vérifier
  chez Mike avant d'utiliser l'app.

### CI GitHub (31/10/2026)

- Job `check` : vert depuis 5846309 (`ideogram_base_repos` manquait à
  `config-sample.txt`, lu par `cz_ui` depuis le lot 1).
- Job `tests` : **rouge assumé jusqu'au lot 6, décision de Mike.** Les 12 fichiers en
  échec sont ceux des 44 tests FLUX.3 connus. Pas de liste d'échecs attendus, pas de
  xfail : ne pas « réparer » le CI avant le lot 6. Conséquence acceptée : le CI ne
  signale pas une régression pendant les lots 2 à 5, donc lancer la suite en local et
  comparer à la liste des 44 (`pytest tests -q`, mêmes noms) avant chaque push.
  `tools/run_tests.py` lance chaque fichier comme un script et s'arrête à la première
  erreur : dans ces 12 fichiers, des tests sains ne tournent pas en CI.

## 9. Sources

- diffusers 0.42.1 : `pipelines/ideogram4/pipeline_ideogram4.py`, `prompt_enhancer.py`
- https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers
- https://ideogram.ai/blog/ideogram-4.0/
- https://developer.ideogram.ai/api-reference/tools
- https://docs.ideogram.ai/using-ideogram/generation-settings/available-models
- https://www.orcarouter.ai/blog/ideogram-4-5
