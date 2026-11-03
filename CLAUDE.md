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
- Les 44 tests FLUX.3 et le job CI `tests` rouge : traités le 01/11/2026 (ménage).
- Un `config.txt` local écrit avant le lot 1 garde 16 steps / guidance 1,0 : à vérifier
  chez Mike avant d'utiliser l'app.

### Lot 2 (01/11/2026) : légendes JSON

- **Schéma de référence** : celui d'ideogram-oss/ideogram4 (docs/prompting.md +
  `caption_verifier.py`), plus riche que le sous-ensemble du magic prompt de diffusers :
  `style_description` (photo OU art_style, palette <= 16), éléments avec `bbox`
  [y0, x0, y1, x1] en 0..1000 et palette <= 5, ordre des clés strict, hex #RRGGBB
  majuscules, JSON minifié UTF-8. Le §2 parlait de `colour_palette` : la clé réelle est
  `color_palette`. Code Apache-2.0 repris tel quel dans `third_party/ideogram4/`
  (vérificateur, prompt v1), licence et provenance à côté.
- Filtre de sécurité : intégré aux POIDS (le modèle dessine lui-même une carte grise
  « Image blocked by safety filter »), pas de classifieur dans diffusers ; Hive n'existe
  que comme appel cloud optionnel dans le script de référence d'Ideogram. Faux positifs
  fréquents en texte libre, documentés par Ideogram (vu chez Mike sur une affiche
  anodine). Licence §4 : interdiction de contourner ou désactiver les filtres. Variante
  « NSFW » demandée par Mike : refusée pour cette raison.
- 2a `cz_caption` (construction, relecture, validation, boîtes, couleurs) ; une légende
  JSON contourne styles Fooocus et variantes {a|b} (`cz_prompt`, dit dans le log).
- 2b `cz_design` : 9 presets (affiche A, logo, carte 85x55, livre 2:3, slide 16:9,
  packaging, miniature YouTube, story 9:16, bannière 6:1), tailles natives.
- 2c `cz_magic` : magic prompt local (Qwen3-VL-8B + tête `diffusers/qwen3-vl-8b-instruct-
  lm-head`, `outlines` ajouté aux dépendances), Ollama (schéma passé en `format`), API
  Ideogram (désactivée par défaut, ligne `CLOUD:` dans le log à chaque appel).
- 2d `cz_caption_ui` : accordéon sous le prompt (presets, moteur, formulaire, tableau des
  éléments, plateau où l'on trace les boîtes à la souris, palette, JSON brut, « Use as
  prompt »). Vérifié dans Chromium : preset, boîte tracée posée sur le bon élément,
  JSON valide dans le prompt.
- 2e Describe local : l'encodeur nf4 contient la tour vision de Qwen3-VL (699 tenseurs
  `visual.*`) ; image -> légende avec boîtes demandées en `bbox_2d` (format natif Qwen,
  x1, y1, x2, y2) puis converties dans l'ordre Ideogram. Describe cloud : lot 4.
- `tools/gpu_lot2_check.py` : légende à la main + magic prompt local + Describe sur la
  5090, détecteur de carte « blocked », `--fp-test` (texte libre contre JSON).
- Suite CPU : 553 passent, les mêmes 44 échecs, 1 skip ; smoke 22/22.
- Mesure GPU faite, voir le bilan du lot 2 plus bas.
- **Piège (01/11/2026)** : ne JAMAIS faire `pip install -r requirements.txt` dans le
  `.venv` isolé. gradio 5.50 impose pillow<12, pip rétrograde donc Pillow 13 en 11.3
  (rembg exige >=12.1, et les CVE de Pillow ne sont corrigées qu'en 13). `install.bat`
  filtre Pillow, onnxruntime et rembg exprès (voir FORK-flux3-CLAUDE.md §3). Pour une
  nouvelle dépendance, l'installer seule à la version du lock :
  `pip install outlines==1.3.3 outlines-core==0.2.14`, puis si besoin
  `pip install --no-deps pillow==13.0.0`. Le même pip a remis `huggingface-hub` à
  1.33.0 (la version du lock) alors que le venv de Mike avait 2.4.0 : c'est bien la
  version attendue.

### Bilan du lot 2 (01/11/2026) : critère d'acceptation atteint sur la 5090

`tools/gpu_lot2_check.py --fp-test`, offload 'auto' -> 'model' encodeur seul, 48 steps,
outlines 1.3.3 installé :

| | résultat |
|---|---|
| Légende faite à la main (preset affiche, vrais textes) | schéma valide, image 720x1024 en 55,9 s, pic 16,7 Go |
| Magic prompt local, idée 1 (affiche suisse « 4.5 ») | 18,5 s, schéma valide, image en 66 s |
| Magic prompt local, idée 2 (logo café « BEAN THERE ») | 10,7 s, schéma valide |
| Premier magic prompt de la session | 43,7 s (chargement de la tête 1,24 Go et compilation du schéma outlines) |
| Describe local de l'affiche | 21,7 s, schéma valide, « GRAND OPENING » et « SATURDAY » relus |
| Cartes « blocked » en texte libre / en JSON | 1 sur 2 / 0 sur 2 (mêmes idées, même seed) |

- Sous Windows, outlines-core compile son noyau avec torch.compile sans Triton : deux
  pages de trace avant repli. Corrigé (dynamo coupé pendant l'appel sans Triton).
- Points ouverts : qualité des légendes générées et images à juger par Mike ; éditeur
  à essayer à la main dans l'app ; moteur Ollama jamais lancé contre un vrai serveur ;
  API Ideogram (magic prompt) jamais appelée avec une clé, et Describe cloud : lot 4.

### Lot 3 (01/11/2026) : img2img et inpaint émulés, acceptation CPU atteinte

- `IdeogramPipe._prepare_emulated` : SDEdit sur la fin du planning propre d'Ideogram
  (sigmas logit-normaux recalculés avec les fonctions de diffusers, mu selon la
  résolution, std 1,5, aucun décalage du scheduler), `strength` = fraction des steps
  (règle de crispz-flux3, au moins un step), image encodée en tokens Ideogram (inverse
  exact du décodage), `scheduler.set_timesteps` enveloppé le temps de l'appel, guidance
  décalée pour que la fin à 3,0 reste en fin. Masque : mélange latent à chaque step,
  max-pool sur la grille de 16 px ; masque tout blanc = img2img simple.
- Débloque refine, upscale par tuiles, inpaint, outpaint, reframe, détailleur.
  `cz_protocol` : img2img et inpaint à True, `supports.emulated`. UI : mention
  « émulé » ; mode « Outpaint LoRA » (FLUX.3) retiré du menu.
- Tests CPU (`tests/test_ideogram_emulated.py`, 8) : fin du planning, extrêmes de
  strength, image plus proche à faible strength, tokens hors masque EXACTEMENT gardés
  (comparés en latent : le décodeur VAE a une attention globale, les pixels ne se
  comparent pas), masque plein, Stop qui rend l'image partielle, max-pool, édition
  toujours refusée. Chemins de l'app testés (inpaint_run, _refine_whole, reframe,
  outpaint directionnel). Dossier extra des checkpoints : parcours récursif vérifié par
  un test (les fichiers uniques Ideogram y sont trouvés puis écartés, raison au log).
- `tools/gpu_lot3_check.py` : img2img 0,3 / 0,6, inpaint, outpaint, reframe sur la
  5090 (dry run CPU fait). Suite : 564 passent, les mêmes 44 échecs, 1 skip.
- Reste ouvert : `_refine_whole` au repli 'model' (l'appelant tient le pipe ; rare,
  img2img plafonné à ~1 Mpx) ; barre de progression de diffusers affichée sur le total
  des steps en img2img (cosmétique).

### Bilan du lot 3 (01/11/2026) : VALIDÉ par Mike

Acceptation remplie : tests CPU (`tests/test_ideogram_emulated.py` et chemins de l'app),
puis `tools/gpu_lot3_check.py` sur la 5090 (offload 'auto' -> 'model' encodeur seul,
48 steps, guidance 7, init = `bench/lot2/hand.png`, l'affiche du lot 2 en 720x1024).
5 rendus sur 5, sans erreur ni repli d'offload, 3 min 41 au total ; images jugées
correctes par Mike (texte gardé en img2img, zone hors masque intacte, raccords propres).

| | temps | taille |
|---|---|---|
| img2img 0,3 (~14 steps, chargement ~20 s compris) | 29,1 s | 720x1024 |
| img2img 0,6 (~29 steps) | 26,1 s | 720x1024 |
| inpaint (48 steps) | 40,4 s | 720x1024 |
| outpaint à droite | 57,0 s | 936x1024 |
| reframe 16:9 | 62,5 s | 1376x768 |

Environ 0,85 à 0,9 s/step à 0,7 Mpx, cohérent avec le lot 1 (1,27 s/step à 1 Mpx). Les
durées des deux img2img confirment que le planning est bien tronqué selon `strength`.

Points ouverts, aucun bloquant :
- `_refine_whole` au repli 'model' : l'appelant tient le pipe (rare, img2img plafonné
  à ~1 Mpx) ;
- barre de progression de diffusers sur le total des steps en img2img (cosmétique) ;
- presets d'upscale à 12 à 16 steps hérités de FLUX.3 distillé : non mesurés sur
  Ideogram (upscale par tuiles non lancé sur la carte), à régler si le rendu déçoit ;
- détailleur visages et mains non lancé sur la carte (même chemin que l'inpaint).

### Lot 4 (01/11/2026) : fonctions cloud Ideogram, acceptation sans clé atteinte

- **Chemins exacts relevés** dans la spec OpenAPI publiée
  (`https://developer.ideogram.ai/openapi/ideogram-api-v2.json`, lue le 01/11/2026). L'API
  v2 a un chemin par modèle ; toutes en multipart sauf Transparent (JSON) :
  | fonction | chemin | réponse |
  |---|---|---|
  | Edit précis 4.5 | `POST /v2/image/precise-edit/ideogram-4-5` | sync |
  | Replace background 4.5 | `POST /v2/image/replace-background/ideogram-4-5` | async |
  | Remix 4.0 | `POST /v2/image/remix/ideogram-4` | sync |
  | Remove background | `POST /v2/image/remove-background/ideogram-1` | sync |
  | Vectoriser | `POST /v2/svg/vectorize/ideogram-vector-1` | async |
  | Générer un SVG | `POST /v2/svg/generate/ideogram-vector-1` | async |
  | Upscale | `POST /v2/image/upscale/{auto, topaz-text-refine, ...}` | sync |
  | Transparent 4.0 | `POST /v2/image/generate/ideogram-4-transparent` | sync |
  | Describe 4.0 | `POST /v2/image/describe/ideogram-4` (pas de devis) | sync |
  | Résultat async | `GET /v2/generations/{id}` (`usage_cost_usd_micros`) | |
  Clé dans l'en-tête `Api-Key`. `?dry_run=true` valide et chiffre sans rien générer ni
  facturer, mais **envoie quand même l'image**. Le schéma du `PriceQuote` n'est pas publié :
  affiché et journalisé tel quel, montant repéré par le nom des champs.
- **Pièges de l'API relevés** :
  - masque de l'edit inversé par rapport à l'app : **noir = zone à éditer**, blanc = à
    garder, même taille que l'image, les deux zones présentes ; converti par `api_mask` ;
  - edit au-delà de 2048x2048 en surface : l'image revient RÉDUITE sauf avec un masque
    (`context_window=auto`) ; dit dans le log ;
  - 4 références au plus, 3 avec masque (le masque prend une place) ; l'excédent est dit ;
  - remove background : résultat PUBLIC par défaut selon l'offre ; `private=true` forcé
    partout où le champ existe (remove / replace background, SVG) ; remix garde la source
    dans le compte Ideogram (dit dans l'UI et le log) ; precise edit n'a pas le champ.
- `cz_ideogram_api.py` : constructeurs de requêtes (rien n'est envoyé à la construction),
  un seul point réseau (`_http`, remplacé dans les tests), ligne `CLOUD:` à chaque appel
  (ce qui part et où), devis, polling (Retry-After sur 429, échec avec sa raison, délai
  `ideogram_cloud_timeout`), résultats bloqués par la sécurité comptés, liens signés
  téléchargés SANS la clé, sauvegarde avec métadonnées (PNG + sidecar `.json` ; détourages
  et transparents toujours en PNG ; SVG en `.svg` + `.json`), journal des coûts
  `<sortie>/ideogram_cloud_journal.jsonl` (jamais la clé, jamais les images).
- `cz_cloud_ui.py` : onglet « Cloud (Ideogram) » (Input Image) : bandeau Cloud / États-Unis,
  réglages (case + clé masquée, sauvés dans preferences.json, clé jamais réaffichée),
  sélecteur de fonction avec aide, éditeur (le pinceau = masque de l'edit), 4 références,
  boutons « Cloud: Estimate price (dry run) » et « Cloud: Run », grisés avec la raison tant
  que c'est coupé ; état relu à chaque chargement de page. Résultats dans la galerie et
  l'historique. Describe cloud ajouté dans l'éditeur de légende (bouton Cloud séparé).
- `cz_protocol` : `supports.cloud` (activé ou non, fonctions, `via: ui`) ; le protocole
  ne lance jamais d'appel cloud (les données partiraient à la demande d'un appelant).
- Tests : `tests/test_ideogram_api.py` (28) sur un faux serveur. Vérifié dans Chromium avec
  un faux serveur : grisé, Save, rechargement, devis, run, fichier sauvé, log `CLOUD:`.
- Suite CPU : **631 passent, 0 échec, 1 skip** ; smoke 22/22.
- **Reste pour l'acceptation** (clé de Mike) : `tools/api_lot4_check.py` (statut, devis de
  toutes les fonctions), puis `--spend` (edit qualité low + remove background, facturés),
  sortie `bench/lot4/`.

### Bilan du lot 4 (02/11/2026) : clos sur l'acceptation SANS clé, cloud caché

- **Décision de Mike** : il n'utilisera pas le cloud Ideogram, seulement le modèle local.
  Le code du lot 4 est GARDÉ mais CACHÉ tant que `ideogram_cloud_enabled` vaut false (lu
  au démarrage) : pas d'onglet Cloud, pas de bouton Describe cloud, pas de moteur « Ideogram
  API » dans le magic prompt. Un moteur cloud resté dans les préférences retombe sur le
  local, dit une fois dans le log. Pour le réactiver : `"ideogram_cloud_enabled": true`
  dans config.txt, redémarrer, puis la clé dans l'onglet Cloud.
- Acceptation sans clé vérifiée chez Mike : `tools/api_lot4_check.py` refuse et n'envoie
  rien. La partie « avec une clé » n'a PAS été faite : aucun appel réel n'a jamais été
  envoyé à l'API, les chemins et champs ne sont vérifiés que contre la spec publiée.
- Tests : 4 de plus dans `test_ideogram_api.py` (entrées cloud cachées quand c'est coupé,
  visibles quand c'est activé, repli du moteur). Suite : 635 passent, 0 échec, 1 skip.
- Conséquence pour le lot 5 (alternatives locales) : il devient le seul chemin pour le
  détourage, la vectorisation et l'édition.

### Lot 5 (02/11/2026) : alternatives locales, livré

Le seul chemin pour le détourage, la vectorisation et l'édition depuis que le cloud est
caché. Rien ne quitte la machine.
- **Remove BG local** (rembg, u2net), onglet déjà là : vérifié sur CPU (détourage net,
  55 s au premier appel dont le téléchargement du modèle de 176 Mo, puis 1,1 s). Le PNG
  sauvé porte désormais ses métadonnées (`local`, `rembg (u2net)`).
- **Vectorize (SVG)**, nouvel onglet, `cz_vector.py` : vtracer 0.6.15 (MIT, wheels
  Windows 3.10 à 3.14, ajouté à `requirements.txt` et au lock). Quatre presets (logo à
  aplats, illustration, noir et blanc, angles vifs), réduction des couleurs avant le tracé
  (le bruit d'un rendu de diffusion devient sinon des formes), détourage local en option
  (plus de rectangle de fond), au-delà de 4096 px réduit et dit. Aperçu par `<img>` en
  data URI (jamais de SVG inline), fichier `.svg` + `.json` dans la sortie. Mesuré : un
  logo, 3 chemins en 0,05 s ; une photo 768 px, environ 2 000 chemins et 2 Mo en 3 s,
  signalé comme lourd.
- **Edit in crispz-flux3**, nouvel onglet, `cz_handoff.py` : l'image (et jusqu'à 3
  références) écrites en PNG sous `<sortie>/handoff/<date>/`, puis le CLI protocole v1 DE
  crispz-flux3 (`cz_protocol.py`, lancé avec son propre `.venv`). Son app tourne : l'edit
  passe par son instance. Sinon : `--local`, crispz-ideogram libère sa VRAM d'abord (dit).
  Une instance qui répond sous le nom crispz-ideogram (même port dans les deux configs)
  n'est pas prise pour crispz-flux3. Réglage `crispz_flux3_dir` (champ dans l'onglet, env
  `CRISPZ_FLUX3_DIR`). Ligne `HANDOFF:` dans le log, `handoff.json` à côté des fichiers.
  **Piège relevé dans crispz-flux3 (f371426)** : l'op `edit` remplace `spec.refs` par la
  seule image d'entrée ; avec des références on envoie donc `gen` + `refs` (même voie
  multi-références) à la taille de l'image. Les deux specs passent le vrai
  `validate_spec` de crispz-flux3 sans avertissement.
- **Reste FLUX.3 trouvé** : l'onglet « Edit (references) » et le mode d'entrée du même
  nom étaient visibles (`omni_on` ne regardait que `OMNI_MODEL`, que le code FLUX.3 pose
  au repo de base) ; un clic aboutissait à un refus. Désormais `omni_on` suit
  `cz_protocol.OMNI_SUPPORTED` (False), test à l'appui.
- Tests : `test_vector.py` (10), `test_handoff.py` (9, dont un vrai sous-processus contre
  un faux dossier crispz-flux3), `test_local_tools.py` (3). Vérifié dans Chromium
  (Vectorize, onglet crispz-flux3 grisé avec sa raison). Suite : 657 passent, 0 échec,
  1 skip ; smoke 22/22.
- À essayer chez Mike : l'onglet crispz-flux3 avec son vrai dossier, app crispz-flux3
  ouverte puis fermée. Ports : Gradio donne 7861 à la seconde app lancée, et crispz-flux3
  cherche son instance sur 7860 : lancer crispz-flux3 en premier (sinon route à froid, dite).

### Lot 6 (02/11/2026) : tests, doc, livraison (sur `wip/ideogram-engine`)

- **Test GPU e2e** `tests/test_ideogram_e2e.py` (remplace `test_flux3_e2e.py`) : chargement,
  gen 1024, gen 2048 (`CZ_E2E_2048=0` pour le sauter), img2img 0,35, inpaint (hors masque
  identique au pixel près), Stop au step 3 ; temps et pics dans `tests/e2e_timings.txt`.
  Se saute sans CUDA ou sans accès au repo. **Pas encore lancé sur la 5090.**
- **README.md** réécrit (anglais) : licence non commerciale en tête, ce qui est local et ce
  qui est cloud (caché), mesures de tous les lots, conseils (JSON, filtre), crispz-flux3 et
  les ports. **FORK.md** (français) : choix de la base, différences, mesures, pièges.
  **README_CLI.md** : exemples Ideogram (48 steps, guidance 7, légende JSON),
  `--ideogram-model` / `--ideogram-transformer` (anciens noms gardés en alias),
  `--expand-lora` refusé AVANT le téléchargement. **CHANGELOG** : entrée 0.1.0-dev.
- **Launcher Pinokio** préparé dans `pinokio/` (le repo `crispz-ideogram.pinokio`
  n'existe pas encore) : repris de `crispz-flux3.pinokio`, vérification
  `Ideogram4Pipeline` + bitsandbytes + outlines + vtracer, message de licence et de token,
  README. L'icône est celle de crispz-flux3 (la lune), à remplacer. Il clone `main` :
  utile seulement après la fusion de `wip/ideogram-engine` dans `main`.
- **Interface** : plus aucun libellé FLUX.3 visible (sampler, schedule, offload, encodeur,
  CivitAI « Ideogram only », messages de modèle), accordéon « Edit model » caché ; test
  `test_no_visible_flux3_wording_in_the_ui` (parcourt l'UI construite, composants
  visibles seulement).
- **Tirets cadratins** : plus aucun dans le repo hors `third_party/` (530 remplacés par
  « : » ou « , », CHANGELOG et vieux FORK compris).
- Suite CPU : 658 passent, 0 échec, 1 skip (le e2e GPU) ; smoke 22/22.
- **Reste à décider par Mike** : fusion de `wip/ideogram-engine` dans `main`, passage de
  `APP_VERSION` de 0.1.0-dev à 0.1.0, création du repo `crispz-ideogram.pinokio` (et son
  icône), lancement du e2e GPU.

### Lot 7 (03/11/2026) : points ouverts des lots 1 et 3, sans GPU

Mike absent (« avance sur le prochain lot ») ; les six lots du cahier étant livrés, ce
lot traite les points ouverts techniques qui ne demandent pas sa décision. Son dossier
local n'est pas accessible depuis la session cloud (Remote Control : dossier non proposé),
le e2e GPU du lot 6 reste donc à lancer chez lui.
- **`_refine_whole` au repli 'model'** (ouvert depuis le lot 1) : l'appelant tient le
  pipe pendant `free_vram()`, les 15 Go restaient sur la carte au rechargement.
  `IdeogramPipe.release()` lâche le vrai pipeline ; `free_vram` l'appelle sur tout pipe
  Ideogram en cache ; une référence périmée est remplacée par le pipe courant
  (`cz_pipeline._fresh`, dans `_qwen_call` et `_refine_whole`), un usage direct lève une
  erreur claire. Test : `test_refine_whole_downgrade_frees_the_weights_the_caller_still_holds`
  (échoue sans `release`, vérifié).
- **Presets d'upscale et refine** : dimensionnés pour FLUX.3 distillé (12 à 16 steps),
  soit 3 ou 4 steps réellement joués à denoise 0,3 sur Ideogram. Multipliés par 3 (48/16) :
  24 à 48 selon le preset, `default_refine_steps` 36, curseur 4 à 80. **À juger sur
  images** : `tools/gpu_lot7_check.py` compare 12 et 36 steps sur l'affiche.
- **Compteur de l'img2img émulé** : la barre de diffusers se dimensionnait sur le planning
  entier (14/48 puis arrêt) et l'aperçu live disait « step 5 sur 48 ». Les deux comptent
  désormais les steps joués (`_cz_run_steps`), tout est remis en place après l'appel.
- **`tools/gpu_lot7_check.py`** : upscale x2 + refine à 12 puis 36 steps (même seed),
  upscale x4 en tuiles (preset 4K, `--skip-tiled` pour le sauter), détailleur visages sur
  un portrait généré ; temps et pics dans `bench/lot7/report.md`. Répété à blanc sur CPU.
- Suite CPU : 661 passent, 0 échec, 1 skip ; smoke 22/22.
- À faire chez Mike : `git pull`, `pip install vtracer==0.6.15`, le e2e du lot 6, puis
  `tools\gpu_lot7_check.py` (app et crispz-flux3 fermées).

### e2e GPU du lot 6 sur la 5090 (03/11/2026) : PASSÉ, temps expliqués

`tests/test_ideogram_e2e.py`, offload 'model' : les 6 étapes passent leurs assertions
(chargement, gen 1024, gen 2048, img2img 0,35, inpaint hors masque identique au pixel,
Stop au step 3). Pics de VRAM normaux (17,3 Go à 1024, 23,4 Go à 2048).

| | e2e | référence lots 1 à 3 |
|---|---|---|
| chargement | 40,8 s | 20 à 23 s |
| 1024, 48 steps | 112,3 s | 61 s |
| 2048, 48 steps | 734,6 s | 394 s |
| img2img 0,35 | 36,8 s (2,2 s/step) | ~1,3 s/step |
| Stop (3 steps) | 5,4 s | |

Tout était environ 1,8 fois plus lent, chargement compris. **Cause : la machine, pas le
code.** Carte non bridée (2 790 MHz, 520 W, 99 %, 72 °C) mais partagée : deux processus
Python 3.10 étrangers à crispz-ideogram (une app Pinokio probablement) tenaient le GPU.
Contre-mesure le même jour, carte libérée, `gpu_lot1_check.py --offload model --skip-raw
--skip-2048` : chargement 18,6 s, 1024 en 63,1 s à froid et **60,6 s à chaud
(1,26 s/step)**, Stop rendu 0,6 s après la demande, aperçu 44,6 contre 78,2 : les
chiffres du lot 1. Les temps du README restent valables. Leçon : vérifier que la carte est
libre (`nvidia-smi`, 0 à 5 % d'utilisation) avant toute mesure.

### Bilan du lot 7 (03/11/2026) : mesuré sur la 5090, clos

`tools/gpu_lot7_check.py`, carte libre, offload 'model' encodeur seul, affiche du lot 2 :

| | temps | pic |
|---|---|---|
| portrait 1024 (pour le détailleur) | 76,4 s | 17,3 Go |
| upscale x2 + refine 0,30, 12 steps | 35,6 s | 16,4 Go |
| upscale x2 + refine 0,30, 36 steps | 54,4 s | 16,3 Go |
| upscale x4 en tuiles (preset 4K) | 167,5 s, 2880x4096 | 17,5 Go |
| détailleur visages | 37,5 s | 16,5 Go |

- **12 contre 36 steps** : images identiques à l'œil (écart moyen 2 sur 255, textes
  inchangés au trait près à 100 %), 36 coûte 53 % de plus. **Décision : retour aux presets
  d'origine** (12 à 16 steps, `default_refine_steps` 12) ; le curseur reste à 80.
  L'hypothèse « 3 ou 4 steps joués, c'est trop peu » ne tient pas à ces denoise.
- **Tuiles x4** : aucune couture visible, texte intact, VRAM sous 18 Go à 2880x4096.
- **Détailleur** : visage trouvé et retouché (dents, yeux plus nets), sans couture, reste
  de l'image inchangé. Validé sur la carte.
- Ainsi se ferment les points ouverts des lots 1 et 3 (refine au repli, presets,
  compteur, tuiles, détailleur). Suite CPU : 661 passent, 0 échec, 1 skip ; smoke 22/22.

### Livraison 0.1.0 (03/11/2026), décision de Mike

- `APP_VERSION` 0.1.0 ; `wip/ideogram-engine` fusionnée dans `main` (avance rapide, sans
  commit de fusion : `main` n'avait pas bougé depuis 2272b08).
- Launcher publié dans son propre repo `mikecastrodemaria/crispz-ideogram.pinokio` (créé
  par Mike) ; le dossier `pinokio/` quitte ce repo (une seule source). Icône provisoire
  « Aa » aux couleurs de l'affiche du lot 2 (Inter Display Black, licence OFL), à
  remplacer par une icône générée avec l'app (preset Logo) si Mike le souhaite.
- Suite des travaux : depuis `main`, sur une nouvelle branche par sujet.

### CI GitHub (31/10/2026)

- Job `check` : vert depuis 5846309 (`ideogram_base_repos` manquait à
  `config-sample.txt`, lu par `cz_ui` depuis le lot 1).
- Job `tests` : rouge assumé jusqu'au ménage du 01/11/2026 (voir plus bas), avancé à la
  demande de Mike. Depuis : suite entière verte, en local comme en CI. Une régression
  se voit donc de nouveau dans le CI.

### Ménage des tests hérités (01/11/2026), demandé par Mike avant le lot 4

Les 44 échecs FLUX.3 traités un par un, aucun désactivé, aucun xfail :
- **Réécrits pour Ideogram** (même intention, fixtures aux noms de tenseurs Ideogram,
  `llm_cond_proj`, `layers.N.attention.*`, largeur 4608) : `test_quant_formats` (23),
  `test_lora_dialect`, `test_lora_offload`, `test_lokr_merge`, `test_prescaled_fp8`,
  `test_protocol`, `test_edit_lora_presets` (un LoRA FLUX.3 est désormais refusé en
  nommant crispz-flux3), `test_encoder_trim`, `test_detailer_steps` (profil 48),
  `test_preset_model_switch` (un seul repo de base, fichier unique refusé),
  `test_klein_variant` (FLUX.2 et FLUX.3 refusés en nommant leur studio, Ideogram passe).
- **Renommé et réécrit** : `test_flux3_profiles.py` devient `test_model_profiles.py`.
- **Retirés** (comportement FLUX.3 que l'app refuse désormais, couvert côté Ideogram par
  `test_ideogram_wiring` / `test_ideogram_engine`) : dans `test_flux3_wiring.py`, guidance
  par variante et chargement des fichiers uniques FLUX.3 (le test GGUF devient « un GGUF
  FLUX.3 est refusé avant tout chargement ») ; `test_undistilled_profile.py` en entier
  (fichiers FLUX.2 Klein distillés ou non). Le reste de `test_flux3_wiring` garde le
  chemin FLUX.3 de `cz_pipeline` exécutable pour les fusions depuis crispz-flux3.
- **Vrais défauts trouvés dans le code** :
  - profil par défaut à 16 steps / guidance 1,0 (`config-sample.txt` et repli de
    `cz_core.profile_for_model`) : un modèle au nom inconnu (dossier local renommé,
    miroir) tournait à 16 steps avec un pic de guidance à 1,0. Désormais 48 / 7,0 ;
  - le sélecteur de précision sauvait `flux3_precision`, le moteur lit
    `ideogram_precision` ; un message citait `flux3_base_repos` ;
  - avertissement faux « guidance sans effet » dans `cz_protocol`, et refs ignorées
    expliquées en termes FLUX.3 ;
  - messages de refus (dequant, variantes, base) qui parlaient de FLUX.3 ;
  - changement de repo de base : le cache de dimension n'était pas purgé ;
  - aides de `config-sample.txt` qui décrivaient FLUX.3 (encodeur, trim, empreinte
    VRAM, édition, négatif, détailleur, LoRAs).
- Suite CPU : **603 passent, 0 échec, 1 skip** (`pytest tests -q`) ; `tools/run_tests.py`
  (ce que lance le CI) vert ; check_config_keys, check_lock, compileall OK.
- Reste hérité, hors ménage des tests : ~600 tirets cadratins dans le code et la doc de
  la famille (commentaires, CHANGELOG, README_CLI...), à retirer au lot 6 ;
  `third_party/ideogram4` est repris tel quel et n'est pas touché.

## 9. Sources

- diffusers 0.42.1 : `pipelines/ideogram4/pipeline_ideogram4.py`, `prompt_enhancer.py`
- https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers
- https://ideogram.ai/blog/ideogram-4.0/
- https://developer.ideogram.ai/api-reference/tools
- https://docs.ideogram.ai/using-ideogram/generation-settings/available-models
- https://www.orcarouter.ai/blog/ideogram-4-5
