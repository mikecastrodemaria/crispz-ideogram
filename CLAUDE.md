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
- **Lot 1 livré sur `wip/ideogram-engine`, à valider sur la 5090** :
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

## 9. Sources

- diffusers 0.42.1 : `pipelines/ideogram4/pipeline_ideogram4.py`, `prompt_enhancer.py`
- https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers
- https://ideogram.ai/blog/ideogram-4.0/
- https://developer.ideogram.ai/api-reference/tools
- https://docs.ideogram.ai/using-ideogram/generation-settings/available-models
- https://www.orcarouter.ai/blog/ideogram-4-5
