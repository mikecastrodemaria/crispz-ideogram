# crispz-flux3 : cahier des charges pour Claude Code

Ce fichier est lu automatiquement par Claude Code à l'ouverture du repo. Il décrit
le contexte, les règles, l'état exact du portage FLUX.3 et les actions restantes,
dans l'ordre, avec leurs critères d'acceptation.

## 1. Contexte et objectif

crispz-flux3 est un fork de crispz-klein (studio Gradio local, FLUX.2 Klein) porté sur
**FLUX.3 Image Klein** (`black-forest-labs/flux-3-image-klein`, 5,6B, sorti le
15/10/2026, FLUX Kommunity License, repo HF à accès conditionnel).

Objectif de la v0.1.0 : la même surface fonctionnelle que crispz-klein (txt2img, édition
multi-références, img2img/refine/upscale, inpaint/outpaint, détailleur, LoRAs FLUX.3,
checkpoints single-file FLUX.3, CLI et protocole v1) sur FLUX.3, installable via
`install.bat`, testable sur RTX 5090 (32 Go).

Hors périmètre v0.1.0 : prompt upsampler `flux-3-image-klein-pup` (9B, ~28 Go),
remplacement de l'encodeur texte, samplers autres qu'Euler, bounding boxes dans l'UI
(prévus en v0.2, voir §8).

## 2. Règles de travail (non négociables)

- **Commits au nom de Mike** : `Mike Castro Demaria <supersoniquestudio@gmail.com>`,
  auteur et committer. **Aucune** ligne `Co-Authored-By: Claude` ni lien de session.
- Jamais de tiret cadratin (le caractère U+2014) ni d'emoji, ni dans le code, ni dans les
  commits, ni dans la doc. Utiliser `:` ou `,`.
- Travail **étape par étape** : à la fin de chaque lot (§5), s'arrêter, résumer, et
  attendre la validation de Mike avant le lot suivant.
- Branche de travail : `wip/flux3-engine`. Ne jamais pousser sur `main` avant le lot 6.
- Style du code existant : anglais, commentaires longs qui expliquent le POURQUOI
  (mesures, incidents datés). Le conserver. Toute dégradation est **annoncée** dans le
  log (`_log`), jamais silencieuse (règle maison).
- Garder les noms publics du module (`ZIMAGE_TRANSFORMER`, `generate_omni`, `_qwen_call`,
  `BASE_REPO`...) : cz_ui, cz_cli et cz_protocol les importent, et ils réduisent les
  conflits de merge depuis crispz-klein (remote `klein`).

## 3. Environnement de test

Pas de GPU nécessaire pour les tests unitaires. Pile validée sur CPU (Python 3.13) :

```bash
python -m venv .venv-test && . .venv-test/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "diffusers==0.42.1" "transformers>=5,<6" accelerate peft gguf sentencepiece \
            spandrel "gradio>=4.44,<6" pillow numpy scipy fastapi uvicorn pytest
python -m pytest tests -q --ignore=tests/test_klein_e2e.py
```

Sous Windows, le venv validé est `.venv-test` à la racine (Python 3.13, torch 2.14 cpu,
diffusers 0.42.1, transformers 5.21) : `.venv-test/Scripts/python -m pytest tests -q
--ignore=tests/test_klein_e2e.py`.

Pièges déjà rencontrés :
- **Jamais de test GPU pendant que l'app de Mike tourne** (28/10) : un rendu de test lancé par
  Claude a chargé un second FLUX.3 (19 Go) à côté de celui de l'app ; 38 Go sur 32, la
  carte a débordé en mémoire partagée (5 Go) et l'app de Mike a planté. `nvidia-smi` au
  début ne suffit pas (l'app charge le modèle à la première image) : demander à Mike de
  fermer l'app, ou qu'il donne son feu vert, avant tout `.venv-lock` sur la carte.
- `boot_check.bat` sans `.venv` (28/10) : le diagnostic tournait sur `py -3.10` et `run.bat`
  lançait `py -3.13` sans gradio, d'où un crash au démarrage. Les trois .bat suivent la
  même règle (`.venv`, puis `env` de Pinokio, puis Python 3.13 > 3.12 > 3.11) et la
  vérification diffusers/gradio est bloquante. Sur une machine sans `.venv` : `install.bat`.
- `.gitignore` excluait tout `tests/` (règle héritée pour des images locales) : le premier
  `tests/test_flux3_adapter.py` n'a jamais été commité et a dû être réécrit. Corrigé en
  `tests/*` + `!tests/*.py` ; vérifier `git status` après tout nouveau fichier de test.
- rembg contre gradio (28/10) : tout rembg depuis 2.0.75 exige pillow>=12.1, gradio 5.50 exige
  pillow<12 : demandés ensemble, pip et uv refusent (ResolutionImpossible). Les installeurs
  filtrent pillow, onnxruntime ET rembg du lock, posent Pillow 13 en --no-deps, puis rembg
  seul ; le launcher Pinokio installe requirements-extra.txt après Pillow 13.
- torchvision doit venir du MÊME index que torch (`+cpu`), sinon
  `operator torchvision::nms does not exist` casse tout import transformers.
- diffusers 0.42.1 exige `huggingface-hub>=1.32`, incompatible avec transformers 4.x
  (`<1.0`) : transformers **5.x obligatoire**.
- Le repo officiel est gated (401 sans token). Pour lire les configs sans token :
  le miroir `unsloth/flux-3-image-klein-FP8` (configs identiques, `modular_model_index.json`,
  `transformer/config.json`, `vae/config.json`, `text_encoder/config.json`).

## 4. Architecture retenue (décisions déjà prises, ne pas rediscuter)

**Un adaptateur plutôt qu'une réécriture.** `cz_flux3.py` contient `Flux3Pipe`, qui
enveloppe le `Flux3DistilledModularPipeline` / `Flux3ModularPipeline` de diffusers derrière
la convention d'appel Flux2Klein utilisée par les ~30 call sites de `cz_pipeline.py` :

```
pipe(prompt=, image=, mask_image=, strength=, width=, height=, num_inference_steps=,
     generator=, guidance_scale=, negative_prompt=, prompt_embeds=,
     callback_on_step_end=, output_type=).images
```

Points clés de `Flux3Pipe` (déjà codés et testés) :
- Chaîne de blocs = blocs officiels **moins** `prompt_upsampler` et `text_encoder`, plus
  un bloc `crispz_hook` à la fin de chaque boucle de débruitage.
- L'encodage texte (Qwen3-VL-4B, 8 couches cachées empilées, padding multiple de 80) est
  fait par `encode_prompt()` : le cache d'embeddings de cz_pipeline fonctionne tel quel.
- **img2img** : SDEdit sur le chemin flow-matching (`img2img_schedule`) : on encode
  l'image, on la bruite au premier sigma conservé, on passe la queue du planning
  (sigmas non décalés, le scheduler applique lui-même le shift 6,93).
- **inpaint** : à chaque pas, hors masque, les tokens sont remis à l'image d'origine
  bruitée au sigma atteint (masque max-poolé sur la grille de 16 px). Un masque tout
  blanc = img2img simple.
- `callback_on_step_end(pipe, i, t, {"latents"})` et `pipe._interrupt` (Stop, garde VRAM,
  aperçu live) via le hook ; un Stop décode les derniers latents.
- Offload `model` / `sequential` via accelerate, avec les noms de DiffusionPipeline
  (`enable_model_cpu_offload`, `maybe_free_model_hooks`, `_all_hooks`, `remove_all_hooks`).
- Attributs inconnus redirigés vers le pipeline modulaire (`pipe.transformer`,
  `pipe.load_lora_weights` via `Flux3LoraLoaderMixin`...).

**Garde par FAMILLE, pas par dimension.** FLUX.3 a la même largeur cachée (3072) que
FLUX.2-klein-4B : l'ancien tri 4B/9B laisserait passer un fichier FLUX.2.
`cz_flux3.family_from_keys()` lit les noms de tenseurs (marqueurs
`early_stream_modulations.`, `mode_blocks.`... contre `double_blocks.`,
`single_stream_modulation.`...).

Piège diffusers : `ModularPipeline.blocks` renvoie une **copie** ; les vrais blocs sont
dans `mp._blocks` (c'est là qu'on rattache `hook.owner`).

## 5. État au 28/10/2026 (lots A à F livrés, 0.1.0 fusionnée dans `main`)

Après la 0.1.0 (28/10 soir) : 2K et 4K rendus natifs mesurés (3840x2160 en 113 s), presets
2K / 4K dans Settings, curseurs jusqu'à 4096 ; fichiers CivitAI : bf16 single-file OK,
INT8 comfy-quant déquantifié (fused qkv séparé, variante lue dans le fichier : le
"Flux 3 Klein INT8" est le modèle BASE, config et blocs suivent le fichier), INT6 w6a8
refusé dès le menu, GGUF Q4_K_M chargeable une fois le dtype de calcul épinglé
(`cz_flux3._pin_compute_dtype` : diffusers annonce uint8 comme dtype d'un GGUF). CivitAI
batch : GGUF inclus, note de hachage, catalogue rafraîchi. Validés sur CPU ; INT8 et GGUF
restent à rendre sur la 5090 (demander la carte à Mike).

Fait au lot F : README réécrit pour FLUX.3 avec les captures réelles (`assets/`, de nouveau
versionné : capture de l'onglet Text to Image prise dans l'UI sur la 5090, rendus gen / edit /
inpaint du e2e), README_CLI et drapeaux CLI `--flux3-model` / `--flux3-transformer` (les
anciens restent des alias), cache dequant retiré (`tools/rebuild_dequant_cache.py`,
`rebuild_cache.bat`, clés `dequant_cache*`), `APP_VERSION` 0.1.0, CHANGELOG 0.1.0, FORK.md
(mesures et feuille de route). Launcher Pinokio livré : mikecastrodemaria/crispz-flux3.pinokio
(venv Python 3.11, torch 2.11 cu128, garde sur les blocs modulaires FLUX.3).

Fait au lot E : les 47 tests FLUX.2 sont réécrits avec la même intention sur FLUX.3 (garde
par famille, libellés FLUX.3 / FLUX.2, profils 16 / 50, catalogue d'edit LoRAs vide mais
déclarable en config) ou retirés avec leur ligne dans le CHANGELOG
(`test_precache_variants.py`, `test_prefixed_single_file_is_loaded_as_a_state_dict`).
`_gguf_layout` reconnaît un GGUF FLUX.3 ; `_preset_ckpt_refusal` nomme crispz-klein pour un
fichier FLUX.2 au lieu de proposer un repo. `tests/test_flux3_e2e.py` (GPU, 6 étapes :
chargement, gen, edit 1 ref, edit 2 refs, inpaint, refine, plus le Stop) se saute sans
CUDA ou sans accès au repo. Suite CPU : **533 passent, 0 échec, 1 skip (le e2e)**.

**Le e2e GPU est passé sur la 5090** (`tests/test_flux3_e2e.py`, bf16, offload none) :
chargement 50,6 s à froid / 21,8 s à chaud, txt2img 1024x1024 16 steps 6,6 à 7,1 s, édition
1 ref 13,5 s, 2 refs 23,4 s, inpaint 3,6 à 5,7 s, refine 0,35 2,3 s, 18,8 Go résidents, pic
23,7 Go. Détail et leçons dans FORK.md (édition timide sans upsampler : identique au pixel
près au chemin diffusers brut, donc le modèle ; l'inpaint veut le prompt de toute la scène ;
la force img2img est une fraction des steps, règle diffusers, corrigée après mesure).
Les images sont dans `tests/e2e_*.png` (local), à reprendre pour le README au lot F.

Fait au lot D : `requirements.txt` sur `diffusers==0.42.1` (PyPI), `transformers>=5,<6`,
`huggingface-hub>=1.32` ; `requirements-lock.txt` régénéré depuis un venv vierge Python
3.13 + torch 2.11.0 cu128 (`import app` et les blocs modulaires FLUX.3 vérifiés) ;
**Python 3.11 minimum** (diffusers 0.42.1 et Pillow 13 ne sont pas publiés pour 3.10) :
install.bat/.sh, run.bat/.sh, update.bat/.sh choisissent 3.13 > 3.12 > 3.11 et refusent
en dessous, la CI passe en 3.13, `PYTHONUTF8=1` posé par les installeurs (trustmark) ;
marqueur d'installation = `diffusers.modular_pipelines.Flux3DistilledAutoBlocks` ;
Pillow 13.0.0 installé à part (gradio 5.50 déclare <12), `rembg>=2.0.82`.

**Cette machine EST la RTX 5090** (32 Go, driver 610.60) et la session HF locale a accès
aux deux repos gated FLUX.3 : le e2e GPU (lot E) et les rendus du README (lot F) peuvent
tourner ici, avec l'accord de Mike (environ 20 Go à télécharger).

Dependabot après le lot D (62 alertes sur `main` au 27/10, toutes sur
requirements-lock.txt) : Pillow (< 13.0.0), transformers (< 5.10), accelerate (<= 1.14)
et rembg (< 2.0.82) sont closes par le nouveau lock une fois fusionné dans `main`.
Restent ouvertes et assumées : **gradio** (correctifs en 6.6 à 6.16, la borne <6 est
volontaire, bug Brotli/h11 de gradio 6, à réévaluer en v0.2) et **torch** (< 2.13 ; 2.11
est la dernière build de l'index cu128, une roue CPU de PyPI n'est pas une option).

Fait au lot C : `FLUX3_DISTILLED` / `FLUX3_BASE` dans cz_ui (depuis cz_flux3), radio
"FLUX.3 checkpoint" Distilled / Base, radio "Precision" bf16 / fp8r / nvfp4 branchée sur
`cz_pipeline.set_precision()` (persistée dans `flux3_precision`, swap du transformer seul
au prochain run), `BASE_REPO_NOTES` par repo (licence Kommunity, gated, VRAM), aide de
l'onglet Edit avec `<ref_image_N>`, `cz_protocol.MAX_REFS` = 10 (l'onglet garde 4 slots),
Sampler / Schedule et Text encoder grisés avec leur explication, textes "Klein" / "FLUX.2"
remplacés, le bouton CivitAI n'applique plus le sampler. Test : `set_precision` et le
sélecteur dans `tests/test_flux3_profiles.py` (5). Repo nettoyé (commit cabef91) : notes de
travail, benches klein, `tests/test_klein_e2e.py` et `assets/` restent en local.

Fait au lot B : profils `flux-3-image-klein-base` (50 / 3,5) avant `flux-3-image-klein`
(16 / 1,0) dans `cz_core` et `config-sample.txt`, garde contre la clé `klein` d'un
config.txt hérité (sautée sur un nom FLUX.3, dit une fois), presets Distilled / Fast /
Base, clés `flux3_model` / `flux3_transformer` / `flux3_precision` / `flux3_base_repos`
(cz_ui lit `flux3_base_repos` puis `klein_base_repos`), `preferences.example.json`,
catalogue d'edit LoRAs vidé avec un commentaire daté, `cz_protocol.OMNI_DEFAULT` sur
FLUX.3. Test : `tests/test_flux3_profiles.py` (4).

Fait au lot A (voir §6 pour le détail) : `get_pipe` rend la base pour img2img/inpaint,
sampler Euler imposé et dit une fois, `Flux3Pipe.register_modules`, GGUF via
`GGUFQuantizationConfig`, code FLUX.2 inatteignable supprimé de `_load_transformer`,
`_trim_text_encoder` annonce qu'il ne fait rien sur Qwen3-VL, messages de refus par famille
(un fichier FLUX.2 = "runs in crispz-klein"), `model_info['variant']` = la famille
(`FLUX.3` / `FLUX.2`). Tests : `tests/test_flux3_adapter.py` (16) et
`tests/test_flux3_wiring.py` (12), verts, composants miniatures dans `tests/flux3_tiny.py`.

Fait avant (commit 8bd62a6) :
- `main` (f558fb7) : squelette, renommage crispz-klein vers crispz-flux3, FORK.md.
- `cz_flux3.py` : moteur complet ; `tests/test_flux3_adapter.py` : 12/12 sur CPU
  (composants aléatoires miniatures, aucun téléchargement).
- `cz_pipeline.py`, partiel : repo par défaut FLUX.3, clés `flux3_model`,
  `flux3_transformer`, `flux3_precision` (bf16 / fp8r / nvfp4), chargement par
  `cz_flux3.load()` dans `_ensure_base()` (clé de cache incluant `PRECISION`), guidance et
  négatif réels sur la variante base (`_cfg`, `_qwen_call`), estimation VRAM
  (`_base_vram_need_gb`), `_load_transformer` redirigé vers cz_flux3, listing et refus
  des checkpoints par famille (`_flux3_file_refusal`, `list_checkpoints`,
  `checkpoint_refusal`), refus des LoRAs FLUX.2 (`_lora_unsupported`).

Suite de tests mesurée après le lot C : **489 passent, 47 échouent** (B : 495 / 40 ;
A : 498 / 33 ; avant : 491 / 24). Les 7 échecs ajoutés par le lot C : `test_civitai_search`
(4, libellés CivitAI "Flux.2 Klein") et `test_preset_model_switch` (3, les constantes
`KLEIN_BASE_4B/9B` supprimées). Les 7 échecs ajoutés par le lot B attendent tous 4 steps ou
le catalogue Consistence-Edit de FLUX.2 : `test_detailer_steps` (2),
`test_edit_lora_presets` (3), `test_undistilled_guidance::..._gets_the_right_profile`,
`test_preset_model_switch::test_the_refusal_leads_with_the_action`. Les 33 échecs testent tous le comportement FLUX.2 (4B/9B par largeur, dequant
FP8 ComfyUI, encodeur trim 9B, libellés de menu "4B"/"9B") ; les 9 nouveaux viennent de la
garde par famille et des nouveaux messages du lot A, et sont attendus :

```
test_edit_lora_presets::test_variant_guard_refuses_the_other_model
test_encoder_trim::test_the_budget_follows_the_deed_not_the_intent
test_encoder_trim::test_a_trimmed_9B_still_gets_the_offload_it_needs
test_klein_variant::(4 tests : mismatch_both_ways, summary_once, unknown_base,
                     4B_lora_on_9B)                               [3 nouveaux]
test_lokr_merge::test_a_lokr_is_kept_out_of_the_peft_set
test_lora_dialect::(6 tests)                                      [1 nouveau]
test_lora_offload::(2 tests)
test_model_labels::(3 tests : le menu dit FLUX.3/FLUX.2, plus 4B/9B) [3 nouveaux]
test_precache_variants::(2 tests)
test_preset_model_switch::(7 tests)
test_quant_formats::(5 tests : prefixed_single_file, mislabelled_gguf,
                     gguf_arch_and_layout, gguf_layout_beats_mislabelled,
                     gguf_qwen_diffusers_layout)                  [3 nouveaux]
```
(`test_lora_extra_dirs::test_the_setter...`, dans la liste d'avant, passe maintenant.)

## 6. Actions restantes, par lot

### Lot A : finir le branchement de cz_pipeline.py (FAIT le 28/10/2026)

Livré tel que décrit ci-dessous, plus : le `.gitignore` corrigé (§3), `_hf_token()` passé
au chargeur du transformer, un fichier ComfyUI "scaled" FP8 est transmis tel quel à
diffusers avec une ligne de log (le dequant de crispz-klein n'est pas porté : FLUX.3 a
ses propres fp8r / nvfp4). Point 6 : vérifié, Qwen3VLForConditionalGeneration range ses
blocs sous `model.language_model.layers`, la fonction sort en l'annonçant.

1. `get_pipe(kind)` : si la base est un `cz_flux3.Flux3Pipe`, `img2img` et `inpaint`
   renvoient la base elle-même (`_DERIVED[kind] = base`), sans `Flux2KleinInpaintPipeline`
   ni `from_pipe`. À insérer juste après le bloc `if kind in _DERIVED`.
2. `_apply_sampler(pipe)` : sur un `Flux3Pipe`, ne rien changer (le distillé est une
   flow map : chaque pas transmet au transformer le timestep suivant ; un sampler
   multistep ou un planning remappé le casse). Logguer une seule fois si l'utilisateur a
   choisi autre chose qu'`euler/sgm_uniform` (set `_SAMPLER_IGNORED_SAID`).
3. `cz_flux3.Flux3Pipe.register_modules(**kw)` : à ajouter (utilisé par
   `_swap_transformer`), délègue à `setattr` donc à `update_components`.
4. `cz_flux3.load_transformer` : `.gguf` en single-file avec
   `GGUFQuantizationConfig(compute_dtype=dtype)` (miroir `unsloth/flux-3-image-klein-GGUF`).
5. Nettoyer `_load_transformer` : le code FLUX.2 inatteignable après le `return` doit
   être supprimé ou clairement isolé (le `# noqa: E501` actuel est faux).
6. `_trim_text_encoder` : vérifier qu'il ne touche à rien sur Qwen3-VL (attribut
   `enc.model.layers` absent, donc sortie anticipée). Option v0.2 : tronquer à 33 couches
   sur 36 (couche max lue = 32).
7. Variantes de messages : `_variant_name`, `_variant_fix`, `_variant_skip_summary`,
   `_variant_refusal`, `_FLUX2_VARIANTS` ne doivent plus produire de texte "4B/9B" pour
   un fichier FLUX.3. Tout fichier FLUX.2 détecté = "runs in crispz-klein".
8. Docstrings et logs restants "FLUX.2 Klein" dans les chemins actifs : à mettre à jour
   (pas l'historique en commentaire).

Acceptation : `tests/test_flux3_adapter.py` vert ; un nouveau test
`tests/test_flux3_wiring.py` qui, avec un `Flux3Pipe` miniature injecté dans
`cz_pipeline._BASE_PIPE`, fait passer `generate`, `generate_omni`, `process_one`
(refine), `inpaint_run` et `outpaint` sans exception et avec la bonne taille de sortie.

### Lot B : profils, presets et configuration (FAIT le 28/10/2026)

1. **Piège** : `cz_core.profile_for_model()` matche par sous-chaîne et le profil `klein`
   (4 steps) matcherait `flux-3-image-klein`. Remplacer dans `config-sample.txt` et les
   défauts de `cz_core` par des profils `flux-3-image-klein-base` (50 steps, 3,5) placé
   AVANT `flux-3-image-klein` (16 steps, 1,0) : premier match gagnant.
2. `performance_presets` : Distilled (16 / 1,0), Fast (8 / 1,0, à mesurer), Base (50 / 3,5).
3. `config-sample.txt` : remplacer toutes les clés et aides `klein_*` (lignes ~50 à 263)
   par `flux3_model`, `flux3_transformer`, `flux3_precision`, `flux3_base_repos`, avec des
   aides qui parlent de FLUX.3 (licence Kommunity, gated, `hf_token`, VRAM mesurée).
4. `preferences.example.json` : `flux3_model`.
5. `cz_edit_loras.py` : le catalogue de presets LoRA d'édition est FLUX.2 ; le vider pour
   FLUX.3 (aucun LoRA d'édition publié à ce jour) avec un commentaire daté.

Acceptation : `profile_for_model("black-forest-labs/flux-3-image-klein")` = (16, 1.0) ;
`...-base` = (50, 3.5) ; test dédié.

### Lot C : interface (cz_ui.py) (FAIT le 28/10/2026 ; l'onglet Edit garde 4 slots, le protocole prend 10 refs)

1. `KLEIN_BASE_4B/9B` deviennent `FLUX3_DISTILLED/FLUX3_BASE` (depuis cz_flux3) ;
   `ZIMAGE_BASE_REPOS` lit `flux3_base_repos`.
2. Le Radio "Klein family" `["4B", "9B"]` devient "FLUX.3 checkpoint"
   `["Distilled", "Base"]` ; `_variant_of_repo` et `_apply_variant` suivent.
3. Ajouter un Radio "Precision" `["bf16", "fp8r", "nvfp4"]` à côté, branché sur un
   nouveau `cz_pipeline.set_precision()` (persisté dans les prefs, rechargement du seul
   transformer via la clé de cache). nvfp4 : afficher qu'il faut Blackwell + flashinfer.
4. `BASE_REPO_NOTES` : une note par repo (licence Kommunity : gratuit en commercial sous
   5 M$ de CA ; gated ; VRAM bf16 environ 19 Go, fp8r 14,4, nvfp4 13,5).
5. Onglet Edit : indiquer dans l'aide du prompt que les références se citent
   `<ref_image_0>`, `<ref_image_1>`... (jusqu'à 10) ; relever la limite de références
   de 4 à 10 si elle est codée en dur.
6. Masquer ou griser les menus Sampler/Schedule (Euler imposé) et le choix d'encodeur
   texte, avec une phrase d'explication.
7. Textes visibles "Klein checkpoint", "FLUX.2" : remplacer.

Acceptation : `tools/smoke_test.py` vert (22 checks, `build_ui()` compris) ;
`tests/test_ui_binding_returns.py` vert.

### Lot D : dépendances et installation (FAIT le 28/10/2026, Python 3.11+ requis)

1. `requirements.txt` : `diffusers==0.42.1` (PyPI, plus de git+commit),
   `transformers>=5,<6`, `huggingface-hub>=1.32`, reste inchangé. Vérifier que gradio 5.50
   s'installe avec hub 1.x (OK en local le 27/10).
2. `requirements-lock.txt` : régénérer depuis un venv propre (torch cu128 côté Windows,
   torchvision assorti).
3. `install.bat` / `install.sh` : marqueur d'installation (aujourd'hui l'inpaint
   Flux2) : tester l'import de `diffusers.modular_pipelines.Flux3DistilledAutoBlocks`.
4. Vérifier les alertes Dependabot après la montée de versions (62 sur main au 27/10,
   héritées de crispz-klein) et lister celles qui restent.

Acceptation : installation propre dans un venv vierge ; `python -c "import app"` OK.

### Lot E : tests (FAIT le 28/10/2026, CPU et e2e GPU sur la 5090)

1. Les 24 tests FLUX.2 listés au §5 : pour chacun, soit le réécrire pour FLUX.3 (même
   intention, ex. "un LoRA de l'autre famille est refusé en une phrase"), soit le
   supprimer avec une ligne dans le CHANGELOG qui dit pourquoi. Ne pas les désactiver
   en silence.
2. Nouveaux tests : famille des checkpoints et LoRAs sur des en-têtes safetensors
   synthétiques (FLUX.2 refusé, FLUX.3 accepté, LoRA dans le dossier checkpoints
   refusé), `set_precision`, profils (lot B).
3. `tests/test_klein_e2e.py` (GPU) : le transformer en `tests/test_flux3_e2e.py`
   (mêmes 6 étapes : chargement, gen, edit 1 ref, edit 2 refs, inpaint, upscale) pour
   que Mike le lance sur la 5090 ; skip propre sans CUDA ou sans token HF.

Acceptation : suite complète verte sur CPU, 0 échec, 0 test ignoré sans raison écrite.

### Lot F : documentation et livraison (FAIT le 28/10/2026, launcher Pinokio compris)

Demande de Mike (28/10) : dès qu'une version est testable et utilisable, faire un vrai
rendu d'image (sur la 5090, le e2e) pour remettre des captures dans le README, et
documenter la nouvelle version en SUPPRIMANT du README, du CHANGELOG courant et de
config-sample ce qui venait de crispz-klein et n'est plus d'actualité (4B/9B, dequant
ComfyUI, encodeur de remplacement, trim de l'encodeur, Consistence-Edit, 4 steps...).
L'historique reste dans FORK-klein.md.

1. README : retirer le bandeau WIP, décrire FLUX.3 (16 steps, références `<ref_image_N>`,
   précisions, licence). Les captures d'écran et mesures viendront du e2e de Mike ;
   `assets/` est en local seulement (voir .gitignore), à re-versionner avec les captures
   FLUX.3.
2. FORK.md : mettre à jour la feuille de route (étapes faites) et les décisions (§4).
3. CHANGELOG : entrée 0.1.0.
4. Merge `wip/flux3-engine` dans `main` (fast-forward ou merge commit, au nom de Mike),
   push, puis Mike teste sur la 5090 avec `tests/test_flux3_e2e.py`.
5. Ensuite seulement : launcher Pinokio `crispz-flux3.pinokio` (Mike crée le repo vide,
   on part de `crispz-klein.pinokio`).

## 7. Validation GPU (Mike, RTX 5090)

À mesurer et reporter dans FORK.md : temps de chargement, VRAM au repos et en pic,
temps par image en bf16 / fp8r / nvfp4 à 1024x1024, 16 steps ; edit 1 et 2 références ;
inpaint ; upscale. Vérifier visuellement : l'aperçu live (la matrice latent vers RGB de
klein est réutilisée car le VAE est le même AutoencoderKLFlux2, à confirmer), le bouton
Stop, l'inpaint sans couture visible.

## 8. Après la v0.1.0 (pour mémoire)

- Prompt upsampler : soit le pup 9B chargé puis déchargé avant la génération, soit le
  `system_prompts.json` de BFL envoyé à Ollama via `cz_ollama.py` (plus léger).
- Bounding boxes (grille 0-1000, `[y_min, x_min, y_max, x_max]`) dans l'onglet Edit.
- Troncature de l'encodeur Qwen3-VL (33 couches sur 36).
- Variante Base : vérifier la qualité du CFG et du négatif, régler les presets.
