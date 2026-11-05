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

## Origine des tests hérités

Le 05/11/2026, les docstrings de `tests/` ont été ramenées à trois lignes (règle 4 bis
du CLAUDE.md) : elles disent ce que chaque fichier verrouille ici. L'histoire des bugs qui
les ont fait écrire, dans les studios précédents (crispz-klein, crispz-krea2, crispz-flux3,
crispz-qwen-edit, Fooocus2026), est gardée ici, une ligne par fichier. Les fichiers nés
dans crispz-ideogram n'ont pas d'entrée.

- `flux3_tiny.py` : fixture of crispz-flux3, shared by the FLUX.3 merge-path tests.
- `test_ab_index.py` : every opening re-read the PNG tags of ALL the images (295 s for
  9278 images, the SPA polling timed out: 'there are images missing').
- `test_asset_browser_dirs.py` : the Models tab came out EMPTY on a normal install (only
  the main folder, only .safetensors; a library of 22 models kept on another disk showed
  0).
- `test_checkpoints_subfolders.py` : tidying a library into 4b/ and 9b/ made every model
  vanish (the listing was one os.listdir) and broke the presets naming them.
- `test_civitai_extra_dirs.py` : the button joined the relative path to the main folder
  alone: any LoRA kept outside the app folder answered 'model file not found'.
- `test_civitai_search.py` : ported from crispz-studio; crispz-klein replaced the equality
  match on the CivitAI base label ('Z-Image') by a normalised prefix match ('Flux.2 Klein
  4B-base'), the loaded variant first.
- `test_describe_styles.py` : measured 2026-09-11 (Agents-A1-4B, muse-glimmer,
  descriptions regenerated by klein): without the medium up front a drawing came back as a
  photo, quoted text came back shuffled, a small model wrote 'No text is visible.'.
- `test_detailer_steps.py` : a detailer pass ran at 12 steps on klein, a model distilled
  for 4: triple cost per hand and face for an MAE of 0.7 (RTX 5090: 4B 2.0 -> 0.9 s/hand,
  9B GGUF 12.2 -> 6.4 s/hand).
- `test_edit_lora_presets.py` -> `test_flux3_merge_edit_loras.py` : the catalogue ships
  one preset, the CivitAI 'FLUX 3 Image Klein Outpaint LoRA' (2026-10-28); crispz-klein's
  two Consistence-Edit files are FLUX.2.
- `test_encoder_trim.py` : FLUX.2 reads three intermediate layers of Qwen3-8B; dropping
  the eight blocks after the last one is exact (15.3 -> 11.2 GB, encoding 5.0 -> 3.2 s).
  Two traps: diffusers names the method _get_qwen3_prompt_embeds (the first trim never
  fired, and said so in a log nobody read), and the budget must follow what was really
  trimmed.
- `test_flux3_adapter.py` -> `test_flux3_merge_adapter.py` : crispz-flux3's adapter tests,
  unchanged.
- `test_flux3_wiring.py` -> `test_flux3_merge_wiring.py` : crispz-flux3's wiring tests;
  the FLUX.3 policies (guidance by variant, single files through cz_flux3) were removed on
  2026-11-01.
- `test_gen_meta.py` : three holes: the edit LoRAs were absent, the list was the LoRAs
  asked for rather than applied (a discarded variant or LyCORIS signed an image it did not
  touch), the base repo vanished as soon as a single file was chosen.
- `test_hands_available.py` : the hands detailer was declared absent when only ultralytics
  was missing, although it only serves the one-time export and must not live in the
  diffusion process.
- `test_klein_guidance.py` : a GPU probe, not a test (no test function, exit code always
  0): rendered the same seed at guidance 1.0 / 4.0 / 8.0 on klein-4B to settle whether a
  distilled model ignores guidance_scale (it does, bit for bit). Removed on 2026-11-05:
  Ideogram 4 is not distilled and runs a guidance schedule.
- `test_klein_variant.py` -> `test_checkpoint_family_guard.py` : crispz-klein told its 4B
  and 9B apart by hidden width (a 9B in a 4B pipeline blew up after reading gigabytes:
  'expected shape [18432, 3072], but got [24576, 4096]'); since crispz-flux3 the family is
  read from the tensor names.
- `test_latent_preview.py` : getting the packed grid wrong does not raise, it shows a
  scrambled image or the wrong aspect ratio.
- `test_lokr_merge.py` : on FLUX.2 the qkv is fused on the checkpoint side and separate in
  diffusers, a Kronecker product cannot be cut in three (SNOFS: w1 [4,4], w2 [3072,1024]);
  LyCORIS applies no scalar on full factors while peft computes alpha/r, ai-toolkit writes
  alpha = lora_dim (1e10 on SNOFS) so both give 1.0.
- `test_lora_dialect.py` : lrzjason/Consistance_Edit_Lora mixed 160 PEFT keys with 40
  down/up keys and peft loaded it partially without an error; on klein the edit pipe is
  the base pipe, so the two LoRA sets fought over cz_lora_i and set_adapters silently
  disabled the base ones.
- `test_lora_hotswap.py` : activating a LoRA used to free_vram() and reload the whole
  pipeline (~9 s).
- `test_lora_offload.py` : crispz-klein 1.36.6: two DoRA edit LoRAs failed on 'Cannot copy
  out of meta tensor', then every render on 'Cannot generate a cpu tensor from a generator
  of type cuda'.
- `test_model_labels.py` : a library of 27 checkpoints and 105 LoRAs showed bare file
  names, and the one thing that decided whether a file loaded (4B or 9B) was invisible
  until it failed.
- `test_model_profiles.py` : the default inherited from FLUX.3 was 16 steps at guidance
  1.0: on Ideogram a schedule peaking at 1.0, almost no CFG.
- `test_model_swap.py` : switching checkpoint A -> B on the same base repo used to
  free_vram() and reload the whole pipeline.
- `test_mxfp8_scale.py` : snofs14Flux2Klein9b_14Distilled (mxfp8, group 32): the [out, nb]
  scale first crashed the broadcast, then read as a linear multiplier gave weights 470000x
  too large (std 17093 instead of 0.024), a smoothed image and no error.
- `test_ollama_thinking.py` : Describe / Improve / Vision Mix returned 'Okay, the user
  wants a prompt for...' stuck in front of the real prompt, which went as is into the text
  encoder.
- `test_pipe_lock.py` : Gradio does not serialise the events of different listeners: Free
  VRAM under a running denoise took the model out from under it.
- `test_prescaled_fp8.py` : kleinFinalcutFP16FP8_comfyQuant (2026-09-10) rendered coloured
  noise: the file stores scaled FP8 weights and supplies weight_scale = amax / 448 anyway,
  the loader multiplied (weights 1,200 to 1,700 times too small); the criterion
  max|stored| / (scale x range) is 1.03 for it, 71 to 1,691 for the 17 other quantised
  files of the library.
- `test_preset_model_switch.py` : crispz-klein 1.19: a preset naming a discarded
  checkpoint pushed it into the dropdown, Gradio refused the value and re-read the old
  one, 'I change model, it is still the default model'.
- `test_prompt_embed_cache.py` : in 'model' offload the encoder transfer is paid on every
  call: a hand pass at 4 steps on klein-9B GGUF cost 5.8 s of prompt and setup for 0.3 s
  of diffusion.
- `test_protocol_size.py` : inherited from crispz-qwen-edit: size_explicit was only
  computed in the edit branch, a gen with refs kept the reference's dimensions (an
  800x1312 panel came back in landscape and got cropped at layout time).
- `test_real_cfg.py` -> `test_flux3_merge_cfg.py` : the pipeline decides
  do_classifier_free_guidance from config.is_distilled of the BASE repo; with an
  undistilled single file the app passed the guidance and diffusers ignored it ('Guidance
  scale 3.5 is ignored for step-wise distilled models'); seen on the 2026-09-10 bench,
  kleinForeskin at 28 steps cost 0.6 s/step like a distilled one.
- `test_sampler_race.py` : crispz-krea2, 2026-09-28: checkpoint switched, then 'Apply
  CivitAI recommended settings', then Generate; the running denoise stepped a fresh
  scheduler and raised IndexError in _init_step_index halfway through the render.
- `test_text_encoder.py` : FLUX.2 reads three intermediate hidden states in a
  context_embedder 3 x hidden wide: a Qwen3-4B 'abliterated' suits the 4B, on the 9B (4096
  wide) the refusal must come before reading 8 GB; id(enc) alone is not a cache key,
  CPython recycles ids.
- `test_ui_binding_returns.py` : two .then(set_sampler, [sampler_dd], None) chains warned
  on every 'Apply CivitAI recommended settings' click in three forks; the regex version
  missed four forms Gradio treats like outputs=None.
- `test_undistilled_guidance.py` -> `test_flux3_merge_guidance.py` : klein-4B/9B are
  distilled (guidance inert, measured bit for bit), so _qwen_call forced 1.0; an
  'undistilled' community checkpoint ('use with Turbo Lora') needs a real CFG and rendered
  a blurry mush without a word.
- `test_vae_guard.py` : 2026-09-10: diffusion_pytorch_model.safetensors (160 MB, 251 VAE
  tensors) passed every guard, appeared in the menu and crashed on 'Cannot copy out of
  meta tensor'; the bundles carry the same 251 VAE keys with the transformer keys prefixed
  model.diffusion_model.double_blocks.
- `test_vram_retry.py` : crispz-klein 1.36.4: an Edit batch with Upscale and the detailer
  ran out of VRAM at the fourth image, then every later render failed until a restart.
- `test_xyz_edit_lora.py` -> `test_flux3_merge_xyz_edit_lora.py` : an axis varying the
  edit weight on a txt2img grid rendered N identical images with no error; csv.reader
  refused a newline in an unquoted field and the error left as a Gradio trace (the parser
  tests moved to test_xyz.py on 2026-11-05).
