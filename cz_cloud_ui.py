"""crispz-ideogram - the "Cloud (Ideogram)" tab (lot 4).

One tab for the Ideogram cloud functions (cz_ideogram_api): pick the function, give the
image(s) and the instruction, then "Estimate price" (dry run, nothing billed) or "Run".
Every button says Cloud, the banner says where the data goes, and both buttons stay
greyed out with the reason while the switches are off (`ideogram_cloud_enabled`, the
API key). The switches can be set here; they are saved in preferences.json.

The handlers are plain functions (handle, settings_save, op_help) so the tests drive them
without a browser.
"""

import os

from PIL import Image, ImageChops

import cz_ideogram_api as api
from cz_core import _log

OP_ORDER = ("edit", "replace_background", "remix", "remove_background", "upscale",
            "transparent", "vectorize", "svg")

# What each function takes and sends, shown under the selector.
OP_HELP = {
    "edit": ("**Precise edit (Ideogram 4.5)**: an instruction in plain words (or a JSON "
             "caption). Paint on the image to limit the edit to that area (optional). Up to "
             "4 reference images (3 with a painted area). The result keeps the image's exact "
             "size; above 2048x2048 in area, paint an area or it comes back scaled down."),
    "replace_background": ("**Replace background (Ideogram 4.5)**: Reference 1 = a photo of "
                           "the setting to match. The people of the source are kept. Sent "
                           "private."),
    "remix": ("**Remix (Ideogram 4.0)**: the image transformed by the prompt. Image weight "
              "0 = let Ideogram choose. Ideogram keeps the source in your account."),
    "remove_background": "**Remove background**: a transparent PNG. Sent private. "
                         "(A local Remove BG exists in its own tab: nothing leaves.)",
    "upscale": ("**Upscale** x2 / x4 / x8, up to 8192 px a side; Topaz models run at "
                "Ideogram. 'topaz-text-refine' suits text and posters. (Local ESRGAN upscale "
                "exists in Upscale or img2img.)"),
    "transparent": ("**Transparent (Ideogram 4.0)**: a generation on a real alpha "
                    "background. The prompt can be a JSON caption (used as is). The aspect "
                    "follows Width / Height of the Generate tab."),
    "vectorize": "**Vectorize**: the image as an SVG (logos, flat colours). Sent private.",
    "svg": ("**Generate SVG**: a vector image from the prompt; the source image and the "
            "references, when given, guide it. Sent private."),
}


def editor_image_mask(editor_value):
    """(image, mask) from a gr.ImageEditor: mask = what was painted (white), or None when
    nothing was."""
    if not editor_value:
        return None, None
    if isinstance(editor_value, Image.Image):
        return editor_value, None
    bg = editor_value.get("background")
    if bg is None:
        return None, None
    bg = bg.convert("RGB")
    mask = Image.new("L", bg.size, 0)
    comp = editor_value.get("composite")
    if comp is not None:
        diff = ImageChops.difference(comp.convert("RGB"), bg).convert("L")
        mask = diff.point(lambda p: 255 if p > 8 else 0)
    for ly in (editor_value.get("layers") or []):
        if ly is not None:
            a = ly.convert("RGBA").split()[-1]
            mask = ImageChops.lighter(mask, a.point(lambda p: 255 if p > 8 else 0))
    return bg, (mask if mask.getbbox() else None)


def banner():
    ok, why = api.status()
    where = (f"**Cloud.** Everything in this tab is sent to {api.WHERE}: the image(s) and "
             f"the prompt leave this machine. Every call is logged (`CLOUD:` lines) and "
             f"written to the cost journal (`{api.JOURNAL_NAME}` in the output folder). "
             f"Images made through the API follow Ideogram's API terms, not the "
             f"non-commercial licence of the local weights.")
    return where + ("\n\n**On.**" if ok else f"\n\n**Off:** {why}")


def op_help(op):
    return OP_HELP.get(op, "")


def _int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def build_request(op, editor_value, refs, prompt, main_prompt, opts, width=1024,
                  height=1024):
    """The UI values -> an api.Request (nothing is sent here)."""
    image, mask = editor_image_mask(editor_value)
    refs = [r for r in (refs or []) if r is not None]
    text = (prompt or "").strip() or (main_prompt or "").strip()
    seed = _int(opts.get("seed"), -1)
    seed = None if seed is None or seed < 0 else seed
    n = max(1, _int(opts.get("num_images"), 1) or 1)
    needs_image = op not in ("transparent", "svg")
    if needs_image and image is None:
        raise api.CloudError("drop the source image first")
    if op == "edit":
        return api.build_edit(image, text, mask=mask, refs=refs,
                              quality=opts.get("quality", "medium"), seed=seed, num_images=n)
    if mask is not None:
        _log(f"cloud {op}: the painted area is used by Precise edit only, ignored here")
    if op == "replace_background":
        if not refs:
            raise api.CloudError("Replace background needs Reference 1: a photo of the "
                                 "setting to match")
        return api.build_replace_background(image, refs[0], num_images=min(n, 4))
    if op == "remix":
        return api.build_remix(image, text, image_weight=_int(opts.get("image_weight"), 0),
                               resolution=opts.get("remix_resolution") or "",
                               rendering_speed=opts.get("rendering_speed", "default"),
                               num_images=n)
    if op == "remove_background":
        return api.build_remove_background(image)
    if op == "upscale":
        return api.build_upscale(image, factor=opts.get("factor", "x2"),
                                 model=opts.get("upscale_model", "auto"), seed=seed)
    if op == "transparent":
        return api.build_transparent(text, width, height,
                                     magic_prompt=opts.get("magic_prompt", "auto"),
                                     output_resolution=opts.get("output_resolution", "1k"),
                                     rendering_speed=opts.get("rendering_speed", "default"),
                                     seed=seed, num_images=n)
    if op == "vectorize":
        return api.build_vectorize(image, mode=opts.get("svg_mode", "auto"))
    if op == "svg":
        return api.build_svg(text, images=([image] if image is not None else []) + refs,
                             aspect_ratio=opts.get("svg_aspect", "auto"),
                             quality=opts.get("svg_quality", "medium"), seed=seed,
                             num_images=n)
    raise api.CloudError(f"unknown cloud function {op!r}")


def handle(op, dry_run, editor_value, refs, prompt, main_prompt, opts, width, height,
           save_mode, output_dir, output_format, history):
    """-> (gallery_images, status_markdown, new_history)."""
    history = list(history or [])
    try:
        req = build_request(op, editor_value, refs, prompt, main_prompt, opts, width, height)
        if dry_run:
            _q, line = api.quote(req, output_dir=output_dir)
            return [], f"{api.OPS[op].label}: {line}. Nothing was generated or billed.", history
        res = api.run(req, output_dir=output_dir)
    except api.CloudError as e:
        return [], f"Cloud: {e}", history
    except Exception as e:
        _log(f"cloud {op} failed: {type(e).__name__}: {e}")
        return [], f"Cloud: {type(e).__name__}: {e}", history
    paths = []
    try:
        paths = api.save(req, res, save_mode, output_dir, output_format)
    except Exception as e:
        _log(f"cloud {op}: results not saved ({type(e).__name__}: {e})")
    shown = list(res.images)
    no_preview = 0
    for svg in res.svgs:
        pv = api.svg_preview(svg)
        if pv is not None:
            shown.append(pv)
        else:
            no_preview += 1
    lines = [f"{api.OPS[op].label}: {len(res.images)} image(s)"
             + (f", {len(res.svgs)} SVG(s)" if res.svgs else "")
             + (f", billed {res.cost_usd:.4f} USD" if res.cost_usd is not None else "")
             + ". Call logged."]
    if res.blocked:
        lines.append(f"{res.blocked} result(s) refused by Ideogram's safety checks.")
    if no_preview:
        lines.append("No SVG preview here (install `cairosvg` for one); the .svg file is "
                     "saved.")
    if paths:
        lines.append("Saved: " + ", ".join(f"`{os.path.basename(p)}`" for p in paths))
    elif save_mode == "display":
        lines.append("Display only: nothing saved (Save tab).")
    new_hist = (shown + history)[:200]
    return shown, " ".join(lines), new_hist


def settings_save(enabled, key):
    """The switches of this tab -> preferences.json and the running app. The key is never
    shown back. -> (key_box_update_value, status_markdown, usable)."""
    from cz_core import _prefs, _save_prefs_keys
    upd = {"ideogram_cloud_enabled": bool(enabled)}
    key = (key or "").strip()
    if key:
        upd["ideogram_api_key"] = key
    try:
        _save_prefs_keys(upd)
    except Exception as e:
        _log(f"cloud settings not persisted: {e}")
    _prefs.update(upd)
    ok, why = api.status()
    _log(f"cloud functions {'ENABLED' if enabled else 'disabled'}"
         + (", API key updated" if key else ""))
    return "", (("Cloud functions **on**. " if ok else f"**Off:** {why} ")
                + ("Key saved (preferences.json, gitignored)." if key else "")), ok


# --- Gradio --------------------------------------------------------------------------------

def build_tab():
    """Builds the tab's components (call inside a gr.Tab). -> dict of components."""
    import gradio as gr
    import cz_magic
    ok, why = api.status()
    p = {}
    p["banner"] = gr.Markdown(banner())
    with gr.Accordion("Cloud settings (off by default)", open=not ok):
        with gr.Row():
            p["enabled"] = gr.Checkbox(value=cz_magic.cloud_enabled(),
                                       label="Enable the Ideogram cloud functions",
                                       info="Sends images and prompts to the United States "
                                            "when a Cloud button is pressed.")
            p["key"] = gr.Textbox(label="Ideogram API key", type="password",
                                  placeholder=("a key is set" if cz_magic.api_key()
                                               else "ideogram.ai > API"))
            p["save"] = gr.Button("Save", size="sm")
        p["settings_status"] = gr.Markdown("")
    choices = [(api.OPS[k].label, k) for k in OP_ORDER]
    p["op"] = gr.Radio(choices, value="edit", label="Cloud function")
    p["help"] = gr.Markdown(op_help("edit"))
    p["src"] = gr.ImageEditor(type="pil", label="Source image (paint = area to edit, "
                                                 "Precise edit only)",
                              brush=gr.Brush(default_size=40, colors=["#ffffff"],
                                             default_color="#ffffff"),
                              layers=False, transforms=[], height=360)
    with gr.Accordion("Reference images", open=False):
        with gr.Row():
            p["refs"] = [gr.Image(type="pil", label=f"Reference {i}", height=140)
                         for i in range(1, 5)]
    p["prompt"] = gr.Textbox(label="Instruction / prompt (empty = the main prompt)", lines=2)
    with gr.Row():
        p["num_images"] = gr.Slider(1, 4, value=1, step=1, label="Images")
        p["seed"] = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
        p["quality"] = gr.Dropdown(list(api.EDIT_QUALITIES), value="medium",
                                   label="Edit quality (sets the price)")
    with gr.Accordion("Options of the other functions", open=False):
        with gr.Row():
            p["image_weight"] = gr.Slider(0, 100, value=0, step=1,
                                          label="Remix image weight (0 = auto)")
            p["remix_resolution"] = gr.Dropdown([""] + list(api.REMIX_RESOLUTIONS), value="",
                                                label="Remix resolution (empty = keep shape)")
            p["rendering_speed"] = gr.Dropdown(list(api.RENDERING_SPEEDS), value="default",
                                               label="Rendering speed (remix, transparent)")
        with gr.Row():
            p["upscale_model"] = gr.Dropdown(list(api.UPSCALE_MODELS), value="auto",
                                             label="Upscale model")
            p["factor"] = gr.Dropdown(list(api.UPSCALE_FACTORS), value="x2",
                                      label="Upscale factor")
            p["magic_prompt"] = gr.Dropdown(list(api.MAGIC_PROMPT), value="auto",
                                            label="Transparent: magic prompt")
            p["output_resolution"] = gr.Dropdown(list(api.OUTPUT_RESOLUTIONS), value="1k",
                                                 label="Transparent: output resolution")
        with gr.Row():
            p["svg_mode"] = gr.Dropdown(list(api.SVG_MODES), value="auto",
                                        label="Vectorize detail")
            p["svg_aspect"] = gr.Dropdown(["auto"] + list(api.ASPECT_RATIOS), value="auto",
                                          label="SVG aspect")
            p["svg_quality"] = gr.Dropdown(list(api.SVG_QUALITIES), value="medium",
                                           label="SVG quality")
    with gr.Row():
        p["quote_btn"] = gr.Button("Cloud: Estimate price (dry run)", interactive=ok)
        p["run_btn"] = gr.Button("Cloud: Run", variant="primary", interactive=ok)
    p["status"] = gr.Markdown("" if ok else f"*Buttons greyed out: {why}*")
    return p


OPT_KEYS = ("num_images", "seed", "quality", "image_weight", "remix_resolution",
            "rendering_speed", "upscale_model", "factor", "magic_prompt",
            "output_resolution", "svg_mode", "svg_aspect", "svg_quality")


def wire(p, main_prompt, width, height, save_mode, output_dir, output_format, history,
         out, report, history_gallery, extra_buttons=(), demo=None):
    """Connects the tab: results go to the main gallery and the history like the other
    image tools. `extra_buttons` (other Cloud buttons of the app, e.g. the caption
    editor's Describe) follow the switches saved here."""
    import gradio as gr
    p["op"].change(op_help, [p["op"]], [p["help"]])

    def go(dry_run):
        def fn(op, src, r1, r2, r3, r4, prompt, mp, w, h, sm, od, of, hist, *opt_vals):
            opts = dict(zip(OPT_KEYS, opt_vals))
            imgs, msg, new_hist = handle(op, dry_run, src, [r1, r2, r3, r4], prompt, mp,
                                         opts, w, h, sm, od, of, hist)
            if dry_run:
                return gr.update(), msg, gr.update(), hist, gr.update()
            return imgs, msg, msg, new_hist, new_hist
        return fn
    ins = ([p["op"], p["src"], *p["refs"], p["prompt"], main_prompt, width, height,
            save_mode, output_dir, output_format, history] + [p[k] for k in OPT_KEYS])
    outs = [out, p["status"], report, history, history_gallery]
    p["quote_btn"].click(go(True), ins, outs)
    p["run_btn"].click(go(False), ins, outs)

    extra = list(extra_buttons or [])

    def save(enabled, key):
        cleared, msg, ok = settings_save(enabled, key)
        return (cleared, msg, gr.update(interactive=ok), gr.update(interactive=ok),
                banner(), "" if ok else f"*Buttons greyed out: {api.status()[1]}*",
                *[gr.update(interactive=ok) for _ in extra])
    p["save"].click(save, [p["enabled"], p["key"]],
                    [p["key"], p["settings_status"], p["quote_btn"], p["run_btn"],
                     p["banner"], p["status"], *extra])

    # The buttons' state is read again at every page load: a switch saved here, or in
    # preferences.json, holds after a reload without restarting the app.
    def refresh():
        ok, why = api.status()
        import cz_magic
        return (gr.update(interactive=ok), gr.update(interactive=ok), banner(),
                "" if ok else f"*Buttons greyed out: {why}*",
                gr.update(value=cz_magic.cloud_enabled()),
                *[gr.update(interactive=ok) for _ in extra])
    if demo is not None:
        demo.load(refresh, None, [p["quote_btn"], p["run_btn"], p["banner"], p["status"],
                                  p["enabled"], *extra])
