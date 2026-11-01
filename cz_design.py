"""crispz-ideogram - design presets (lot 2): a size and a caption skeleton per format.

Each preset is what a designer would set up before the first word: the canvas (real
proportions, multiples of 16, within Ideogram 4's 256..2048 and 6:1), a graphic-design
style, a background, and placeholder elements with boxes where that format puts them.
Loading one fills the caption editor; the placeholders (in CAPITALS) are meant to be
replaced. Boxes are [y_min, x_min, y_max, x_max] in 0..1000.

No torch, no gradio: cz_ui reads PRESETS, cz_caption builds the JSON.
"""

import cz_caption

# name -> {size, hint, high_level_description, style, background, elements}
PRESETS = {
    "Poster (A-series, portrait)": {
        "size": (1440, 2032),
        "hint": "A-series proportions (1:1.41), printable at A4/A3",
        "high_level_description": "A bold event poster, flat graphic design with a large "
                                  "headline, a central illustration and event details.",
        "style": {"kind": "art", "aesthetics": "bold, high contrast, clean layout",
                  "lighting": "flat, even", "medium": "graphic_design",
                  "detail": "flat vector illustration, strong sans-serif typography"},
        "background": "A solid colour field filling the whole poster with a subtle paper "
                      "grain, no scene and no depth.",
        "elements": [
            {"type": "text", "text": "HEADLINE", "bbox": [60, 80, 230, 920],
             "desc": "Very large bold condensed sans-serif headline across the top."},
            {"type": "obj", "bbox": [260, 120, 760, 880],
             "desc": "A central flat illustration of THE SUBJECT, simple bold shapes."},
            {"type": "text", "text": "DATE · PLACE", "bbox": [800, 120, 880, 880],
             "desc": "Medium uppercase details line, wide letter spacing, centered."},
        ],
    },
    "Logo (square)": {
        "size": (1024, 1024),
        "hint": "1:1, a mark and a wordmark on a plain field",
        "high_level_description": "A minimalist logo: a simple geometric mark above a "
                                  "clean wordmark, centered on a plain background.",
        "style": {"kind": "art", "aesthetics": "minimal, geometric, balanced",
                  "lighting": "flat, even", "medium": "graphic_design",
                  "detail": "flat vector logo, two colours, no gradients"},
        "background": "A plain solid background, no texture, no scene.",
        "elements": [
            {"type": "obj", "bbox": [200, 330, 560, 670],
             "desc": "A simple geometric symbol representing THE BRAND IDEA."},
            {"type": "text", "text": "BRAND", "bbox": [620, 200, 760, 800],
             "desc": "Clean bold sans-serif wordmark centered below the symbol."},
        ],
    },
    "Business card (85 x 55 mm)": {
        "size": (1584, 1024),
        "hint": "European card proportions (85 x 55 mm), front side",
        "high_level_description": "The front of a clean modern business card, flat "
                                  "graphic design.",
        "style": {"kind": "art", "aesthetics": "minimal, professional, generous whitespace",
                  "lighting": "even, diffuse", "medium": "graphic_design",
                  "detail": "flat vector design, sans-serif typography"},
        "background": "A solid off-white card surface with a subtle paper texture filling "
                      "the frame.",
        "elements": [
            {"type": "text", "text": "FIRST LAST", "bbox": [330, 90, 470, 700],
             "desc": "Bold dark sans-serif name, left aligned."},
            {"type": "text", "text": "Job title", "bbox": [490, 90, 560, 700],
             "desc": "Smaller grey job title under the name."},
            {"type": "text", "text": "name@company.com", "bbox": [780, 90, 860, 700],
             "desc": "Small contact line near the bottom left."},
        ],
    },
    "Book cover (2:3)": {
        "size": (1344, 2016),
        "hint": "2:3, trade paperback / ebook cover",
        "high_level_description": "A book cover with a striking central image, the title "
                                  "at the top and the author's name at the bottom.",
        "style": {"kind": "art", "aesthetics": "dramatic, elegant, cinematic",
                  "lighting": "moody, directional", "medium": "illustration",
                  "detail": "painterly digital illustration, refined serif typography"},
        "background": "An atmospheric backdrop matching the book's mood, filling the "
                      "whole cover.",
        "elements": [
            {"type": "text", "text": "TITLE", "bbox": [70, 100, 250, 900],
             "desc": "Large elegant serif title across the top third."},
            {"type": "obj", "bbox": [280, 120, 800, 880],
             "desc": "The central image: THE STORY'S KEY SUBJECT."},
            {"type": "text", "text": "AUTHOR NAME", "bbox": [850, 200, 930, 800],
             "desc": "Author name in small capitals near the bottom edge."},
        ],
    },
    "Slide (16:9)": {
        "size": (2048, 1152),
        "hint": "16:9 presentation slide",
        "high_level_description": "A clean presentation slide with a title, a short "
                                  "subtitle and a simple illustration on the right.",
        "style": {"kind": "art", "aesthetics": "corporate, clean, airy",
                  "lighting": "flat, even", "medium": "graphic_design",
                  "detail": "flat vector shapes, modern sans-serif typography"},
        "background": "A light solid background with a thin accent band along the left "
                      "edge.",
        "elements": [
            {"type": "text", "text": "SLIDE TITLE", "bbox": [180, 70, 330, 600],
             "desc": "Large bold dark title, left aligned."},
            {"type": "text", "text": "One-line key message", "bbox": [360, 70, 440, 600],
             "desc": "Medium grey subtitle under the title."},
            {"type": "obj", "bbox": [200, 640, 820, 950],
             "desc": "A simple flat illustration of THE TOPIC on the right."},
        ],
    },
    "Packaging (front, square)": {
        "size": (1536, 1536),
        "hint": "1:1 front face of a box, a can label or a pouch",
        "high_level_description": "The front face of a product package, flat packaging "
                                  "design with the brand, the product name and a hero "
                                  "illustration.",
        "style": {"kind": "art", "aesthetics": "appetizing, premium, bold",
                  "lighting": "flat, even", "medium": "graphic_design",
                  "detail": "flat packaging design, bold display typography"},
        "background": "A solid brand colour filling the package face, no shelf and no "
                      "photo context.",
        "elements": [
            {"type": "text", "text": "BRAND", "bbox": [60, 250, 180, 750],
             "desc": "Brand wordmark centered at the top."},
            {"type": "obj", "bbox": [220, 180, 700, 820],
             "desc": "Hero illustration of THE PRODUCT in the center."},
            {"type": "text", "text": "PRODUCT NAME", "bbox": [740, 120, 860, 880],
             "desc": "Large product name below the illustration."},
        ],
    },
    "YouTube thumbnail (16:9)": {
        "size": (1280, 720),
        "hint": "YouTube's own 1280 x 720",
        "high_level_description": "An eye-catching YouTube thumbnail with a huge short "
                                  "headline and one strong subject.",
        "style": {"kind": "art", "aesthetics": "punchy, saturated, high contrast",
                  "lighting": "bright, crisp", "medium": "graphic_design",
                  "detail": "bold outlined display type, cut-out subject"},
        "background": "A saturated colour gradient filling the frame.",
        "elements": [
            {"type": "text", "text": "BIG HOOK", "bbox": [120, 50, 520, 560],
             "desc": "Huge heavy uppercase headline with a thick dark outline, left half."},
            {"type": "obj", "bbox": [80, 560, 1000, 980],
             "desc": "THE SUBJECT, large, cut out, on the right half."},
        ],
    },
    "Vertical story (9:16)": {
        "size": (1152, 2048),
        "hint": "9:16 Instagram / TikTok story",
        "high_level_description": "A vertical social media story with a headline at the "
                                  "top, a central visual and a call to action at the "
                                  "bottom.",
        "style": {"kind": "art", "aesthetics": "modern, energetic, social media",
                  "lighting": "bright, even", "medium": "graphic_design",
                  "detail": "flat shapes, bold rounded sans-serif typography"},
        "background": "A vertical colour gradient filling the frame.",
        "elements": [
            {"type": "text", "text": "HEADLINE", "bbox": [120, 80, 260, 920],
             "desc": "Bold headline in the upper part, centered."},
            {"type": "obj", "bbox": [300, 100, 760, 900],
             "desc": "The central visual: THE SUBJECT."},
            {"type": "text", "text": "SWIPE UP", "bbox": [820, 250, 900, 750],
             "desc": "Call to action in a rounded pill shape near the bottom."},
        ],
    },
    "Banner (6:1)": {
        "size": (1920, 320),
        "hint": "6:1, Ideogram 4's widest ratio (web header, roll-up top)",
        "high_level_description": "A wide horizontal web banner with a headline on the "
                                  "left and a small illustration on the right.",
        "style": {"kind": "art", "aesthetics": "clean, wide, uncluttered",
                  "lighting": "flat, even", "medium": "graphic_design",
                  "detail": "flat vector, bold sans-serif typography"},
        "background": "A solid colour band filling the whole banner.",
        "elements": [
            {"type": "text", "text": "HEADLINE", "bbox": [200, 40, 800, 600],
             "desc": "Bold headline on one line, left aligned, vertically centered."},
            {"type": "obj", "bbox": [100, 700, 900, 960],
             "desc": "A small flat illustration of THE SUBJECT at the right end."},
        ],
    },
}


def caption_of(name):
    """The preset's caption dict (keys in the canonical order)."""
    p = PRESETS[name]
    return cz_caption.build(p["high_level_description"], p["style"], p["background"],
                            p["elements"])


def size_of(name):
    return PRESETS[name]["size"]
