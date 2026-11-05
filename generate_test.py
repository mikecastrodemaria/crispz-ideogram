from cz_ideogram import IdeogramPipe

# Initialisation avec le bon paramètre
pipe = IdeogramPipe(model_path="F:/sdlibs/models/Stable-diffusion/ideogram/ideogram-instant-nsfw")

# Génération
images = pipe(
    prompt="A poster with 'CRISPZ IDEOGRAM' in bold red text on white background",
    width=1024,
    height=1024,
    num_inference_steps=24,
    seed=42
)

# Sauvegarde
images[0].save("D:/Github/crispz-ideogram/test_output.png")
print("✅ Image générée: D:/Github/crispz-ideogram/test_output.png")
