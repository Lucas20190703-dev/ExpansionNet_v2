from PIL import Image
from pathlib import Path

# Load your image (use PNG or JPEG)
icon_path = Path(__file__).parent / "resources" / "icons" / "appicon.png"

img = Image.open(icon_path)

# Resize for icon (recommended: 256x256 or provide multiple sizes)
img.save("icon.ico", format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])