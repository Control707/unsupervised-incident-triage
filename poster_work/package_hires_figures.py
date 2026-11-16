from pathlib import Path
import zipfile
root=Path(__file__).resolve().parent
assets=root/'hires_figures'
files=[assets/f'{n}.{e}' for n in ['spatial','rolling','pooling','payoff'] for e in ['png','pdf','svg']]+[assets/'README.txt']
assert all(p.is_file() for p in files)
with zipfile.ZipFile(root/'Poster_High_Resolution_Figures.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.name)
print(root/'Poster_High_Resolution_Figures.zip')
