"""Verify changed content is confined to the four chart images."""
from pathlib import Path
import json,zipfile
from pptx import Presentation
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'Overcoming-Template-Mega-Clusters-and-Pooling-Cancellation-in-Unsupervised-Incident-Triage-2/Incident_Triage_Poster_With_Paper_Figures.pptx'
new=ROOT/'poster_work/Incident_Triage_Poster_High_Resolution.pptx'
# Final export path is supplied as an argument.
import sys
if len(sys.argv)>1:new=ROOT/sys.argv[1]
a,b=Presentation(old),Presentation(new)
assert (a.slide_width,a.slide_height)==(b.slide_width,b.slide_height)
assert len(a.slides)==len(b.slides)==1
assert len(a.slides[0].shapes)==len(b.slides[0].shapes)
changed=[]
for i,(x,y) in enumerate(zip(a.slides[0].shapes,b.slides[0].shapes)):
 assert (x.left,x.top,x.width,x.height)==(y.left,y.top,y.width,y.height),(i,'geometry')
 if x.has_text_frame:
  assert x.text==y.text,(i,'text')
  assert x._element.xml==y._element.xml,(i,'text formatting')
 elif hasattr(x,'image'):
  if x.image.blob!=y.image.blob:
   changed.append({'shape':i,'pixels':y.image.size,'panel_aspect':round(y.width/y.height,6),'image_aspect':round(y.image.size[0]/y.image.size[1],6)})
   assert abs(y.width/y.height-y.image.size[0]/y.image.size[1])<.002
assert len(changed)==4,changed
result={'slides':1,'size_inches':[b.slide_width/914400,b.slide_height/914400],'unchanged_text_formatting_and_bounds':True,'changed_images':changed}
print(json.dumps(result,indent=2))
(ROOT/'poster_work/hires_figures/poster-verification.json').write_text(json.dumps(result,indent=2))
