"""Video of tested rich evidence and unedited Llama outputs, including limitations."""
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'
DATA = OUT / 'advisory-rich-tested'
W,H,FPS = 1600,900,12
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
WHITE,GREY,CYAN,ORANGE = '#edf3fa','#bbc8d8','#7ed6ff','#ffc080'


def text(im, value, x, y, width, size=25, color=WHITE, bold=False):
    d=ImageDraw.Draw(im)
    f=ImageFont.truetype(BOLD if bold else FONT,size)
    for para in value.split('\n'):
        line=''
        for word in para.split():
            trial=(line+' '+word).strip()
            if line and d.textlength(trial,font=f)>width:
                d.text((x,y),line,font=f,fill=color)
                y+=size+8
                line=word
            else:
                line=trial
        d.text((x,y),line,font=f,fill=color)
        y+=size+8
    return y


def page(title,subtitle):
    im=Image.new('RGB',(W,H),'#101927')
    text(im,title,45,30,1510,38,bold=True)
    text(im,subtitle,45,95,1510,23,GREY)
    text(im,'Saved local inference | Llama 3.2 1B | experimental, not automatic wafer acceptance',45,853,1510,20,GREY)
    return im


frames=[]
p=page('Richer YOLO evidence -> Llama','Tested on all seven local samples; selected examples follow')
text(p,'New evidence: box size, image position, area, edge contact, overlapping predictions, low-score candidates and historical class metrics.',80,220,1430,34,CYAN)
text(p,'All 7 responses passed structural checks after bounding output length and candidate IDs. Requests took 4.9-8.6 seconds.',80,420,1430,29)
text(p,'Important: valid JSON does not mean valid advice. The video includes actual model errors, not cleaned-up answers.',80,610,1430,29,ORANGE)
frames.append((p,8))
cases=[('10206','Model claims the region is visually clear, but received no pixels. Only one candidate exists, despite its reference to both candidates.'),
       ('10792','Valid IDs and checks, but the advice is generic: it does not exploit the small region size.'),
       ('10720','Model invents a geometry score. No such measurement exists; coordinates must not be treated as confidence.'),
       ('10775','Check all asserted boundary contacts against the structured evidence. Overlap does not prove several physical defects.')]
for prefix,warning in cases:
    ep=next(DATA.glob(prefix+'*.evidence.json'))
    e=json.loads(ep.read_text())
    a=json.loads(ep.with_name(ep.name.replace('.evidence.json','.advisory.json')).read_text())
    c=e['candidates'][0] if e['candidates'] else None
    p=page('Evidence | sample '+prefix,'Computed from YOLO output; Llama receives JSON, not this image')
    img=Image.open(ROOT/'samples'/ep.name.replace('.evidence.json','.jpg')).convert('RGB')
    img.thumbnail((690,490),Image.Resampling.LANCZOS)
    left,top=45+(690-img.width)//2,180+(490-img.height)//2
    p.paste(img,(left,top))
    draw=ImageDraw.Draw(p)
    for b in e['candidates']:
        x1,y1,x2,y2=b['box_xyxy_normalized']
        draw.rectangle((left+x1*img.width,top+y1*img.height,left+x2*img.width,top+y2*img.height),outline=ORANGE,width=3)
    text(p,f"Candidate boxes >=0.05: {len(e['candidates'])}\nPython policy: {e['policy']['decision']}",45,705,690,27,CYAN)
    if c:
        details=(f"Top candidate #{c['id']}: {c['class']}\nScore: {c['score']} ({c['score_band']})\n"
                 f"Position: {c['image_region']}\nBox size: {c['box_size_source_pixels']} source pixels\n"
                 f"Box area: {c['box_area_percent_of_image']}% of image\nEdges: {', '.join(c['touches_edges']) or 'none'}\n"
                 f"Overlapping box pairs: {len(e['overlapping_predictions'])}\n"
                 f"Class test recall: {c['historical_test_metrics']['recall']} (aggregate)")
        text(p,details,790,190,760,27)
    text(p,'Box area is not defect area. Scores are not calibrated probabilities.',790,690,760,25,ORANGE)
    frames.append((p,9))
    p=page('Actual Llama response | sample '+prefix,f"Recorded request: {a['elapsed_seconds']:.2f} seconds | structural validation passed")
    advice=a['advisory']
    y=text(p,'SUMMARY: '+advice['summary'],45,170,1490,25,CYAN)
    for priority in advice['priorities']:
        y=text(p,f"IDs {priority['candidate_ids']} | {priority['check']}",45,y+15,1490,25,bold=True)
        y=text(p,priority['why'],45,y,1490,24)
    y=text(p,'UNCERTAINTY: '+advice['uncertainty'],45,y+15,1490,24,GREY)
    if y>700:
        raise RuntimeError('Response text exceeds available slide space')
    text(p,'REVIEW NOTE: '+warning,45,730,1490,25,ORANGE)
    frames.append((p,13))
p=page('Test conclusion','Richer evidence is available; model reasoning remains the bottleneck')
text(p,'The interface now supplies much more useful inspection context.\nLlama 1B still produces generic advice, invented measurements and incomplete phrases.\nKeep factual measurements and Python policy separate from model-generated prose.',80,220,1430,31)
text(p,'Next improvement: compare a stronger text model on the same saved evidence.\nImages: Wafer Defect by Wafer, Roboflow Universe, CC BY 4.0.',80,575,1430,27,ORANGE)
frames.append((p,9))
video=OUT/'yolo_llama_rich_evidence_demo.mp4'
proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24',
    '-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','-','-an','-c:v','libx264','-crf','20',
    '-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
try:
    for frame,seconds in frames:
        data=frame.tobytes()
        for _ in range(FPS*seconds):
            proc.stdin.write(data)
finally:
    proc.stdin.close()
if proc.wait():
    raise SystemExit('Encoding failed')
frames[1][0].save(OUT/'rich_evidence_preview.jpg')
frames[6][0].save(OUT/'rich_response_preview.jpg')
print(video)
