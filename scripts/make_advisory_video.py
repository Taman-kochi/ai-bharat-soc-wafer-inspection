"""Render saved local evidence and Llama responses into an explanatory video."""
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'
W, H, FPS = 1600, 900, 12
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
white, grey, cyan, green, orange = '#edf3fa', '#b8c6d8', '#78cfff', '#6ce3ac', '#ffbd79'


def text(im, value, x, y, width, size=25, color=white, bold=False):
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(BOLD if bold else FONT, size)
    for para in value.split('\n'):
        line = ''
        for word in para.split():
            nxt = (line + ' ' + word).strip()
            if line and draw.textlength(nxt, font=font) > width:
                draw.text((x, y), line, font=font, fill=color)
                y += size + 9
                line = word
            else:
                line = nxt
        draw.text((x, y), line, font=font, fill=color)
        y += size + 9
    return y


def base(title, subtitle):
    im = Image.new('RGB', (W, H), '#101927')
    text(im, title, 45, 30, 1510, 38, bold=True)
    text(im, subtitle, 45, 92, 1510, 23, grey)
    text(im, 'Local prototype | saved real outputs | no live camera | no audio', 45, 852, 1510, 20, grey)
    return im


frames = []
im = base('YOLO + Llama 3.2 1B', 'Structured evidence, a system prompt, and a review explanation')
text(im, '1. YOLO detects regions on this computer.\n2. Python packages classes, scores and coordinates as JSON.\n3. Python applies a provisional 0.5 review threshold.\n4. Llama receives text only and explains that decision.', 90, 220, 1400, 32)
text(im, 'Llama does not see pixels or reference labels. REVIEW means human inspection, not a proven defect.', 90, 510, 1400, 29, orange)
text(im, 'This video replays saved local inference results; playback timing is for readability.', 90, 665, 1400, 25, grey)
frames.append((im, 7))

im = base('The system prompt', 'Actual prompt used by scripts/advise.py')
prompt = ('Return JSON. Copy policy_decision and eligible_detection_ids exactly. '
          'For reason, restate the supplied instruction in one sentence. '
          'Do not infer causes, visual details, severity, or PASS/FAIL.')
text(im, prompt, 90, 235, 1390, 36, cyan)
text(im, 'Output fields: decision, reason, detection_ids\nThe JSON schema constrains the decision and evidence references.\nIt does not guarantee that the explanation is factually correct.', 90, 530, 1390, 28, orange)
frames.append((im, 9))

prefixes = ['10206', '10113', '10284', '10076', '10792']
for index, prefix in enumerate(prefixes, 1):
    ep = next((OUT / 'advisory').glob(prefix + '*.evidence.json'))
    ap = ep.with_name(ep.name.replace('.evidence.json', '.advisory.json'))
    evidence, advisory = json.loads(ep.read_text()), json.loads(ap.read_text())
    image_path = ROOT / 'samples' / ep.name.replace('.evidence.json', '.jpg')
    image = Image.open(image_path).convert('RGB')
    im = base(f'Example {index}/5 | {prefix}', 'Image shown for you; Llama receives only structured text')
    image.thumbnail((650, 390), Image.Resampling.LANCZOS)
    left, top = 45 + (650-image.width)//2, 175 + (390-image.height)//2
    im.paste(image, (left, top))
    draw = ImageDraw.Draw(im)
    for det in evidence['detections']:
        x1,y1,x2,y2 = det['box_xyxy_normalized']
        draw.rectangle((left+x1*image.width, top+y1*image.height,
                        left+x2*image.width, top+y2*image.height), outline=orange, width=3)
    text(im, 'YOLO evidence (excerpt)', 45, 595, 650, 26, cyan, True)
    summary = '\n'.join(f"id={d['id']} | {d['class']} | score={d['confidence']:.4f}"
                        for d in evidence['detections']) or 'detections: []'
    text(im, summary, 45, 640, 650, 25)
    text(im, 'Normalized box coordinates are also included in the JSON.', 45, 755, 650, 20, grey)
    text(im, 'PYTHON POLICY: ' + advisory['policy_decision'], 750, 175, 800, 31, green, True)
    y = text(im, advisory['policy_summary'], 750, 230, 790, 25)
    text(im, 'LLAMA RESPONSE (verbatim)', 750, max(355, y+25), 790, 27, cyan, True)
    y = text(im, advisory['raw_response'], 750, max(405, y+75), 790, 24)
    text(im, f"Recorded request: {advisory['elapsed_seconds']:.2f} s", 750, max(650,y+20), 790, 23, grey)
    text(im, 'Model prose is experimental, even when its JSON passes validation.', 750, 750, 790, 23, orange)
    frames.append((im, 10))

im = base('What this demonstrates', 'A working local interface, with an important model limitation')
text(im, 'The 1B model made comparison errors with rules-only prompts.\nPython now computes the threshold decision and factual summary.\nLlama writes an experimental explanation, saved separately.\nConfidently wrong YOLO predictions still trigger REVIEW.', 90, 220, 1400, 30)
text(im, 'No detections does not establish a clean wafer.\nImages: Wafer Defect by Wafer, Roboflow Universe, CC BY 4.0.', 90, 565, 1400, 26, orange)
frames.append((im, 8))

video = OUT / 'yolo_llama_advisory_demo.mp4'
proc = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo',
    '-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','-',
    '-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)], stdin=subprocess.PIPE)
try:
    for frame, seconds in frames:
        for _ in range(seconds*FPS):
            proc.stdin.write(frame.tobytes())
finally:
    proc.stdin.close()
if proc.wait():
    raise SystemExit('Encoding failed')
frames[2][0].save(OUT / 'yolo_llama_advisory_preview.jpg')
print(video)
