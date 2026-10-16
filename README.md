# MODEX B&W → SVG Telegram Bot

Converts black-and-white raster images into **real SVG vector paths** with:
- Black foreground (`#000000`)
- Transparent background (no background rectangle)
- Enclosed cutouts preserved using SVG `fill-rule="evenodd"`
- Automatic Otsu thresholding or an adjustable manual threshold
- Small-speck removal for a bolder silhouette
- Cropped SVG viewBox around the detected artwork

## Telegram menu

The bot includes both:
- An inline welcome menu shown by `/start`, with buttons for conversion, automatic mode, threshold settings, status, and help.
- Telegram's native **Menu** button beside the message bar, registered automatically with `set_my_commands()` and `set_chat_menu_button()` at startup.

## Telegram commands

- `/start` — welcome and instructions
- `/help` — help
- `/auto` — automatic thresholding for each image
- `/threshold 160` — manual threshold from 1 to 254
- `/threshold` — show threshold settings
- `/status` — show current settings

Send an image as a Telegram photo or image document. The bot returns an SVG file.

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
python bot.py
```

On Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
$env:TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
python bot.py
```

Create a bot token through Telegram's official `@BotFather`. Do not hardcode the token in the source code.

## Deploy on Railway

1. Create a GitHub repository and upload the files in this folder.
2. In Railway, create a project and deploy the repository.
3. Open the service's **Variables** section.
4. Add `TELEGRAM_BOT_TOKEN` with the token from BotFather.
5. The included `Procfile` starts the bot as a worker. If a start command is requested, use `python bot.py`.
6. Deploy and check logs for `Starting MODEX B&W to SVG bot`.

### Persist settings across redeploys

Threshold preferences are stored in SQLite at `DATA_DIR/settings.db`. Railway's default filesystem may be ephemeral. To persist settings:
1. Attach a Railway Volume to the service.
2. Set the mount path to `/data`.
3. Add variable `DATA_DIR=/data`.
4. Redeploy.

Without a volume, the bot still works, but saved settings may reset when the container is replaced.

### Hosting note

This bot uses Telegram long polling and should run as a worker/background service, not as an HTTP web server. Run only one replica for a given bot token.

## Logo tracing engine

The bot uses VTracer spline tracing with corner detection to produce cleaner curves and crisp logo edges. It thresholds the image first, avoids blur and morphological opening, and filters tiny isolated components conservatively. White tracing/background shapes are removed from the final SVG so the canvas remains transparent. For best results, send a high-resolution PNG as a document.

## How tracing works

1. Normalize the image and convert it to grayscale.
2. Use Otsu thresholding or the selected manual threshold.
3. Respect existing transparency.
4. Remove small noise and contours.
5. Simplify contours and write them as SVG paths.
6. Use VTracer spline tracing with corner detection for cleaner paths.
7. Remove white tracing/background shapes so the SVG background is transparent.
8. Do not embed the original bitmap in the SVG.

## Limitations

This is deterministic bitmap tracing, not an AI redraw. Blurry or jagged source images can produce imperfect paths. Automatic thresholding works best when foreground and background have clear contrast. Tiny details may be removed intentionally for a bold silhouette. For fine-detail logos, start with a high-resolution source and lower `MIN_COMPONENT_AREA` in `bot.py` if needed.
