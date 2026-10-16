import io
import os
import sqlite3
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import vtracer
from PIL import Image, ImageOps, UnidentifiedImageError
from telegram import (
    BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, MenuButtonCommands, Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters,
)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "settings.db"

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_THRESHOLD = 160
MIN_THRESHOLD = 1
MAX_THRESHOLD = 254
MIN_COMPONENT_AREA = 2.0
PADDING = 2

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("modex-bw-to-svg")


def db_connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                mode TEXT NOT NULL DEFAULT 'auto',
                threshold INTEGER NOT NULL DEFAULT 160
            )
        """)


def get_settings(user_id: int) -> tuple[str, int]:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT mode, threshold FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            return row[0], int(row[1])
        conn.execute(
            "INSERT OR IGNORE INTO user_settings (user_id, mode, threshold) VALUES (?, 'auto', ?)",
            (user_id, DEFAULT_THRESHOLD),
        )
        return "auto", DEFAULT_THRESHOLD


def set_settings(user_id: int, mode: str, threshold: int | None = None):
    _, current_threshold = get_settings(user_id)
    if threshold is None:
        threshold = current_threshold
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO user_settings (user_id, mode, threshold)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                mode = excluded.mode,
                threshold = excluded.threshold
        """, (user_id, mode, int(threshold)))


def threshold_mask(gray: np.ndarray, mode: str, threshold: int):
    """White pixels in the returned mask represent the foreground artwork."""
    if mode == "auto":
        used_threshold, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        return mask, int(used_threshold)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask, int(threshold)


def make_svg(image_bytes: bytes, mode: str, threshold: int):
    """Trace black logo artwork with VTracer spline paths and a transparent canvas."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("I couldn't read that image. Please send a PNG, JPG, WEBP, BMP or TIFF.") from exc

    rgba = np.array(image)
    alpha = rgba[:, :, 3]
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    mask, used_threshold = threshold_mask(gray, mode, threshold)
    mask[alpha < 10] = 0

    # Remove only tiny isolated components. No blur or morphology is applied,
    # so thin cuts and sharp logo details are not rounded or erased.
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == label] = 255

    if not np.any(cleaned):
        raise ValueError("No dark artwork was detected. Try /threshold 190, or send a higher-contrast image.")

    # VTracer takes a raster file. Feed it a clean black/white image; then
    # convert its traced contours into a single even-odd path so enclosed
    # cutouts stay empty instead of being filled with white.
    h, w = cleaned.shape
    bw = np.full((h, w, 3), 255, dtype=np.uint8)
    bw[cleaned > 0] = (0, 0, 0)

    with tempfile.TemporaryDirectory(prefix="modex_vtracer_") as tmp:
        input_path = Path(tmp) / "input.png"
        output_path = Path(tmp) / "traced.svg"
        Image.fromarray(bw, mode="RGB").save(input_path, format="PNG")
        try:
            vtracer.convert_image_to_svg_py(
                str(input_path),
                str(output_path),
                colormode="binary",
                hierarchical="stacked",
                mode="spline",
                filter_speckle=VTRACER_FILTER_SPECKLE,
                color_precision=6,
                layer_difference=16,
                corner_threshold=VTRACER_CORNER_THRESHOLD,
                length_threshold=VTRACER_LENGTH_THRESHOLD,
                max_iterations=10,
                splice_threshold=45,
                path_precision=VTRACER_PATH_PRECISION,
            )
        except Exception as exc:
            logger.exception("VTracer conversion failed")
            raise ValueError("VTracer couldn't trace this image. Try a higher-resolution PNG or JPG.") from exc

        try:
            root = ET.parse(output_path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError("The vectorizer returned an invalid SVG. Please try another image.") from exc

    # Collect vector path data. VTracer's binary output can contain white
    # regions as separate layers; edge-touching white regions are background.
    # Interior contours are included in one even-odd path to preserve cutouts.
    path_data = []
    number_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
    ns = "{http://www.w3.org/2000/svg}"

    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        d = element.attrib.get("d", "").strip()
        if not d:
            continue
        fill = element.attrib.get("fill", "#000000").strip().lower()
        nums = [float(n) for n in number_re.findall(d)]
        if len(nums) >= 4:
            xs = nums[0::2]
            ys = nums[1::2]
            touches_edge = (
                min(xs) <= 1.0 or min(ys) <= 1.0
                or max(xs) >= w - 1.0 or max(ys) >= h - 1.0
            )
        else:
            touches_edge = False

        is_white = fill in {"white", "#fff", "#ffffff", "rgb(255,255,255)"}
        if is_white and touches_edge:
            continue
        path_data.append(d)

    if not path_data:
        raise ValueError("No usable vector paths were generated. Try /threshold 190.")

    # Trim the canvas to the detected foreground bounds and translate paths.
    ys, xs = np.where(cleaned > 0)
    min_x = max(0, int(xs.min()) - PADDING)
    min_y = max(0, int(ys.min()) - PADDING)
    max_x = min(w - 1, int(xs.max()) + PADDING)
    max_y = min(h - 1, int(ys.max()) + PADDING)
    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)

    # Translate using an SVG group; VTracer path coordinates remain untouched.
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{min_x} {min_y} {width} {height}">
  <path d="{' '.join(path_data)}" fill="#000000" fill-rule="evenodd" clip-rule="evenodd"/>
</svg>
""".encode("utf-8")

    return svg, {
        "width": width,
        "height": height,
        "contours": len(path_data),
        "threshold": used_threshold,
        "mode": mode,
        "engine": "VTracer spline",
    }


HELP_TEXT = (
    "🖤 *MODEX B&W → SVG*\n\n"
    "Send a logo or black-and-white image and I'll trace it with VTracer spline paths. "
    "The foreground is black and the background transparent.\n\n"
    "*Commands*\n"
    "/auto — automatic Otsu thresholding\n"
    "/threshold 160 — manual threshold from 1 to 254\n"
    "/status — show your current settings\n"
    "/help — show this help\n\n"
    "*Tips*\n"
    "• Dark pixels become black vector shapes.\n"
    "• White areas become transparent, including enclosed cutouts.\n"
    "• VTracer spline tracing gives cleaner curves with corner detection.\n"
    "• Tiny isolated specks are filtered conservatively.\n"
    "• Send the highest-resolution image for cleaner paths.\n\n"
    "This is bitmap tracing, not AI redraw. Very fine details may depend on source quality."
)


def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🖼 Convert an image", callback_data="menu_convert")],
        [InlineKeyboardButton("⚙️ Automatic mode", callback_data="menu_auto"),
         InlineKeyboardButton("🎚 Threshold settings", callback_data="menu_threshold")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="menu_help"),
         InlineKeyboardButton("📊 My settings", callback_data="menu_status")],
    ])


async def post_init(application: Application):
    """Register Telegram's native command menu beside the message bar."""
    await application.bot.set_my_commands([
        BotCommand("start", "Open the main menu"),
        BotCommand("auto", "Automatic black and white detection"),
        BotCommand("threshold", "Set or view threshold"),
        BotCommand("status", "View your current settings"),
        BotCommand("help", "Show instructions"),
    ])
    await application.bot.set_chat_menu_button(menu_button=MenuButtonCommands())


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    action = query.data

    if action == "menu_convert":
        await query.message.reply_text("Send me a black-and-white image as a photo or image document, and I'll return a true SVG with black paths and a transparent background.")
    elif action == "menu_auto":
        _, threshold = get_settings(user_id)
        set_settings(user_id, "auto", threshold)
        await query.message.reply_text("Automatic threshold enabled. Send an image to convert it. Use /threshold 160 for manual thresholding.")
    elif action == "menu_threshold":
        mode, threshold = get_settings(user_id)
        await query.message.reply_text(
            f"Current mode: {mode}\nManual threshold: {threshold}\n\n"
            "Set a value by sending /threshold 160 (choose 1–254). Higher values include more gray pixels."
        )
    elif action == "menu_status":
        mode, threshold = get_settings(user_id)
        shown = "Otsu (automatic per image)" if mode == "auto" else str(threshold)
        await query.message.reply_text(
            f"Mode: {mode}\nThreshold: {shown}\n"
            "Output: black #000000 SVG paths, transparent background"
        )
    elif action == "menu_help":
        await query.message.reply_text(HELP_TEXT, parse_mode="Markdown", reply_markup=main_menu_keyboard())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    get_settings(update.effective_user.id)
    await update.message.reply_text(
        "🖤 *MODEX B&W → SVG*\n\n"
        "Welcome! Send a black-and-white image and I'll trace it into a real SVG path. "
        "The foreground is black and the background is transparent.\n\n"
        "Choose an option below, or tap the Telegram menu beside the message bar.",
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _, threshold = get_settings(user_id)
    set_settings(user_id, "auto", threshold)
    await update.message.reply_text(
        "Automatic threshold enabled. Send an image to convert it.\n"
        "Use /threshold 160 to switch to a manual threshold."
    )


async def threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        mode, threshold = get_settings(user_id)
        await update.message.reply_text(
            f"Current mode: {mode}\nManual threshold: {threshold}\n\n"
            "Usage: /threshold 160 (choose a value from 1 to 254).\n"
            "Lower values keep only darker pixels; higher values include more gray pixels."
        )
        return
    try:
        value = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Please use a number, e.g. /threshold 160.")
        return
    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        await update.message.reply_text("Choose a threshold between 1 and 254.")
        return
    set_settings(user_id, "manual", value)
    await update.message.reply_text(
        f"Manual threshold set to {value}. Send an image to convert it.\n"
        "Use /auto to return to automatic thresholding."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode, threshold = get_settings(update.effective_user.id)
    shown = "Otsu (automatic per image)" if mode == "auto" else str(threshold)
    await update.message.reply_text(
        f"Mode: {mode}\nThreshold: {shown}\n"
        "Output: black #000000 SVG paths, transparent background\n"
        f"Speck filter: connected components smaller than {MIN_COMPONENT_AREA:g} px are removed."
    )


async def get_image_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if message.photo:
        file_obj = await context.bot.get_file(message.photo[-1].file_id)
        filename = "image.jpg"
    elif message.document:
        doc = message.document
        mime = (doc.mime_type or "").lower()
        name = doc.file_name or "image"
        allowed_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
        if not (mime.startswith("image/") or name.lower().endswith(allowed_ext)):
            raise ValueError("Please send an image file (PNG, JPG, WEBP, BMP or TIFF).")
        if doc.file_size and doc.file_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("That file is too large. Please send an image under 20 MB.")
        file_obj = await context.bot.get_file(doc.file_id)
        filename = name
    else:
        raise ValueError("Please send an image as a photo or image document.")

    downloaded = await file_obj.download_as_bytearray()
    if len(downloaded) > MAX_DOWNLOAD_BYTES:
        raise ValueError("That file is too large. Please send an image under 20 MB.")
    return bytes(downloaded), filename


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    try:
        image_bytes, original_name = await get_image_bytes(update, context)
        mode, threshold = get_settings(update.effective_user.id)
        await message.reply_text("Tracing your image with VTracer spline mode…")
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT
        )
        svg_bytes, info = make_svg(image_bytes, mode, threshold)
        stem = Path(original_name).stem or "vector"
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:48]
        output = io.BytesIO(svg_bytes)
        output.name = f"{safe_stem}_black_transparent.svg"
        caption = (
            f"✅ SVG ready\nCanvas: {info['width']} × {info['height']} px\n"
            f"Threshold: {info['threshold']} ({info['mode']})\n"
            f"Vector paths: {info['contours']}\n"
            "Engine: VTracer spline\n"
            "Foreground: #000000 · Background: transparent"
        )
        await message.reply_document(document=output, filename=output.name, caption=caption)
    except ValueError as exc:
        await message.reply_text(str(exc))
    except Exception:
        logger.exception("Image conversion failed")
        await message.reply_text(
            "Something went wrong while converting that image. Try a PNG/JPG with stronger black-and-white contrast."
        )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Unknown command. Use /help to see available commands.")


def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN environment variable. Add your bot token in Railway Variables."
        )
    init_db()
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^menu_"))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("auto", auto_command))
    application.add_handler(CommandHandler("threshold", threshold_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    logger.info("Starting MODEX B&W to SVG bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
