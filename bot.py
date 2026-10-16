
import io
import os
import re
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
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonCommands,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================
# CONFIGURATION
# ============================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "settings.db"

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_THRESHOLD = 160
MIN_THRESHOLD = 1
MAX_THRESHOLD = 254
MIN_COMPONENT_AREA = 2

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("modex-bw-to-svg")

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


# ============================================================
# DATABASE
# ============================================================

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
            "INSERT OR IGNORE INTO user_settings "
            "(user_id, mode, threshold) VALUES (?, 'auto', ?)",
            (user_id, DEFAULT_THRESHOLD),
        )
        return "auto", DEFAULT_THRESHOLD


def set_settings(
    user_id: int,
    mode: str,
    threshold: int | None = None,
):
    _, old_threshold = get_settings(user_id)
    if threshold is None:
        threshold = old_threshold

    with db_connect() as conn:
        conn.execute("""
            INSERT INTO user_settings (user_id, mode, threshold)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                mode = excluded.mode,
                threshold = excluded.threshold
        """, (user_id, mode, int(threshold)))


# ============================================================
# THRESHOLDING
# ============================================================

def threshold_mask(
    gray: np.ndarray,
    mode: str,
    threshold: int,
):
    """Dark source pixels become white foreground in the mask."""

    if mode == "auto":
        used, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        return mask, int(used)

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )
    return mask, int(threshold)


def remove_tiny_components(mask: np.ndarray) -> np.ndarray:
    """Remove tiny isolated specks without blurring logo edges."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        8,
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == label] = 255

    return cleaned


# ============================================================
# OPENCV FALLBACK
# ============================================================

def opencv_fallback_svg(
    mask: np.ndarray,
) -> tuple[bytes, int, int, int]:
    """Build simple SVG paths using the original full canvas."""

    contours, hierarchy = cv2.findContours(
        mask.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("OpenCV could not find any logo contours.")

    height, width = mask.shape
    hierarchy_rows = hierarchy[0] if hierarchy is not None else None
    paths = []

    for index, contour in enumerate(contours):
        area = abs(float(cv2.contourArea(contour)))
        parent = (
            int(hierarchy_rows[index][3])
            if hierarchy_rows is not None
            else -1
        )

        if area < MIN_COMPONENT_AREA and parent < 0:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.1, perimeter * 0.00035)

        points = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        ).reshape(-1, 2)

        if len(points) < 3:
            continue

        d = f"M {int(points[0][0])} {int(points[0][1])} "
        d += " ".join(
            f"L {int(x)} {int(y)}"
            for x, y in points[1:]
        )
        d += " Z"
        paths.append(d)

    if not paths:
        raise ValueError("OpenCV could not create usable paths.")

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="{SVG_NS}"
     width="{width}px"
     height="{height}px"
     viewBox="0 0 {width} {height}"
     version="1.1">
  <path d="{' '.join(paths)}"
        fill="#000000"
        fill-rule="evenodd"
        clip-rule="evenodd"/>
</svg>
"""

    result = svg.encode("utf-8")
    validate_svg(result)
    return result, width, height, len(paths)


# ============================================================
# SVG SANITIZATION FOR CANVA
# ============================================================

def local_name(tag: str) -> str:
    """Return an XML tag name without its namespace."""
    return tag.rsplit("}", 1)[-1].lower()


def sanitize_vtracer_svg(
    root: ET.Element,
    width: int,
    height: int,
) -> tuple[bytes, int]:
    """
    Normalize VTracer output to a small, standard SVG.

    Preserve groups and transforms. Remove metadata, external
    references, stylesheets, scripts and unsupported elements.
    Keep the full input canvas so paths do not become cropped.
    """

    if not any(
        local_name(element.tag) == "path"
        and element.attrib.get("d", "").strip()
        for element in root.iter()
    ):
        raise ValueError("VTracer returned no path data.")

    # Normalize the outer canvas. This fixes the known problem
    # where the viewBox is offset although paths use local coords.
    clean_root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{width}px",
            "height": f"{height}px",
            "viewBox": f"0 0 {width} {height}",
            "version": "1.1",
        },
    )

    allowed_group_attrs = {
        "transform",
        "opacity",
    }

    allowed_path_attrs = {
        "d",
        "fill",
        "fill-rule",
        "clip-rule",
        "transform",
        "opacity",
        "stroke",
        "stroke-width",
    }

    def copy_safe_element(source, destination):
        tag = local_name(source.tag)

        if tag == "g":
            target = ET.SubElement(
                destination,
                f"{{{SVG_NS}}}g",
            )
            for key, value in source.attrib.items():
                key_name = key.rsplit("}", 1)[-1].lower()
                if key_name in allowed_group_attrs:
                    target.set(key_name, value)

            for child in list(source):
                copy_safe_element(child, target)
            return

        if tag == "path":
            d = source.attrib.get("d", "").strip()
            if not d:
                return

            target = ET.SubElement(
                destination,
                f"{{{SVG_NS}}}path",
            )

            target.set("d", d)

            # Logo output is black. Preserve fill-rule and transforms.
            target.set("fill", "#000000")

            for key, value in source.attrib.items():
                key_name = key.rsplit("}", 1)[-1].lower()

                if key_name in allowed_path_attrs and key_name != "fill":
                    if key_name in {"stroke-width", "opacity"}:
                        try:
                            float(value)
                        except (TypeError, ValueError):
                            continue

                    target.set(key_name, value)

            return

        # Only descend into groups. Ignore metadata, CSS, defs,
        # images, scripts, foreignObject, and external references.
        # This prevents nonportable elements from reaching Canva.
        if tag == "svg":
            for child in list(source):
                copy_safe_element(child, destination)

    for child in list(root):
        copy_safe_element(child, clean_root)

    path_count = sum(
        1 for element in clean_root.iter()
        if local_name(element.tag) == "path"
        and element.attrib.get("d", "").strip()
    )

    if path_count == 0:
        raise ValueError("No usable paths remained after SVG cleanup.")

    result = ET.tostring(
        clean_root,
        encoding="utf-8",
        xml_declaration=True,
    )

    validate_svg(result)
    return result, path_count


def validate_svg(svg_bytes: bytes):
    """Reject malformed SVG and empty path output."""

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise ValueError("The generated SVG is not valid XML.") from exc

    if local_name(root.tag) != "svg":
        raise ValueError("The generated document is not an SVG.")

    viewbox = root.attrib.get("viewBox", "").replace(",", " ").split()

    if len(viewbox) != 4:
        raise ValueError("The SVG has an invalid viewBox.")

    try:
        x, y, width, height = map(float, viewbox)
    except ValueError as exc:
        raise ValueError("The SVG viewBox contains invalid values.") from exc

    if width <= 0 or height <= 0:
        raise ValueError("The SVG canvas dimensions are invalid.")

    if not any(
        local_name(element.tag) == "path"
        and element.attrib.get("d", "").strip()
        for element in root.iter()
    ):
        raise ValueError("The SVG contains no vector paths.")


# ============================================================
# MAIN CONVERTER: VTRACER FIRST, OPENCV SECOND
# ============================================================

def make_svg(
    image_bytes: bytes,
    mode: str,
    threshold: int,
):
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "I couldn't read that image. Send PNG, JPG, WEBP, BMP or TIFF."
        ) from exc

    rgba = np.array(image)
    alpha = rgba[:, :, 3]
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)

    mask, used_threshold = threshold_mask(gray, mode, threshold)
    mask[alpha < 10] = 0
    cleaned = remove_tiny_components(mask)

    if not np.any(cleaned):
        raise ValueError(
            "No dark artwork was detected. Try /threshold 190 "
            "or send a higher-contrast image."
        )

    height, width = cleaned.shape

    # --------------------------------------------------------
    # ENGINE 1: VTRACER
    # --------------------------------------------------------

    try:
        # VTracer traces a full-size image, not a cropped bitmap.
        bw = np.full((height, width, 3), 255, dtype=np.uint8)
        bw[cleaned > 0] = (0, 0, 0)

        with tempfile.TemporaryDirectory(prefix="modex_vtracer_") as tmp:
            input_path = Path(tmp) / "input.png"
            output_path = Path(tmp) / "output.svg"

            Image.fromarray(bw, mode="RGB").save(input_path, format="PNG")

            vtracer.convert_image_to_svg_py(
                str(input_path),
                str(output_path),
                colormode="binary",
                hierarchical="stacked",
                mode="spline",
                filter_speckle=2,
                color_precision=6,
                layer_difference=16,
                corner_threshold=60,
                length_threshold=3.5,
                max_iterations=10,
                splice_threshold=45,
                path_precision=3,
            )

            root = ET.parse(output_path).getroot()

        svg_bytes, path_count = sanitize_vtracer_svg(
            root,
            width,
            height,
        )

        # If output XML, canvas, or path validation fails, OpenCV
        # is attempted automatically by the exception handler below.
        engine = "VTracer spline"

    except Exception as exc:
        logger.warning(
            "VTracer output failed validation; using OpenCV fallback: %s",
            exc,
        )

        svg_bytes, width, height, path_count = opencv_fallback_svg(
            cleaned
        )
        engine = "OpenCV fallback"

    return svg_bytes, {
        "width": width,
        "height": height,
        "contours": path_count,
        "threshold": used_threshold,
        "mode": mode,
        "engine": engine,
    }


# ============================================================
# TELEGRAM MENUS
# ============================================================

HELP_TEXT = (
    "🖤 *MODEX B&W → SVG*\n\n"
    "Send a logo or black-and-white image to trace it into SVG paths.\n\n"
    "*Commands*\n"
    "/auto — automatic threshold\n"
    "/threshold 160 — manual threshold (1–254)\n"
    "/status — view current settings\n"
    "/help — show this help\n\n"
    "*Features*\n"
    "• Black foreground (#000000)\n"
    "• Transparent background\n"
    "• Sharp logo corners\n"
    "• VTracer first, OpenCV fallback\n"
    "• SVG cleanup for common Canva compatibility issues\n\n"
    "Send a high-resolution PNG as a document for best results."
)


def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🖼 Convert an image",
                callback_data="menu_convert",
            )
        ],
        [
            InlineKeyboardButton(
                "⚙️ Automatic mode",
                callback_data="menu_auto",
            ),
            InlineKeyboardButton(
                "🎚 Threshold settings",
                callback_data="menu_threshold",
            ),
        ],
        [
            InlineKeyboardButton(
                "ℹ️ Help",
                callback_data="menu_help",
            ),
            InlineKeyboardButton(
                "📊 My settings",
                callback_data="menu_status",
            ),
        ],
    ])


async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("start", "Open the main menu"),
        BotCommand("auto", "Automatic black and white detection"),
        BotCommand("threshold", "Set or view threshold"),
        BotCommand("status", "View current settings"),
        BotCommand("help", "Show instructions"),
    ])

    await application.bot.set_chat_menu_button(
        menu_button=MenuButtonCommands()
    )


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    get_settings(update.effective_user.id)

    await update.message.reply_text(
        "🖤 *MODEX B&W → SVG*\n\n"
        "Send a logo or black-and-white image. I'll return black vector "
        "paths on a transparent background.\n\n"
        "VTracer is tried first; OpenCV is the automatic fallback.\n\n"
        "Choose an option below or tap Telegram's menu button.",
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        HELP_TEXT,
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _, threshold = get_settings(user_id)
    set_settings(user_id, "auto", threshold)

    await update.message.reply_text(
        "Automatic threshold enabled. Send an image to convert it."
    )


async def threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not context.args:
        mode, threshold = get_settings(user_id)
        await update.message.reply_text(
            f"Current mode: {mode}\nManual threshold: {threshold}\n\n"
            "Usage: /threshold 160 (choose 1–254)."
        )
        return

    try:
        value = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Use a number, e.g. /threshold 160.")
        return

    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        await update.message.reply_text("Choose a threshold between 1 and 254.")
        return

    set_settings(user_id, "manual", value)
    await update.message.reply_text(
        f"Manual threshold set to {value}. Send an image to convert it. "
        "Use /auto to return to automatic thresholding."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode, threshold = get_settings(update.effective_user.id)
    shown = "Otsu (automatic per image)" if mode == "auto" else str(threshold)

    await update.message.reply_text(
        f"Mode: {mode}\nThreshold: {shown}\n"
        "Output: black vector paths, transparent background."
    )


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    action = query.data

    if action == "menu_convert":
        await query.message.reply_text(
            "Send a logo as a photo or image document. "
            "A PNG sent as a document usually preserves more detail."
        )

    elif action == "menu_auto":
        _, threshold = get_settings(user_id)
        set_settings(user_id, "auto", threshold)
        await query.message.reply_text("Automatic threshold enabled.")

    elif action == "menu_threshold":
        mode, threshold = get_settings(user_id)
        await query.message.reply_text(
            f"Current mode: {mode}\nManual threshold: {threshold}\n"
            "Set a value with /threshold 160."
        )

    elif action == "menu_status":
        mode, threshold = get_settings(user_id)
        await query.message.reply_text(
            f"Mode: {mode}\nThreshold: {threshold}"
        )

    elif action == "menu_help":
        await query.message.reply_text(
            HELP_TEXT,
            parse_mode="Markdown",
            reply_markup=main_menu_keyboard(),
        )


# ============================================================
# IMAGE DOWNLOAD
# ============================================================

async def get_image_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    if message.photo:
        file_obj = await context.bot.get_file(message.photo[-1].file_id)
        filename = "image.jpg"

    elif message.document:
        doc = message.document
        mime = (doc.mime_type or "").lower()
        name = doc.file_name or "image"

        extensions = (
            ".png", ".jpg", ".jpeg", ".webp",
            ".bmp", ".tif", ".tiff",
        )

        if not (
            mime.startswith("image/")
            or name.lower().endswith(extensions)
        ):
            raise ValueError("Please send PNG, JPG, WEBP, BMP or TIFF.")

        if doc.file_size and doc.file_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("Please send an image under 20 MB.")

        file_obj = await context.bot.get_file(doc.file_id)
        filename = name

    else:
        raise ValueError("Please send an image as a photo or image document.")

    downloaded = await file_obj.download_as_bytearray()

    if len(downloaded) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Please send an image under 20 MB.")

    return bytes(downloaded), filename


# ============================================================
# IMAGE HANDLER
# ============================================================

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    try:
        image_bytes, original_name = await get_image_bytes(update, context)
        mode, threshold = get_settings(update.effective_user.id)

        await message.reply_text(
            "Tracing with VTracer… I'll use OpenCV automatically if needed."
        )

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.UPLOAD_DOCUMENT,
        )

        svg_bytes, info = make_svg(image_bytes, mode, threshold)

        # Final validation immediately before sending the file.
        validate_svg(svg_bytes)

        stem = Path(original_name).stem or "vector"
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:48]

        output = io.BytesIO(svg_bytes)
        output.name = f"{safe_stem}_black_transparent.svg"

        caption = (
            f"✅ SVG ready\n"
            f"Canvas: {info['width']} × {info['height']} px\n"
            f"Threshold: {info['threshold']} ({info['mode']})\n"
            f"Vector paths: {info['contours']}\n"
            f"Engine: {info['engine']}\n"
            "Foreground: #000000 · Background: transparent"
        )

        await message.reply_document(
            document=output,
            filename=output.name,
            caption=caption,
        )

    except ValueError as exc:
        await message.reply_text(str(exc))

    except Exception:
        logger.exception("Image conversion failed")
        await message.reply_text(
            "Conversion failed. Try a high-resolution PNG with strong contrast."
        )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Unknown command. Use /help to see commands.")


# ============================================================
# START BOT
# ============================================================

def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN environment variable. "
            "Add your bot token in Railway Variables."
        )

    init_db()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        CallbackQueryHandler(menu_callback, pattern=r"^menu_")
    )
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("auto", auto_command))
    application.add_handler(CommandHandler("threshold", threshold_command))
    application.add_handler(CommandHandler("status", status_command))

    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler)
    )
    application.add_handler(
        MessageHandler(filters.COMMAND, unknown_command)
    )

    logger.info("Starting MODEX B&W to SVG bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
