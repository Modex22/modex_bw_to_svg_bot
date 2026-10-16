
import io
import os
import sqlite3
import logging
from pathlib import Path

import cv2
import numpy as np
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

# Lower values preserve smaller logo details.
MIN_COMPONENT_AREA = 3.0

# Smaller epsilon preserves sharp corners and jagged details.
SIMPLIFY_EPSILON_RATIO = 0.0005
MIN_SIMPLIFY_EPSILON = 0.1

# Small transparent margin around the final artwork.
PADDING = 2

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("modex-bw-to-svg")


# ============================================================
# DATABASE AND USER SETTINGS
# ============================================================

def db_connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                mode TEXT NOT NULL DEFAULT 'auto',
                threshold INTEGER NOT NULL DEFAULT 160
            )
            """
        )


def get_settings(user_id: int) -> tuple[str, int]:
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT mode, threshold
            FROM user_settings
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        if row:
            return row[0], int(row[1])

        conn.execute(
            """
            INSERT OR IGNORE INTO user_settings
            (user_id, mode, threshold)
            VALUES (?, 'auto', ?)
            """,
            (user_id, DEFAULT_THRESHOLD),
        )

        return "auto", DEFAULT_THRESHOLD


def set_settings(
    user_id: int,
    mode: str,
    threshold: int | None = None,
):
    _, current_threshold = get_settings(user_id)

    if threshold is None:
        threshold = current_threshold

    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO user_settings
            (user_id, mode, threshold)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                mode = excluded.mode,
                threshold = excluded.threshold
            """,
            (user_id, mode, int(threshold)),
        )


# ============================================================
# IMAGE THRESHOLDING
# ============================================================

def threshold_mask(
    gray: np.ndarray,
    mode: str,
    threshold: int,
):
    """
    Return a binary mask.

    White pixels in the mask represent the foreground artwork.
    Dark image pixels become foreground vector shapes.
    """

    if mode == "auto":
        used_threshold, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )

        return mask, int(used_threshold)

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    return mask, int(threshold)


# ============================================================
# SVG VECTOR CONVERSION
# ============================================================

def make_svg(
    image_bytes: bytes,
    mode: str,
    threshold: int,
):
    """
    Convert a raster image into genuine SVG vector paths.

    Features:
    - Black foreground (#000000)
    - Transparent background
    - Preserves enclosed cutouts
    - Sharp corners and angular edges
    - Conservative speck removal
    - No embedded raster image
    """

    # --------------------------------------------------------
    # 1. Read and normalize the source image
    # --------------------------------------------------------

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "I couldn't read that image. Please send a PNG, JPG, "
            "WEBP, BMP or TIFF."
        ) from exc

    rgba = np.array(image)

    alpha = rgba[:, :, 3]

    gray = cv2.cvtColor(
        rgba[:, :, :3],
        cv2.COLOR_RGB2GRAY,
    )

    # --------------------------------------------------------
    # 2. Create the foreground mask
    # --------------------------------------------------------

    mask, used_threshold = threshold_mask(
        gray,
        mode,
        threshold,
    )

    # Ignore pixels that were already transparent in the source.
    mask[alpha < 10] = 0

    # IMPORTANT:
    # No Gaussian blur, morphological opening, or closing.
    # These operations can round corners, erase thin cuts,
    # and change small details in logos and lettering.

    # CHAIN_APPROX_NONE keeps the original contour points
    # before the controlled simplification step.
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError(
            "No dark artwork was detected. Try /threshold 190 "
            "or send a higher-contrast image."
        )

    # --------------------------------------------------------
    # 3. Filter tiny isolated components and trace contours
    # --------------------------------------------------------

    kept = []

    # RETR_TREE returns contours with a parent/child hierarchy.
    # Keep this information so small intentional cutouts inside
    # letters are not mistaken for stray specks.
    hierarchy_data = hierarchy[0] if hierarchy is not None else None

    for index, contour in enumerate(contours):
        area = abs(float(cv2.contourArea(contour)))

        # Ignore tiny isolated exterior components.
        # Do not automatically discard small interior contours,
        # because these may be holes inside letters or logo shapes.
        parent_index = -1

        if hierarchy_data is not None:
            parent_index = int(hierarchy_data[index][3])

        if area < MIN_COMPONENT_AREA and parent_index == -1:
            continue

        perimeter = cv2.arcLength(contour, True)

        epsilon = max(
            MIN_SIMPLIFY_EPSILON,
            perimeter * SIMPLIFY_EPSILON_RATIO,
        )

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        points = approx.reshape(-1, 2)

        if len(points) >= 3:
            kept.append(points)

    if not kept:
        raise ValueError(
            "No usable vector shapes were detected. Try a higher "
            "threshold with /threshold 190."
        )

    # --------------------------------------------------------
    # 4. Calculate the artwork bounds
    # --------------------------------------------------------

    all_points = np.vstack(kept)

    min_x = max(
        0,
        int(all_points[:, 0].min()) - PADDING,
    )

    min_y = max(
        0,
        int(all_points[:, 1].min()) - PADDING,
    )

    max_x = min(
        gray.shape[1] - 1,
        int(all_points[:, 0].max()) + PADDING,
    )

    max_y = min(
        gray.shape[0] - 1,
        int(all_points[:, 1].max()) + PADDING,
    )

    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)

    # --------------------------------------------------------
    # 5. Build true SVG paths
    # --------------------------------------------------------

    path_parts = []

    for points in kept:
        shifted = points.copy()

        shifted[:, 0] -= min_x
        shifted[:, 1] -= min_y

        coords = shifted.tolist()

        if len(coords) < 3:
            continue

        path_data = (
            f"M {coords[0][0]} {coords[0][1]} "
        )

        path_data += " ".join(
            f"L {x} {y}"
            for x, y in coords[1:]
        )

        path_data += " Z"

        path_parts.append(path_data)

    if not path_parts:
        raise ValueError(
            "The image did not contain a usable vector shape."
        )

    # No background rectangle is included.
    # The evenodd fill rule preserves enclosed transparent holes.
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}" height="{height}"
     viewBox="0 0 {width} {height}">
  <path d="{' '.join(path_parts)}"
        fill="#000000"
        fill-rule="evenodd"
        clip-rule="evenodd"/>
</svg>
""".encode("utf-8")

    return svg, {
        "width": width,
        "height": height,
        "contours": len(path_parts),
        "threshold": used_threshold,
        "mode": mode,
    }


# ============================================================
# HELP AND MENUS
# ============================================================

HELP_TEXT = (
    "🖤 *MODEX B&W → SVG*\n\n"
    "Send a logo or black-and-white image and I'll trace it "
    "into real SVG vector paths.\n\n"
    "*Commands*\n"
    "/auto — automatic threshold detection\n"
    "/threshold 160 — set a manual threshold\n"
    "/status — view your current settings\n"
    "/help — show this help\n\n"
    "*Features*\n"
    "• Black foreground (#000000)\n"
    "• Transparent background\n"
    "• Sharp corners and angular logo details\n"
    "• Preserves enclosed cutouts\n"
    "• Real SVG paths, not an embedded bitmap\n\n"
    "*Tips*\n"
    "Use a high-resolution image with strong contrast. "
    "Use /threshold to adjust which gray pixels become black.\n\n"
    "This bot traces the supplied image; it does not redraw "
    "or reconstruct missing details."
)


def main_menu_keyboard():
    return InlineKeyboardMarkup(
        [
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
        ]
    )


async def post_init(application: Application):
    """Register the native Telegram menu beside the message bar."""

    await application.bot.set_my_commands(
        [
            BotCommand("start", "Open the main menu"),
            BotCommand("auto", "Automatic black and white detection"),
            BotCommand("threshold", "Set or view threshold"),
            BotCommand("status", "View your current settings"),
            BotCommand("help", "Show instructions"),
        ]
    )

    await application.bot.set_chat_menu_button(
        menu_button=MenuButtonCommands()
    )


# ============================================================
# MENU CALLBACKS
# ============================================================

async def menu_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    await query.answer()

    user_id = query.from_user.id
    action = query.data

    if action == "menu_convert":
        await query.message.reply_text(
            "Send me a logo or black-and-white image as a photo "
            "or image document. I'll return an SVG with black "
            "vector paths and a transparent background."
        )

    elif action == "menu_auto":
        _, threshold = get_settings(user_id)

        set_settings(
            user_id,
            "auto",
            threshold,
        )

        await query.message.reply_text(
            "Automatic threshold enabled. Send an image to "
            "convert it.\n\n"
            "Use /threshold 160 for manual thresholding."
        )

    elif action == "menu_threshold":
        mode, threshold = get_settings(user_id)

        await query.message.reply_text(
            f"Current mode: {mode}\n"
            f"Manual threshold: {threshold}\n\n"
            "Set a value with /threshold 160. Choose a value "
            "from 1 to 254.\n\n"
            "Higher values include more gray pixels."
        )

    elif action == "menu_status":
        mode, threshold = get_settings(user_id)

        shown = (
            "Otsu (automatic per image)"
            if mode == "auto"
            else str(threshold)
        )

        await query.message.reply_text(
            f"Mode: {mode}\n"
            f"Threshold: {shown}\n"
            "Output: black SVG paths, transparent background\n"
            "Tracing: sharp-corner logo mode"
        )

    elif action == "menu_help":
        await query.message.reply_text(
            HELP_TEXT,
            parse_mode="Markdown",
            reply_markup=main_menu_keyboard(),
        )


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    get_settings(update.effective_user.id)

    await update.message.reply_text(
        "🖤 *MODEX B&W → SVG*\n\n"
        "Welcome! Send a logo or black-and-white image and "
        "I'll trace it into real SVG paths.\n\n"
        "The foreground is black and the background is "
        "transparent. Sharp corners and angular details "
        "are preserved as closely as possible.\n\n"
        "Choose an option below, or tap the Telegram menu "
        "beside the message bar.",
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        HELP_TEXT,
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def auto_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id

    _, threshold = get_settings(user_id)

    set_settings(
        user_id,
        "auto",
        threshold,
    )

    await update.message.reply_text(
        "Automatic threshold enabled. Send an image to convert it.\n"
        "Use /threshold 160 to switch to a manual threshold."
    )


async def threshold_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id

    if not context.args:
        mode, threshold = get_settings(user_id)

        await update.message.reply_text(
            f"Current mode: {mode}\n"
            f"Manual threshold: {threshold}\n\n"
            "Usage: /threshold 160\n"
            "Choose a value from 1 to 254.\n\n"
            "Lower values keep only darker pixels; higher "
            "values include more gray pixels."
        )

        return

    try:
        value = int(context.args[0])

    except ValueError:
        await update.message.reply_text(
            "Please use a number, e.g. /threshold 160."
        )
        return

    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        await update.message.reply_text(
            "Choose a threshold between 1 and 254."
        )
        return

    set_settings(
        user_id,
        "manual",
        value,
    )

    await update.message.reply_text(
        f"Manual threshold set to {value}.\n"
        "Send an image to convert it.\n\n"
        "Use /auto to return to automatic thresholding."
    )


async def status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    mode, threshold = get_settings(
        update.effective_user.id
    )

    shown = (
        "Otsu (automatic per image)"
        if mode == "auto"
        else str(threshold)
    )

    await update.message.reply_text(
        f"Mode: {mode}\n"
        f"Threshold: {shown}\n"
        "Output: black #000000 SVG paths, transparent background\n"
        "Tracing: sharp-corner logo mode\n"
        f"Speck filter: exterior contours smaller than "
        f"{MIN_COMPONENT_AREA:g} px² are removed."
    )


# ============================================================
# IMAGE DOWNLOAD
# ============================================================

async def get_image_bytes(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message

    if message.photo:
        file_obj = await context.bot.get_file(
            message.photo[-1].file_id
        )

        filename = "image.jpg"

    elif message.document:
        doc = message.document

        mime = (doc.mime_type or "").lower()
        name = doc.file_name or "image"

        allowed_ext = (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
        )

        if not (
            mime.startswith("image/")
            or name.lower().endswith(allowed_ext)
        ):
            raise ValueError(
                "Please send an image file: PNG, JPG, WEBP, "
                "BMP or TIFF."
            )

        if (
            doc.file_size
            and doc.file_size > MAX_DOWNLOAD_BYTES
        ):
            raise ValueError(
                "That file is too large. Please send an image "
                "under 20 MB."
            )

        file_obj = await context.bot.get_file(
            doc.file_id
        )

        filename = name

    else:
        raise ValueError(
            "Please send an image as a photo or image document."
        )

    downloaded = await file_obj.download_as_bytearray()

    if len(downloaded) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            "That file is too large. Please send an image "
            "under 20 MB."
        )

    return bytes(downloaded), filename


# ============================================================
# IMAGE HANDLER
# ============================================================

async def image_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.message

    if not message:
        return

    try:
        image_bytes, original_name = await get_image_bytes(
            update,
            context,
        )

        mode, threshold = get_settings(
            update.effective_user.id
        )

        await message.reply_text(
            "Tracing your image into sharp-edged SVG paths…"
        )

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.UPLOAD_DOCUMENT,
        )

        svg_bytes, info = make_svg(
            image_bytes,
            mode,
            threshold,
        )

        stem = Path(original_name).stem or "vector"

        safe_stem = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in stem
        )[:48]

        output = io.BytesIO(svg_bytes)

        output.name = (
            f"{safe_stem}_black_transparent.svg"
        )

        caption = (
            f"✅ SVG ready\n"
            f"Canvas: {info['width']} × {info['height']} px\n"
            f"Threshold: {info['threshold']} ({info['mode']})\n"
            f"Vector contours: {info['contours']}\n"
            "Foreground: #000000\n"
            "Background: transparent\n"
            "Tracing: sharp-corner mode"
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
            "Something went wrong while converting that image. "
            "Try a high-resolution PNG or JPG with stronger "
            "black-and-white contrast."
        )


# ============================================================
# UNKNOWN COMMAND HANDLER
# ============================================================

async def unknown_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Unknown command. Use /help to see available commands."
    )


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

    application.add_handler(
        CommandHandler("start", start)
    )

    application.add_handler(
        CallbackQueryHandler(
            menu_callback,
            pattern=r"^menu_",
        )
    )

    application.add_handler(
        CommandHandler("help", help_command)
    )

    application.add_handler(
        CommandHandler("auto", auto_command)
    )

    application.add_handler(
        CommandHandler("threshold", threshold_command)
    )

    application.add_handler(
        CommandHandler("status", status_command)
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO | filters.Document.IMAGE,
            image_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.COMMAND,
            unknown_command,
        )
    )

    logger.info("Starting MODEX B&W to SVG bot")

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
