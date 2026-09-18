"""Play the trained model against a website's own "vs computer" bots.

Unlike lichess_bot.py (which uses Lichess's official Bot API), this module
drives a real browser with Playwright and reads/writes the board through the
page's own DOM -- no API, no account needed. It works against whatever site
has a `SiteAdapter` written for it; two are included (chess.com, lichess.org).

This is inherently more fragile than an API integration: adapters depend on
each site's current front-end markup, which can change without notice. Every
selector below was verified against the live sites at the time this was
written; if a site redeploys its front end, the matching adapter may need
its selectors updated (see each adapter's class docstring for what to check).

Headed mode is required: both chessground (lichess) and chess.com's board
component gate drag interactions behind requestAnimationFrame, which
browsers throttle to near-zero in headless tabs, so drags silently never
complete. `--headless` is offered but will not work reliably.

Fair play note: only use this against a site's own bot/computer opponents,
not in real-money or ranked matches against human players -- most sites'
terms of service prohibit engine assistance there, and unlike lichess_bot.py
this account is not marked as a bot.

Usage:
    python src/browser_bot.py --model logs/checkpoints/ppo_chess_final.zip \
        --site lichess --difficulty 3 --color white
    python src/browser_bot.py --model logs/checkpoints/ppo_chess_final.zip \
        --site chess.com --difficulty beginner
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import time
from abc import ABC, abstractmethod
from typing import ClassVar

import chess
from playwright.sync_api import Page, sync_playwright
from sb3_contrib import MaskablePPO

from env import action_to_move, board_to_tensor, compute_action_mask

_SAN_RE = re.compile(r"^(O-O(-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?)[+#]?$")


def _looks_like_san(text: str) -> bool:
    """True if ``text`` is shaped like a single SAN move token.

    Used to filter a site's move-list DOM nodes down to actual moves,
    excluding move-number labels ("1.") and, critically, end-of-game status
    banners ("0-1 Checkmate * Black is victorious") that some sites render
    as a sibling inside the same move-list container -- without this check
    that banner text gets handed to ``board.parse_san()`` and crashes.
    """
    return bool(_SAN_RE.match(text))


def _drag(page: Page, src: tuple[float, float], dst: tuple[float, float], steps: int = 12) -> None:
    """Simulate a real, incremental mouse drag -- required for chessground/wc-chess-board
    to register the move (a plain click-and-teleport is ignored by both)."""
    sx, sy = src
    dx, dy = dst
    page.mouse.move(sx, sy)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(sx + (dx - sx) * i / steps, sy + (dy - sy) * i / steps)
        time.sleep(0.02)
    page.mouse.up()


class SiteAdapter(ABC):
    """One adapter per website: how to start a bot game, read the opponent's
    replies, and play our own moves. Board legality/turn-tracking/game-over
    is handled locally with python-chess -- the adapter only needs to expose
    the opponent's moves and a way to physically drag a move on screen."""

    name: str

    @abstractmethod
    def start_game(self, page: Page, difficulty: str, color: chess.Color) -> None:
        """Navigate to a fresh game against the bot at ``difficulty``, playing ``color``."""

    @abstractmethod
    def board_pixel_box(self, page: Page) -> dict[str, float]:
        """The rendered board's bounding box: {x, y, width, height}."""

    @abstractmethod
    def ply_count(self, page: Page) -> int:
        """How many half-moves the site's move list currently shows."""

    @abstractmethod
    def last_move_san(self, page: Page) -> str | None:
        """The most recent move in Standard Algebraic Notation, or None."""

    def square_to_pixel(
        self, page: Page, square: chess.Square, our_color: chess.Color
    ) -> tuple[float, float]:
        """Pixel center of ``square``, accounting for board orientation."""
        box = self.board_pixel_box(page)
        sq_size = box["width"] / 8
        file_idx = chess.square_file(square)
        rank = chess.square_rank(square) + 1
        if our_color == chess.WHITE:
            col, row_from_top = file_idx, 8 - rank
        else:
            col, row_from_top = 7 - file_idx, rank - 1
        return box["x"] + sq_size * (col + 0.5), box["y"] + sq_size * (row_from_top + 0.5)

    def make_move(self, page: Page, move: chess.Move, our_color: chess.Color) -> None:
        src = self.square_to_pixel(page, move.from_square, our_color)
        dst = self.square_to_pixel(page, move.to_square, our_color)
        _drag(page, src, dst)
        if move.promotion is not None:
            # Best-effort: both sites pop a promotion picker with Queen as the
            # first/topmost choice at the destination square.
            time.sleep(0.4)
            page.mouse.click(*dst)


class LichessAdapter(SiteAdapter):
    """lichess.org, "Play with the computer".

    Verified against the live site's chessground-based board: pieces are
    `<piece class="{color} {type}" style="top:X%;left:Y%">` inside `cg-board`
    (percentages, 0/12.5/.../87.5 per axis). The move list lives in
    `.round__app app`, alternating move-number and move-text children; we
    identify move-text children by NOT matching a bare number, since those
    custom element tag names (e.g. `<z7yx>`) are build-hashed and may change
    on redeploy -- if this adapter stops finding moves, re-verify that
    filter still selects the right children rather than assuming the tag.
    """

    name = "lichess"

    def start_game(self, page: Page, difficulty: str, color: chess.Color) -> None:
        page.goto("https://lichess.org/", wait_until="networkidle", timeout=30000)
        page.click("button.lobby__start__button--ai", timeout=10000)
        page.wait_for_selector(".game-setup", timeout=10000)
        level = str(max(1, min(8, int(difficulty))))
        page.click(f"label[for=sf_level_{level}]")
        color_id = "color-picker-white" if color == chess.WHITE else "color-picker-black"
        page.click(f"label[for={color_id}]")
        page.click(".footer button.lobby__start__button--ai")
        page.wait_for_selector("cg-board piece", timeout=15000)
        time.sleep(1.5)

    def board_pixel_box(self, page: Page) -> dict[str, float]:
        box = page.query_selector("cg-board").bounding_box()
        return {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}

    def _move_nodes(self, page: Page) -> list[str]:
        texts = page.eval_on_selector_all(
            ".round__app app > *",
            "els => els.map(e => e.textContent.trim())",
        )
        return [t for t in texts if _looks_like_san(t)]

    def ply_count(self, page: Page) -> int:
        return len(self._move_nodes(page))

    def last_move_san(self, page: Page) -> str | None:
        moves = self._move_nodes(page)
        return moves[-1] if moves else None


class ChessComAdapter(SiteAdapter):
    """chess.com, "Play Bots".

    Verified against the live site's board web component
    `<wc-chess-board id="board-play-computer">`, whose pieces are
    `<div class="piece {2-letter code} square-{file}{rank}">` (file/rank
    1-8, no orientation math needed to read them -- only to drag). The move
    list is `wc-simple-move-list`, with `.white-move`/`.black-move` nodes in
    order. Difficulty-category selection (Beginner/Intermediate/...) and bot
    picking is best-effort: if any step fails (e.g. a bot is login-locked),
    play proceeds with whatever bot chess.com had pre-selected rather than
    raising.
    """

    name = "chess.com"

    _CATEGORY_NAMES: ClassVar[dict[str, str]] = {
        "new": "New to Chess",
        "beginner": "Beginner",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
        "master": "Master",
        "adaptive": "Adaptive",
    }

    def start_game(self, page: Page, difficulty: str, color: chess.Color) -> None:
        if color == chess.BLACK:
            print(
                "[browser_bot] chess.com color selection isn't implemented; "
                "playing whatever color chess.com assigns (usually White).",
                file=sys.stderr,
            )
        page.goto("https://www.chess.com/play/computer", wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        with contextlib.suppress(Exception):
            page.click("text=Start", timeout=5000)
        time.sleep(1)

        category = self._CATEGORY_NAMES.get(difficulty.lower())
        if category:
            try:
                page.locator(f"div.bot-group-accordion-component:has-text('{category}')").locator(
                    ".bot-group-accordion-toggleClickArea"
                ).click(force=True, timeout=5000)
                time.sleep(0.5)
                avatar = page.locator(
                    f"div.bot-group-accordion-component:has-text('{category}') "
                    "[class*=bot-group-accordion-bots] img"
                ).first
                avatar.click(force=True, timeout=5000)
                time.sleep(0.5)
            except Exception as exc:
                print(
                    f"[browser_bot] difficulty selection failed ({exc}); using the default bot.",
                    file=sys.stderr,
                )

        page.click("button.bot-selection-cta-button-button", timeout=10000)
        page.wait_for_selector("#board-play-computer piece", timeout=15000)
        time.sleep(1.5)

    def board_pixel_box(self, page: Page) -> dict[str, float]:
        box = page.query_selector("#board-play-computer").bounding_box()
        return {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}

    def _move_texts(self, page: Page) -> list[str]:
        texts = page.eval_on_selector_all(
            "wc-simple-move-list .white-move, wc-simple-move-list .black-move",
            "els => els.map(e => e.textContent.trim())",
        )
        return [t for t in texts if _looks_like_san(t)]

    def ply_count(self, page: Page) -> int:
        return len(self._move_texts(page))

    def last_move_san(self, page: Page) -> str | None:
        moves = self._move_texts(page)
        return moves[-1] if moves else None


ADAPTERS: dict[str, type[SiteAdapter]] = {
    "lichess": LichessAdapter,
    "chess.com": ChessComAdapter,
}


def play_game(
    adapter: SiteAdapter,
    page: Page,
    model: MaskablePPO,
    our_color: chess.Color,
    delay: float,
    opponent_timeout: float = 120.0,
) -> str:
    """Play one full game, alternating our model's moves with the site's bot.

    Board legality and game-over detection use our own local `chess.Board`,
    kept in sync by pushing our own moves immediately and the opponent's
    moves as soon as the site's move list shows a new one. Ends on a
    claimable draw (threefold repetition, 50-move rule) even if neither side
    would otherwise stop -- an under-trained model can easily shuffle
    between two "safe" moves forever without this.
    """
    board = chess.Board()
    known_ply = 0

    while not board.is_game_over(claim_draw=True):
        if board.turn == our_color:
            obs = board_to_tensor(board)
            action_masks = compute_action_mask(board)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            move = action_to_move(int(action), board.turn)
            adapter.make_move(page, move, our_color)
            board.push(move)
            known_ply += 1
        else:
            deadline = time.time() + opponent_timeout
            while adapter.ply_count(page) <= known_ply:
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for the opponent's move.")
                time.sleep(0.3)
            san = adapter.last_move_san(page)
            move = board.parse_san(san)
            board.push(move)
            known_ply += 1
        time.sleep(delay)

    return board.result(claim_draw=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play the trained model against a website's own bots.")
    parser.add_argument("--model", type=str, required=True, help="Path to a MaskablePPO .zip checkpoint.")
    parser.add_argument("--site", choices=sorted(ADAPTERS), required=True)
    parser.add_argument(
        "--difficulty",
        type=str,
        default="1",
        help="lichess: 1-8. chess.com: new|beginner|intermediate|advanced|master|adaptive.",
    )
    parser.add_argument("--color", choices=["white", "black"], default="white")
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.6, help="Seconds to pause after each of our moves.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Not recommended: both sites' boards silently ignore drags in headless mode.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model = MaskablePPO.load(args.model)
    our_color = chess.WHITE if args.color == "white" else chess.BLACK
    adapter = ADAPTERS[args.site]()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1400, "height": 1000}, locale="en-US")

        results: list[str] = []
        try:
            for game in range(1, args.num_games + 1):
                print(
                    f"=== Game {game}/{args.num_games} vs {adapter.name} (difficulty={args.difficulty}) ==="
                )
                adapter.start_game(page, args.difficulty, our_color)
                result = play_game(adapter, page, model, our_color, args.delay)
                results.append(result)
                print(f"Game {game} result: {result}")
        finally:
            browser.close()

    print(f"Final results: {results}")


if __name__ == "__main__":
    main()
