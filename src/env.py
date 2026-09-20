"""Gymnasium-compatible chess environment with a Stockfish opponent.

The board is encoded from the perspective of the side to move (channels
0-5 are always "my" pieces, 6-11 are always the opponent's), using
``chess.Board.mirror()`` to canonicalize Black-to-move positions. Actions
are encoded in that same canonical square space, so a single policy
network can play either color without knowing which one it is.
"""

from __future__ import annotations

import contextlib
import random
from typing import Any, ClassVar

import chess
import chess.engine
import gymnasium as gym
import numpy as np
from gymnasium import spaces

PIECE_TYPES: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
PROMOTION_PIECES: tuple[chess.PieceType | None, ...] = (
    None,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)
NUM_PROMOTIONS = len(PROMOTION_PIECES)
ACTION_SPACE_SIZE = 64 * 64 * NUM_PROMOTIONS

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _canonical_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    """Map a real board square into (or out of) the side-to-move's canonical frame.

    ``chess.square_mirror`` is its own inverse, so this same function
    converts real -> canonical and canonical -> real.
    """
    return square if turn == chess.WHITE else chess.square_mirror(square)


def move_to_action(move: chess.Move, turn: chess.Color) -> int:
    """Encode a legal move (in real board coordinates) as a discrete action index."""
    from_sq = _canonical_square(move.from_square, turn)
    to_sq = _canonical_square(move.to_square, turn)
    promo_idx = PROMOTION_PIECES.index(move.promotion)
    return (from_sq * 64 + to_sq) * NUM_PROMOTIONS + promo_idx


def action_to_move(action: int, turn: chess.Color) -> chess.Move:
    """Decode a discrete action index back into a real-board ``chess.Move``.

    The decoded move is not guaranteed to be legal; callers must check it
    against ``board.legal_moves`` (or rely on action masking).
    """
    promo_idx = action % NUM_PROMOTIONS
    remainder = action // NUM_PROMOTIONS
    canonical_to = remainder % 64
    canonical_from = remainder // 64
    from_sq = _canonical_square(canonical_from, turn)
    to_sq = _canonical_square(canonical_to, turn)
    promotion = PROMOTION_PIECES[promo_idx]
    return chess.Move(from_sq, to_sq, promotion=promotion)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a board as a (12, 8, 8) float32 tensor from the side-to-move's view.

    Channels 0-5 hold the side-to-move's pieces (pawn..king), channels 6-11
    hold the opponent's, each as an 8x8 one-hot occupancy plane.
    """
    canonical = board if board.turn == chess.WHITE else board.mirror()
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for piece_idx, piece_type in enumerate(PIECE_TYPES):
        for square in canonical.pieces(piece_type, chess.WHITE):
            row, col = 7 - chess.square_rank(square), chess.square_file(square)
            tensor[piece_idx, row, col] = 1.0
        for square in canonical.pieces(piece_type, chess.BLACK):
            row, col = 7 - chess.square_rank(square), chess.square_file(square)
            tensor[6 + piece_idx, row, col] = 1.0
    return tensor


def compute_action_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean mask over the discrete action space of currently legal moves."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_action(move, board.turn)] = True
    return mask


def material_balance(board: chess.Board, color: chess.Color) -> int:
    """Return ``color``'s material total minus the opponent's, in pawn units."""
    balance = 0
    for piece_type, value in PIECE_VALUES.items():
        balance += value * len(board.pieces(piece_type, color))
        balance -= value * len(board.pieces(piece_type, not color))
    return balance


_DEPTH_RAMP: tuple[int, ...] = (1, 2, 4)  # ply depth for curriculum levels 1..len(_DEPTH_RAMP)


def stockfish_level_to_skill_level(stockfish_level: int) -> int:
    """Convert a literal Stockfish Skill Level (0-20) to ``ChessEnv``'s
    ``skill_level`` scale, which reserves the low end for the random-move
    bootstrap and depth-ramp stages (see :meth:`ChessEnv.set_skill_level`)."""
    return stockfish_level + len(_DEPTH_RAMP) + 1


class ChessEnv(gym.Env):
    """A single-agent chess environment played against a Stockfish opponent.

    Each episode, the agent is randomly assigned White or Black and plays
    to completion against an opponent at ``skill_level``: 0 is a
    uniform-random mover; 1-3 are full-strength Stockfish limited to 1/2/4
    ply of lookahead (a depth ramp bridging the large gap between random
    play and a real engine's full search); 4-24 map to Stockfish Skill Level
    0-20. The observation is a canonical (12, 8, 8) board tensor and the
    action space is a ``Discrete(64 * 64 * 5)`` encoding of (from, to,
    promotion) in that same canonical frame; use :meth:`action_masks` for
    action masking.

    Reward is win/loss/draw (+-1/0) at the end, plus two optional per-move
    shaping terms: ``material_reward_scale`` (change in material balance)
    and ``repetition_penalty`` (a flat penalty the moment a position recurs
    for the second time, well before it would become a claimable draw --
    meant to discourage an under-trained policy from settling into a
    back-and-forth habit instead of making progress).
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human"]}

    def __init__(
        self,
        stockfish_path: str,
        skill_level: int = 0,
        material_reward_scale: float = 0.0,
        repetition_penalty: float = 0.0,
        engine_think_time: float = 0.1,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.stockfish_path = stockfish_path
        self.skill_level = max(0, min(len(_DEPTH_RAMP) + 21, skill_level))
        self.material_reward_scale = material_reward_scale
        self.repetition_penalty = repetition_penalty
        self.engine_think_time = engine_think_time
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12, 8, 8), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.board = chess.Board()
        self.agent_color: chess.Color = chess.WHITE
        self._engine: chess.engine.SimpleEngine | None = None
        self._configured_skill: int | None = None
        self._prev_material = 0

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        if self.skill_level <= len(_DEPTH_RAMP):
            # Depth-ramp stages: full-strength move selection, but starved
            # of lookahead, rather than Stockfish's own "Skill Level" noise.
            stockfish_skill = 20
        else:
            stockfish_skill = self.skill_level - len(_DEPTH_RAMP) - 1
        if self._configured_skill != stockfish_skill:
            self._engine.configure({"Skill Level": stockfish_skill})
            self._configured_skill = stockfish_skill
        return self._engine

    def _opponent_limit(self) -> chess.engine.Limit:
        if self.skill_level <= len(_DEPTH_RAMP):
            return chess.engine.Limit(depth=_DEPTH_RAMP[self.skill_level - 1])
        return chess.engine.Limit(time=self.engine_think_time)

    def set_skill_level(self, skill_level: int) -> None:
        """Update the opponent's difficulty; applied on the next move.

        0 is a uniform-random mover. 1-3 are a depth-ramp of full-strength
        Stockfish limited to 1/2/4 ply of lookahead (a bridge between random
        play and full search -- a real engine even at "Skill Level 0" is
        otherwise a big jump up from a random mover). 4-24 map to Stockfish
        Skill Level 0-20.
        """
        self.skill_level = max(0, min(len(_DEPTH_RAMP) + 21, skill_level))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board = chess.Board()
        self.agent_color = random.choice([chess.WHITE, chess.BLACK])
        self._prev_material = material_balance(self.board, self.agent_color)
        if self.board.turn != self.agent_color:
            self._play_opponent_move()
        return board_to_tensor(self.board), {"agent_color": self.agent_color}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        move = action_to_move(int(action), self.board.turn)
        if move not in self.board.legal_moves:
            return board_to_tensor(self.board), -1.0, True, False, {"illegal_move": True}
        self.board.push(move)

        # claim_draw=True so a repetitive ("shuffling") policy ends the
        # episode as soon as a draw is claimable, instead of playing on
        # until the much-later automatic fivefold-repetition/75-move cutoff.
        terminated = self.board.is_game_over(claim_draw=True)
        if not terminated:
            self._play_opponent_move()
            terminated = self.board.is_game_over(claim_draw=True)

        reward = self._terminal_reward() if terminated else self._shaping_reward()
        info: dict[str, Any] = {"agent_color": self.agent_color}
        if terminated:
            info["result"] = self.board.result(claim_draw=True)
        return board_to_tensor(self.board), reward, terminated, False, info

    def _play_opponent_move(self) -> None:
        if self.skill_level == 0:
            move = random.choice(list(self.board.legal_moves))
            self.board.push(move)
            return
        for _attempt in range(2):
            try:
                engine = self._ensure_engine()
                result = engine.play(self.board, self._opponent_limit())
                if result.move is not None:
                    self.board.push(result.move)
                return
            except (chess.engine.EngineError, TimeoutError, OSError):
                # A hung/crashed Stockfish process (observed after long
                # unattended runs) would otherwise deadlock the whole
                # training run: SubprocVecEnv waits forever on a worker that
                # never replies. Restart the engine and retry once.
                self._restart_engine()
        # Still failing after a retry: don't crash a multi-hour run over one
        # move, just have the opponent pass on a random legal move.
        move = random.choice(list(self.board.legal_moves))
        self.board.push(move)

    def _restart_engine(self) -> None:
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.quit()
            self._engine = None
            self._configured_skill = None

    def _terminal_reward(self) -> float:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == self.agent_color else -1.0

    def _shaping_reward(self) -> float:
        reward = 0.0
        if self.material_reward_scale != 0.0:
            material = material_balance(self.board, self.agent_color)
            reward += self.material_reward_scale * (material - self._prev_material)
            self._prev_material = material
        if self.repetition_penalty != 0.0 and self.board.is_repetition(2):
            reward -= self.repetition_penalty
        return reward

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the action space; required by sb3-contrib's MaskablePPO."""
        return compute_action_mask(self.board)

    def render(self) -> str | None:
        if self.render_mode == "human":
            print(self.board.unicode(borders=True))
            return None
        return self.board.unicode(borders=True)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
