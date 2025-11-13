import copy
import random
from typing import List, Tuple, Optional, Dict

# Board representation:
# - 8x8 list of strings
# - '.' for empty
# - Uppercase letters for White pieces, lowercase for Black pieces
# - Pieces: K, Q, R, B, N, P

# Piece material values (very simple material evaluation)
PIECE_VALUES: Dict[str, int] = {
    'K': 1000, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,
    'k': -1000, 'q': -9, 'r': -5, 'b': -3, 'n': -3, 'p': -1,
}

# Initial board setup (same as other modules)
initial_board: List[List[str]] = [
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
]

Move = Tuple[int, int, int, int]


def evaluate_material(board: List[List[str]]) -> int:
    score: int = 0
    for row in board:
        for piece in row:
            if piece in PIECE_VALUES:
                score += PIECE_VALUES[piece]
    return score


def make_move(board: List[List[str]], move: Move) -> List[List[str]]:
    new_board = copy.deepcopy(board)
    row_from, col_from, row_to, col_to = move
    new_board[row_to][col_to] = new_board[row_from][col_from]
    new_board[row_from][col_from] = '.'
    return new_board


def generate_piece_moves(board: List[List[str]], row: int, col: int, piece: str) -> List[Move]:
    moves: List[Move] = []
    piece_type = piece.lower()
    is_white = piece.isupper()

    if piece_type == 'p':  # Pawn
        direction = -1 if is_white else 1
        start_row = 6 if is_white else 1
        # Forward move
        if 0 <= row + direction < 8 and board[row + direction][col] == '.':
            moves.append((row, col, row + direction, col))
            # Initial two-square move
            if row == start_row and board[row + 2 * direction][col] == '.':
                moves.append((row, col, row + 2 * direction, col))
        # Captures
        for col_offset in [-1, 1]:
            new_col = col + col_offset
            new_row = row + direction
            if 0 <= new_col < 8 and 0 <= new_row < 8:
                target = board[new_row][new_col]
                if target != '.' and target.isupper() != is_white:
                    moves.append((row, col, new_row, new_col))

    elif piece_type == 'n':  # Knight
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2),
        ]
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == '.' or target.isupper() != is_white:
                    moves.append((row, col, new_row, new_col))

    elif piece_type == 'r':  # Rook
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board[new_row][new_col]
                if target == '.':
                    moves.append((row, col, new_row, new_col))
                elif target.isupper() != is_white:
                    moves.append((row, col, new_row, new_col))
                    break
                else:
                    break

    elif piece_type == 'b':  # Bishop
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board[new_row][new_col]
                if target == '.':
                    moves.append((row, col, new_row, new_col))
                elif target.isupper() != is_white:
                    moves.append((row, col, new_row, new_col))
                    break
                else:
                    break

    elif piece_type == 'q':  # Queen
        for dr, dc in [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = board[new_row][new_col]
                if target == '.':
                    moves.append((row, col, new_row, new_col))
                elif target.isupper() != is_white:
                    moves.append((row, col, new_row, new_col))
                    break
                else:
                    break

    elif piece_type == 'k':  # King
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = board[new_row][new_col]
                    if target == '.' or target.isupper() != is_white:
                        moves.append((row, col, new_row, new_col))

    return moves


def generate_moves(board: List[List[str]], is_white_turn: bool) -> List[Move]:
    moves: List[Move] = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == '.':
                continue
            if (is_white_turn and piece.isupper()) or ((not is_white_turn) and piece.islower()):
                moves.extend(generate_piece_moves(board, row, col, piece))
    return moves


def board_has_king(board: List[List[str]], is_white_king: bool) -> bool:
    target = 'K' if is_white_king else 'k'
    for row in board:
        for cell in row:
            if cell == target:
                return True
    return False


class RLValueAgent:
    """Simple value-based RL agent with linear function approximation and TD(0).

    This agent is designed to play as White by default. It uses an epsilon-greedy
    policy over a one-step lookahead with random opponent reply sampling.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        agent_plays_white: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.agent_plays_white = agent_plays_white
        self.random = random.Random(random_seed)
        # 13 features: bias + 6 white counts + 6 black counts
        self.weights: List[float] = [0.0 for _ in range(13)]

    def featurize(self, board: List[List[str]]) -> List[float]:
        # Features: [bias, W_K, W_Q, W_R, W_B, W_N, W_P, B_k, B_q, B_r, B_b, B_n, B_p]
        features: List[float] = [1.0] + [0.0] * 12
        for row in board:
            for cell in row:
                if cell == '.':
                    continue
                # White pieces
                if cell == 'K':
                    features[1] += 1.0
                elif cell == 'Q':
                    features[2] += 1.0
                elif cell == 'R':
                    features[3] += 1.0
                elif cell == 'B':
                    features[4] += 1.0
                elif cell == 'N':
                    features[5] += 1.0
                elif cell == 'P':
                    features[6] += 1.0
                # Black pieces
                elif cell == 'k':
                    features[7] += 1.0
                elif cell == 'q':
                    features[8] += 1.0
                elif cell == 'r':
                    features[9] += 1.0
                elif cell == 'b':
                    features[10] += 1.0
                elif cell == 'n':
                    features[11] += 1.0
                elif cell == 'p':
                    features[12] += 1.0
        return features

    def value(self, board: List[List[str]]) -> float:
        features = self.featurize(board)
        return sum(w * x for w, x in zip(self.weights, features))

    def choose_move(self, board: List[List[str]]) -> Optional[Move]:
        is_white_turn = self.agent_plays_white
        legal_moves = generate_moves(board, is_white_turn)
        if not legal_moves:
            return None
        # Epsilon-greedy exploration
        if self.random.random() < self.epsilon:
            return self.random.choice(legal_moves)
        # Greedy: evaluate approximate value of next position after opponent reply
        best_move: Optional[Move] = None
        best_score: float = float('-inf') if is_white_turn else float('inf')
        for move in legal_moves:
            board_after_agent = make_move(board, move)
            # Sample one opponent random reply (if exists)
            opponent_moves = generate_moves(board_after_agent, not is_white_turn)
            if opponent_moves:
                reply = self.random.choice(opponent_moves)
                next_state = make_move(board_after_agent, reply)
            else:
                next_state = board_after_agent
            score = self.value(next_state)
            if is_white_turn:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        return best_move

    def learn_from_transition(
        self,
        state: List[List[str]],
        reward: float,
        next_state: Optional[List[List[str]]],
        done: bool,
    ) -> None:
        target_value = reward
        if not done and next_state is not None:
            target_value += self.discount_factor * self.value(next_state)
        prediction = self.value(state)
        td_error = target_value - prediction
        features = self.featurize(state)
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * td_error * features[i]


def play_episode(
    agent: RLValueAgent,
    max_steps: int = 200,
    opponent_random_seed: Optional[int] = None,
) -> float:
    """Play one episode with the agent vs a random opponent and learn online.

    Returns the cumulative reward obtained by the agent.
    """
    rng = random.Random(opponent_random_seed)
    state = copy.deepcopy(initial_board)
    is_white_turn = True  # Start from the standard position
    cumulative_reward: float = 0.0

    for _ in range(max_steps):
        if is_white_turn != agent.agent_plays_white:
            # Opponent (random) turn
            legal_opponent = generate_moves(state, is_white_turn)
            if not legal_opponent:
                # Terminal: no legal moves
                agent.learn_from_transition(state, 0.0, None, True)
                break
            state = make_move(state, rng.choice(legal_opponent))
            if not board_has_king(state, True) or not board_has_king(state, False):
                # One king captured => terminal
                agent.learn_from_transition(state, 0.0, None, True)
                break
            is_white_turn = not is_white_turn
            continue

        # Agent's turn
        move = agent.choose_move(state)
        if move is None:
            agent.learn_from_transition(state, 0.0, None, True)
            break
        material_before = evaluate_material(state)
        state_after_agent = make_move(state, move)
        material_after = evaluate_material(state_after_agent)
        reward = float(material_after - material_before) if agent.agent_plays_white else float(material_before - material_after)
        cumulative_reward += reward

        # Opponent reply
        legal_opponent = generate_moves(state_after_agent, not agent.agent_plays_white)
        if legal_opponent:
            next_state = make_move(state_after_agent, rng.choice(legal_opponent))
            done = not board_has_king(next_state, True) or not board_has_king(next_state, False)
        else:
            next_state = state_after_agent
            done = True

        agent.learn_from_transition(state, reward, next_state, done)
        state = next_state
        if done:
            break
        # After agent move followed by opponent reply within the same iteration,
        # the next turn cycles back to the agent.
        is_white_turn = agent.agent_plays_white

    return cumulative_reward


def train_agent(
    episodes: int = 5,
    max_steps_per_episode: int = 200,
    learning_rate: float = 0.05,
    discount_factor: float = 0.99,
    epsilon: float = 0.1,
    agent_plays_white: bool = True,
    seed: Optional[int] = 42,
):
    agent = RLValueAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        agent_plays_white=agent_plays_white,
        random_seed=seed,
    )
    history = {
        'episode_rewards': [],
    }
    for ep in range(episodes):
        ep_reward = play_episode(agent, max_steps=max_steps_per_episode, opponent_random_seed=(seed + ep) if seed is not None else None)
        history['episode_rewards'].append(ep_reward)
    return agent, history


if __name__ == "__main__":
    # Example: quick training run
    trained_agent, hist = train_agent(episodes=2, max_steps_per_episode=100)
    print("Episode rewards:", hist['episode_rewards'])
    print("Weights (first 6 shown):", trained_agent.weights[:6])
