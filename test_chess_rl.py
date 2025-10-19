import copy
import types

import chess_rl as rl


def test_initial_moves_white_has_pawn_and_knight_moves():
    board = copy.deepcopy(rl.initial_board)
    moves = rl.generate_moves(board, is_white_turn=True)
    # Pawn forward moves from rank 2
    pawn_moves = [m for m in moves if board[m[0]][m[1]] == 'P']
    assert len(pawn_moves) >= 8  # at least the single-step forwards
    # Knights have two moves each from b1/g1 initially
    knight_moves = [m for m in moves if board[m[0]][m[1]] == 'N']
    assert len(knight_moves) == 4


def test_featurize_dimensions_and_bias():
    agent = rl.RLValueAgent(random_seed=0)
    feats = agent.featurize(rl.initial_board)
    assert isinstance(feats, list)
    assert len(feats) == 13
    assert feats[0] == 1.0  # bias term


def test_choose_move_returns_legal_move():
    agent = rl.RLValueAgent(random_seed=1, epsilon=0.0)  # greedy to make deterministic given seed
    board = copy.deepcopy(rl.initial_board)
    move = agent.choose_move(board)
    assert move is not None
    is_white_turn = agent.agent_plays_white
    legal_moves = rl.generate_moves(board, is_white_turn)
    assert move in legal_moves


def test_learn_from_transition_updates_weights():
    agent = rl.RLValueAgent(learning_rate=0.1, random_seed=0)
    state = copy.deepcopy(rl.initial_board)
    next_state = rl.make_move(state, (6, 4, 4, 4))  # e2->e4
    before = list(agent.weights)
    agent.learn_from_transition(state, reward=0.5, next_state=next_state, done=False)
    after = list(agent.weights)
    assert after != before


def test_play_episode_runs_and_returns_reward():
    agent = rl.RLValueAgent(random_seed=0)
    total_reward = rl.play_episode(agent, max_steps=20, opponent_random_seed=123)
    assert isinstance(total_reward, float)
