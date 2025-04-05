import copy
import pygame
import sys

# Constants for piece values
PIECE_VALUES = {
    'K': 1000, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,
    'k': -1000, 'q': -9, 'r': -5, 'b': -3, 'n': -3, 'p': -1
}

# Initial board setup
initial_board = [
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
]

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Chess")

# Load pieces
pieces = {}
for piece, folder in [('K', 'white'), ('Q', 'white'), ('R', 'white'), ('B', 'white'), ('N', 'white'), ('P', 'white'),
                      ('k', 'black'), ('q', 'black'), ('r', 'black'), ('b', 'black'), ('n', 'black'), ('p', 'black')]:
    try:
        img = pygame.image.load(f'pieces/{folder}/{piece}.png')
        pieces[piece] = pygame.transform.scale(img, (75, 75))  # Scale to fit the board squares
    except pygame.error as e:
        print(f"Error loading piece image for {piece}: {e}")

def draw_board(board):
    colors = [pygame.Color(235, 235, 208), pygame.Color(119, 148, 85)]
    for row in range(8):
        for col in range(8):
            # Draw the square
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * 75, row * 75, 75, 75))
            
            # Draw the piece
            piece = board[row][col]
            if piece != '.' and piece in pieces:
                screen.blit(pieces[piece], (col * 75, row * 75))

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def evaluate(board):
    """Evaluate the board and return a score."""
    score = 0
    for row in board:
        for piece in row:
            if piece in PIECE_VALUES:
                score += PIECE_VALUES[piece]
    return score

def generate_piece_moves(board, row, col, piece):
    """Generate all possible moves for a given piece."""
    moves = []
    piece_type = piece[0].lower()  # changed to handle labels like 'P2'
    is_white = piece.isupper()

    if piece_type == 'p':  # Pawn
        direction = -1 if is_white else 1
        # Forward move
        if 0 <= row + direction < 8 and board[row + direction][col] == '.':
            moves.append((row, col, row + direction, col))
            # Initial two-square move
            if (row == 6 and is_white) or (row == 1 and not is_white):
                if board[row + 2*direction][col] == '.':
                    moves.append((row, col, row + 2*direction, col))
        # Captures
        for col_offset in [-1, 1]:
            if 0 <= col + col_offset < 8 and 0 <= row + direction < 8:
                target = board[row + direction][col + col_offset]
                if target != '.' and target.isupper() != is_white:
                    moves.append((row, col, row + direction, col + col_offset))

    elif piece_type == 'r':  # Rook
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for i in range(1, 8):
                new_row, new_col = row + direction[0]*i, col + direction[1]*i
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

    elif piece_type == 'n':  # Knight
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                       (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for move in knight_moves:
            new_row, new_col = row + move[0], col + move[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == '.' or target.isupper() != is_white:
                    moves.append((row, col, new_row, new_col))

    elif piece_type == 'b':  # Bishop
        for direction in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for i in range(1, 8):
                new_row, new_col = row + direction[0]*i, col + direction[1]*i
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

    elif piece_type == 'q':  # Queen (combination of Rook and Bishop moves)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for direction in directions:
            for i in range(1, 8):
                new_row, new_col = row + direction[0]*i, col + direction[1]*i
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
        king_moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for move in king_moves:
            new_row, new_col = row + move[0], col + move[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == '.' or target.isupper() != is_white:
                    moves.append((row, col, new_row, new_col))

    return moves

def generate_moves(board, is_white_turn):
    """Generate all possible moves for the current player."""
    moves = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if (is_white_turn and piece.isupper()) or (not is_white_turn and piece.islower()):
                moves.extend(generate_piece_moves(board, row, col, piece))
    return moves

def make_move(board, move):
    """Make a move on the board."""
    new_board = copy.deepcopy(board)
    row_from, col_from, row_to, col_to = move
    new_board[row_to][col_to] = new_board[row_from][col_from]
    new_board[row_from][col_from] = '.'
    return new_board

def alpha_beta(board, depth, alpha, beta, is_white_turn):
    """Alpha-beta pruning algorithm."""
    if depth == 0:
        return evaluate(board)
    
    moves = generate_moves(board, is_white_turn)
    if is_white_turn:
        max_eval = -float('inf')
        for move in moves:
            new_board = make_move(board, move)
            eval = alpha_beta(new_board, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = make_move(board, move)
            eval = alpha_beta(new_board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth, is_white_turn):
    """Find the best move using alpha-beta pruning."""
    best_move = None
    best_eval = -float('inf') if is_white_turn else float('inf')
    moves = generate_moves(board, is_white_turn)
    for move in moves:
        new_board = make_move(board, move)
        eval = alpha_beta(new_board, depth - 1, -float('inf'), float('inf'), not is_white_turn)
        if (is_white_turn and eval > best_eval) or (not is_white_turn and eval < best_eval):
            best_eval = eval
            best_move = move
    return best_move

def parse_move(move_str):
    """Parse a move string in the format 'e2e4'."""
    col_from = ord(move_str[0]) - ord('a')
    row_from = 8 - int(move_str[1])
    col_to = ord(move_str[2]) - ord('a')
    row_to = 8 - int(move_str[3])
    return (row_from, col_from, row_to, col_to)

def is_in_check(board, is_white_turn):
    """Check if the king of the current player is in check."""
    king = 'K' if is_white_turn else 'k'
    king_pos = None
    for r in range(8):
        for c in range(8):
            if board[r][c] != '.' and board[r][c][0] == king:
                king_pos = (r, c)
                break
        if king_pos:
            break
    if king_pos is None:
        return True  # king missing, treat as in check
    
    # Check if any opponent move can capture the king
    opponent_turn = not is_white_turn
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece != '.' and ((opponent_turn and piece.isupper()) or (not opponent_turn and piece.islower())):
                moves = generate_piece_moves(board, r, c, piece)
                for move in moves:
                    _, _, move_r, move_c = move
                    if (move_r, move_c) == king_pos:
                        return True
    return False

def is_legal_move(board, move, is_white_turn):
    """Check if the move is legal."""
    row_from, col_from, row_to, col_to = move
    piece = board[row_from][col_from]
    if (is_white_turn and piece.islower()) or (not is_white_turn and piece.isupper()):
        return False
    legal_moves = generate_piece_moves(board, row_from, col_from, piece)
    if move not in legal_moves:
        return False
    # Check condition: move should not leave the king in check.
    new_board = make_move(board, move)
    if is_in_check(new_board, is_white_turn):
        return False
    return True

# Main game loop
board = initial_board
is_white_turn = True
depth = 3
dragging = False
dragged_piece = None
drag_start_pos = None
score = 0  # Initialize score for the game

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            col = x // 75
            row = y // 75
            if board[row][col] != '.':
                dragging = True
                dragged_piece = board[row][col]
                drag_start_pos = (row, col)
        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging:
                x, y = event.pos
                col = x // 75
                row = y // 75
                # Ensure the move is within bounds
                if 0 <= row < 8 and 0 <= col < 8:
                    # new: update score if a capture occurs
                    captured = board[row][col]
                    if captured != '.':
                        if is_white_turn and captured[0].islower():
                            score += abs(PIECE_VALUES.get(captured[0], 0))
                        elif (not is_white_turn) and captured[0].isupper():
                            score -= abs(PIECE_VALUES.get(captured[0], 0))
                        print("Score:", score)
                    move = (drag_start_pos[0], drag_start_pos[1], row, col)
                    if is_legal_move(board, move, is_white_turn):
                        board = make_move(board, move)
                        is_white_turn = not is_white_turn
                    else:
                        print("Illegal move!")
                        dragging = False
                        dragged_piece = None
                        drag_start_pos = None
                else:
                    print("Move out of bounds!")
                dragging = False
                dragged_piece = None
                drag_start_pos = None
        # Handle dragging
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                x, y = event.pos
                screen.fill((0, 0, 0))
                draw_board(board)
                screen.blit(pieces[dragged_piece], (x - 37.5, y - 37.5))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                dragging = False
                dragged_piece = None
                drag_start_pos = None
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                dragging = False
                dragged_piece = None
                drag_start_pos = None
    # Draw the board and pieces
    screen.fill((0, 0, 0))

    draw_board(board)
    pygame.display.flip()