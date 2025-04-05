import copy
import pygame
import os

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
    ['P2', 'P2', 'P2', 'P2', 'P2', 'P2', 'P2', 'P2'],
    ['R2', 'N2', 'B2', 'Q2', 'K2', 'B2', 'N2', 'R2']
]

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
    if piece.lower() == 'p':
        direction = -1 if piece.isupper() else 1
        if 0 <= row + direction < 8 and board[row + direction][col] == '.':
            moves.append((row, col, row + direction, col))
        if 0 <= row + direction < 8 and col > 0 and board[row + direction][col - 1] != '.' and board[row + direction][col - 1].islower() != piece.islower():
            moves.append((row, col, row + direction, col - 1))
        if 0 <= row + direction < 8 and col < 7 and board[row + direction][col + 1] != '.' and board[row + direction][col + 1].islower() != piece.islower():
            moves.append((row, col, row + direction, col + 1))
    elif piece.lower() == 'n':
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in knight_moves:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == '.' or (target.isupper() != piece.isupper()):
                    moves.append((row, col, new_row, new_col))
    # Add more logic for other pieces (Bishop, Rook, Queen, King) here
    elif piece.lower() == 'r':
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row = row + dr
            new_col = col + dc
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == '.':
                    moves.append((row, col, new_row, new_col))
                elif target.isupper() != piece.isupper():
                    moves.append((row, col, new_row, new_col))
                    break
                else:
                    break
                new_row += dr
                new_col += dc
    elif piece.lower() == 'b':
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_row = row + dr
            new_col = col + dc
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == '.':
                    moves.append((row, col, new_row, new_col))
                elif target.isupper() != piece.isupper():
                    moves.append((row, col, new_row, new_col))
                    break
                else:
                    break
                new_row += dr
                new_col += dc
    elif piece.lower() == 'k':
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = board[new_row][new_col]
                    if target == '.' or (target.isupper() != piece.isupper()):
                        moves.append((row, col, new_row, new_col))
    elif piece.lower() == 'q':
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_row = row + dr
            new_col = col + dc
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == '.':
                    moves.append((row, col, new_row, new_col))
                elif target.isupper() != piece.isupper():
                    moves.append((row, col, new_row, new_col))
                    break
                else:
                    break
                new_row += dr
                new_col += dc
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

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.SQUARE_SIZE = 80
        self.BOARD_SIZE = 8 * self.SQUARE_SIZE
        self.screen = pygame.display.set_mode((self.BOARD_SIZE, self.BOARD_SIZE))
        pygame.display.set_caption("Chess AI")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.BLACK = (0, 0, 0)
        
        # Load pieces
        self.pieces = {}
        piece_chars = 'KQRBNPkqrbnp'
        for piece in piece_chars:
            try:
                img_path = os.path.join('pieces', f'{piece}.png')
                img = pygame.image.load(img_path)
                self.pieces[piece] = pygame.transform.scale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
            except:
                print(f"Couldn't load piece image: {piece}")
    
    def draw_board(self, board):
        for row in range(8):
            for col in range(8):
                # Draw square
                color = self.WHITE if (row + col) % 2 == 0 else self.GRAY
                pygame.draw.rect(self.screen, color, 
                               (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, 
                                self.SQUARE_SIZE, self.SQUARE_SIZE))
                
                # Draw piece
                piece = board[row][col]
                if piece != '.' and piece in self.pieces:
                    self.screen.blit(self.pieces[piece], 
                                   (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE))
        
        pygame.display.flip()

# Replace the main game loop with:
def main():
    board = initial_board
    is_white_turn = True
    depth = 3
    gui = ChessGUI()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        gui.draw_board(board)
        move = find_best_move(board, depth, is_white_turn)
        
        if move is None:
            print("Game over!")
            break
        
        board = make_move(board, move)
        is_white_turn = not is_white_turn
        
        # Add a small delay to make the moves visible
        pygame.time.wait(500)

    pygame.quit()

if __name__ == "__main__":
    main()