from contextlib import redirect_stderr
import copy
import pygame
import os 
import numpy as np
from numba import jit, cuda
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import time
import json
import webbrowser
import asyncio

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

# Global flag to track if we've already printed the GPU error
gpu_error_printed = False

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

@cuda.jit
@cuda.jit(device=True)
def evaluate_board_cuda(board_array, piece_values, result):
    """CUDA kernel for board evaluation"""
    row, col = cuda.grid(2)
    if row < board_array.shape[0] and col < board_array.shape[1]:
        piece = board_array[row, col]
        if piece in piece_values:
            cuda.atomic.add(result, 0, piece_values[piece])
        return

@jit(nopython=True)
def evaluate_cpu(board_array, piece_values):
    """CPU optimized board evaluation"""
    score = 0
    for row in range(8):
        for col in range(8):
            piece = board_array[row, col]
            if piece != ord('.'):
                score += piece_values[piece]
    return score

def evaluate(board):
    """Evaluate the board using either GPU or CPU acceleration"""
    global gpu_error_printed
    board_array = np.array([[ord(piece) for piece in row] for row in board], dtype=np.int32)
    piece_values = np.zeros(128, dtype=np.int32)  # ASCII range
    for piece, value in PIECE_VALUES.items():
        piece_values[ord(piece)] = value
    
    try:
        # Try GPU with optimized grid configuration
        board_array = cuda.to_device(board_array)
        result = np.zeros(1, dtype=np.int32)
        threadsperblock = (8, 8)  # Reduced thread block size for better occupancy
        blockspergrid = (
            (board_array.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
            (board_array.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        )
        
        # Ensure minimum grid size
        if blockspergrid[0] * blockspergrid[1] < 4:  # Minimum 4 blocks for better occupancy
            threadsperblock = (4, 4)
            blockspergrid = (2, 2)
        
        result = cuda.to_device(result)
        piece_values = cuda.to_device(piece_values)
        evaluate_board_cuda[blockspergrid, threadsperblock](board_array, piece_values, result)
        result.copy_to_host()
        return int(result[0])
    except Exception as e:
        # Fall back to CPU
        if not gpu_error_printed:
            print(f"GPU evaluation failed: {str(e)}")
            try:
                # Initialize Spotify client with user authentication
                sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                    client_id="d075525228dd4ec0a38d5adfdb26eb48",
                    client_secret="fad5558fd65d4ee28c5334916acfbacb",
                    redirect_uri="http://localhost:8888/callback",
                    scope="user-modify-playback-state user-read-playback-state"
                ))
                
                # Search for a fun song about computers/GPUs
                results = sp.search(q='computer error song', limit=1)
                if results['tracks']['items']:
                    track = results['tracks']['items'][0]
                    print(f"Playing: {track['name']} by {track['artists'][0]['name']}")
                    
                    # Get available devices
                    devices = sp.devices()
                    if devices['devices']:
                        # Use the first available device
                        device_id = devices['devices'][0]['id']
                        # Start playback
                        sp.start_playback(device_id=device_id, uris=[track['uri']])
                    else:
                        print("No active devices found. Please open Spotify on your device.")
            except Exception as spotify_error:
                print(f"Couldn't play song: {str(spotify_error)}")
            
            gpu_error_printed = True
        # Ensure board_array is contiguous for CPU evaluation
        return evaluate_cpu(board_array, piece_values)

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

@jit(nopython=True, parallel=True)
def generate_moves_cpu(board_array, is_white_turn):
    """JIT-compiled move generation for CPU"""
    moves = []
    for row in range(8):
        for col in range(8):
            piece = chr(board_array[row, col])
            if (is_white_turn and piece.isupper()) or (not is_white_turn and piece.islower()):
                moves.extend(generate_piece_moves(board_array, row, col, piece))
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

@jit(nopython=True)
def alpha_beta_cpu(board_array, depth, alpha, beta, is_white_turn):
    """JIT-compiled alpha-beta pruning for CPU"""
    if depth == 0:
        return evaluate_cpu(board_array, PIECE_VALUES)
    
    moves = generate_moves_cpu(board_array, is_white_turn)
    if is_white_turn:
        max_eval = -float('inf')
        for move in moves:
            new_board = make_move(board_array, move)
            eval = alpha_beta_cpu(new_board, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = make_move(board_array, move)
            eval = alpha_beta_cpu(new_board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

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
        
        # Load pieces
        self.pieces = {}
        self.load_pieces()
        
        self.board = initial_board
        self.is_white_turn = True
        self.running = True
        self.clock = pygame.time.Clock()

    def load_pieces(self):
        """Load piece images from white and black folders."""
        # White pieces (uppercase)
        white_pieces = 'KQRBNP'
        for piece in white_pieces:
            try:
                img_path = os.path.join('pieces', 'white', f'{piece}.png')
                img = pygame.image.load(img_path)
                self.pieces[piece] = pygame.transform.scale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
            except pygame.error as e:
                print(f"Error loading white piece image for {piece}: {e}")
        
        # Black pieces (lowercase)
        black_pieces = 'kqrbnp'
        for piece in black_pieces:
            try:
                img_path = os.path.join('pieces', 'black', f'{piece}.png')
                img = pygame.image.load(img_path)
                self.pieces[piece] = pygame.transform.scale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
            except pygame.error as e:
                print(f"Error loading black piece image for {piece}: {e}")

    def draw_board(self):
        """Draw the chessboard and pieces."""
        for row in range(8):
            for col in range(8):
                # Draw square
                color = self.WHITE if (row + col) % 2 == 0 else self.GRAY
                pygame.draw.rect(self.screen, color, 
                                 (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, 
                                  self.SQUARE_SIZE, self.SQUARE_SIZE))
                
                # Draw piece
                piece = self.board[row][col]
                if piece != '.' and piece in self.pieces:
                    self.screen.blit(self.pieces[piece], 
                                     (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE))
        pygame.display.flip()

    def main_loop(self):
        """Main game loop for the GUI."""
        depth = 5  # Set the AI depth
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # AI move
            move = find_best_move(self.board, depth, self.is_white_turn)
            if move is None:
                print("Game over!")
                self.running = False
            else:
                self.board = make_move(self.board, move)
                self.is_white_turn = not self.is_white_turn
            
            self.draw_board()
            pygame.time.wait(50)  # Add delay for better visualization
            self.clock.tick(30)

if __name__ == "__main__":
    # Initialize Numba JIT
    dummy_board = np.array([[ord('.') for _ in range(8)] for _ in range(8)], dtype=np.int32)
    evaluate_cpu(dummy_board, np.zeros(128, dtype=np.int32))  # Warm up JIT
    ChessGUI().main_loop()