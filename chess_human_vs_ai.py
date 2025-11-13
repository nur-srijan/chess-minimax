import pygame
import os
import sys
import copy

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

class ChessGUI:
    def __init__(self, ai_plays_white=False):
        pygame.init()
        self.SQUARE_SIZE = 80
        self.BOARD_SIZE = 8 * self.SQUARE_SIZE
        self.screen = pygame.display.set_mode((self.BOARD_SIZE, self.BOARD_SIZE))
        pygame.display.set_caption("Chess: Human vs AI")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.HIGHLIGHT = (247, 247, 105, 100)
        
        # Game state
        self.board = [row[:] for row in initial_board]
        self.ai_plays_white = ai_plays_white
        self.human_turn = not ai_plays_white  # Human goes first by default unless AI is white
        self.selected_piece = None
        self.valid_moves = []
        self.game_over = False
        
        # Load piece images
        self.pieces = {}
        self.load_pieces()
        
        # AI settings
        self.ai_depth = 3  # Adjustable AI difficulty
        
    def load_pieces(self):
        """Load piece images from the pieces directory."""
        pieces_dir = 'pieces'
        colors = ['white', 'black']
        
        for color in colors:
            color_dir = os.path.join(pieces_dir, color)
            if not os.path.exists(color_dir):
                continue
                
            for piece_file in os.listdir(color_dir):
                if piece_file.endswith('.svg'):
                    piece_code = piece_file[0]  # First letter is the piece code
                    if color == 'black':
                        piece_code = piece_code.lower()
                    
                    try:
                        img = pygame.image.load(os.path.join(color_dir, piece_file))
                        self.pieces[piece_code] = pygame.transform.scale(img, 
                                                                       (self.SQUARE_SIZE, self.SQUARE_SIZE))
                    except pygame.error as e:
                        print(f"Error loading {piece_file}: {e}")
                        # Create a placeholder for missing pieces
                        surf = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                        surf.fill((255, 0, 255))  # Magenta for missing pieces
                        font = pygame.font.SysFont('Arial', 20)
                        text = font.render(piece_code.upper(), True, (0, 0, 0))
                        text_rect = text.get_rect(center=(self.SQUARE_SIZE//2, self.SQUARE_SIZE//2))
                        surf.blit(text, text_rect)
                        self.pieces[piece_code] = surf
    
    def draw_board(self):
        """Draw the chess board and pieces."""
        for row in range(8):
            for col in range(8):
                # Draw the square
                color = self.WHITE if (row + col) % 2 == 0 else self.GRAY
                pygame.draw.rect(self.screen, color, 
                               (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, 
                                self.SQUARE_SIZE, self.SQUARE_SIZE))
                
                # Highlight selected piece and valid moves
                if self.selected_piece and (row, col) in self.valid_moves:
                    highlight = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
                    highlight.fill((247, 247, 105, 100))  # Semi-transparent yellow
                    self.screen.blit(highlight, (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE))
                
                # Draw the piece
                piece = self.board[row][col]
                if piece != '.':
                    if piece in self.pieces:
                        self.screen.blit(self.pieces[piece], 
                                       (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE))
        
        # Display current turn
        font = pygame.font.SysFont('Arial', 20)
        turn_text = "Your turn (" + ("White" if self.human_turn and not self.ai_plays_white or 
                                    not self.human_turn and self.ai_plays_white else "Black") + ")"
        text_surface = font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
    
    def get_ai_move(self):
        """Get the AI's move using minimax with alpha-beta pruning."""
        best_move = None
        best_eval = -float('inf') if self.ai_plays_white else float('inf')
        
        # Generate all possible moves for the AI
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (self.ai_plays_white and piece.isupper()) or (not self.ai_plays_white and piece.islower()):
                    moves = self.generate_piece_moves(row, col, piece)
                    for move in moves:
                        new_board = self.make_move(move)
                        eval = self.minimax(new_board, self.ai_depth - 1, -float('inf'), float('inf'), False)
                        
                        if (self.ai_plays_white and eval > best_eval) or (not self.ai_plays_white and eval < best_eval):
                            best_eval = eval
                            best_move = move
        
        return best_move
    
    def minimax(self, board, depth, alpha, beta, is_maximizing):
        """Minimax algorithm with alpha-beta pruning."""
        if depth == 0:
            return self.evaluate_board(board)
        
        if is_maximizing:
            max_eval = -float('inf')
            for row in range(8):
                for col in range(8):
                    piece = board[row][col]
                    if (self.ai_plays_white and piece.isupper()) or (not self.ai_plays_white and piece.islower()):
                        moves = self.generate_piece_moves(row, col, piece)
                        for move in moves:
                            new_board = self.make_move(move, board)
                            eval = self.minimax(new_board, depth - 1, alpha, beta, False)
                            max_eval = max(max_eval, eval)
                            alpha = max(alpha, eval)
                            if beta <= alpha:
                                break
            return max_eval
        else:
            min_eval = float('inf')
            for row in range(8):
                for col in range(8):
                    piece = board[row][col]
                    if (not self.ai_plays_white and piece.isupper()) or (self.ai_plays_white and piece.islower()):
                        moves = self.generate_piece_moves(row, col, piece)
                        for move in moves:
                            new_board = self.make_move(move, board)
                            eval = self.minimax(new_board, depth - 1, alpha, beta, True)
                            min_eval = min(min_eval, eval)
                            beta = min(beta, eval)
                            if beta <= alpha:
                                break
            return min_eval
    
    def evaluate_board(self, board):
        """Evaluate the board position."""
        score = 0
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece in PIECE_VALUES:
                    score += PIECE_VALUES[piece]
        return score if self.ai_plays_white else -score
    
    def generate_piece_moves(self, row, col, piece):
        """Generate valid moves for a piece."""
        moves = []
        piece_lower = piece.lower()
        
        if piece_lower == 'p':  # Pawn
            direction = -1 if piece.isupper() else 1
            start_row = 6 if piece.isupper() else 1
            
            # Move forward
            if 0 <= row + direction < 8 and self.board[row + direction][col] == '.':
                moves.append((row, col, row + direction, col))
                # Double move from starting position
                if row == start_row and self.board[row + 2*direction][col] == '.':
                    moves.append((row, col, row + 2*direction, col))
            
            # Capture diagonally
            for dc in [-1, 1]:
                new_col = col + dc
                if 0 <= new_col < 8 and 0 <= row + direction < 8:
                    target = self.board[row + direction][new_col]
                    if target != '.':
                        if (piece.isupper() and target.islower()) or (piece.islower() and target.isupper()):
                            moves.append((row, col, row + direction, new_col))
        
        elif piece_lower == 'n':  # Knight
            knight_moves = [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target == '.' or (piece.isupper() and target.islower()) or (piece.islower() and target.isupper()):
                        moves.append((row, col, new_row, new_col))
        
        else:  # Other pieces (simplified for brevity)
            # This is a simplified version - you should expand this with proper move generation
            # for kings, queens, rooks, and bishops
            pass
            
        return moves
    
    def make_move(self, move, board=None):
        """Make a move on a copy of the board and return the new board."""
        if board is None:
            board = [row[:] for row in self.board]
        else:
            board = [row[:] for row in board]
            
        row_from, col_from, row_to, col_to = move
        board[row_to][col_to] = board[row_from][col_from]
        board[row_from][col_from] = '.'
        return board
    
    def is_valid_move(self, move):
        """Check if a move is valid."""
        row_from, col_from, row_to, col_to = move
        
        # Check if source has a piece
        if self.board[row_from][col_from] == '.':
            return False
            
        # Check if it's the player's turn
        piece = self.board[row_from][col_from]
        if (self.human_turn and ((self.ai_plays_white and piece.islower()) or 
                               (not self.ai_plays_white and piece.isupper()))):
            return False
            
        # Check if the move is in valid moves
        if move in self.valid_moves:
            return True
            
        return False
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        
        while not self.game_over:
            if not self.human_turn:
                # AI's turn
                ai_move = self.get_ai_move()
                if ai_move:
                    self.board = self.make_move(ai_move)
                self.human_turn = True
                self.selected_piece = None
                self.valid_moves = []
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                elif event.type == pygame.MOUSEBUTTONDOWN and self.human_turn:
                    if event.button == 1:  # Left click
                        x, y = event.pos
                        col = x // self.SQUARE_SIZE
                        row = y // self.SQUARE_SIZE
                        
                        # If a piece is already selected, try to move it
                        if self.selected_piece:
                            move = (self.selected_piece[0], self.selected_piece[1], row, col)
                            if self.is_valid_move(move):
                                self.board = self.make_move(move)
                                self.human_turn = False
                                self.selected_piece = None
                                self.valid_moves = []
                            else:
                                # Select a different piece or deselect
                                piece = self.board[row][col]
                                if piece != '.':
                                    self.selected_piece = (row, col)
                                    self.valid_moves = self.generate_piece_moves(row, col, piece)
                                else:
                                    self.selected_piece = None
                                    self.valid_moves = []
                        else:
                            # Select a piece
                            piece = self.board[row][col]
                            if piece != '.':
                                # Only select pieces of the current player's color
                                if (self.ai_plays_white and piece.isupper()) or (not self.ai_plays_white and piece.islower()):
                                    self.selected_piece = (row, col)
                                    self.valid_moves = self.generate_piece_moves(row, col, piece)
                                else:
                                    self.selected_piece = None
                                    self.valid_moves = []
                            else:
                                self.selected_piece = None
                                self.valid_moves = []
            
            self.draw_board()
            clock.tick(30)

if __name__ == "__main__":
    # Create and run the game
    # Set ai_plays_white=True if you want the AI to play as white
    game = ChessGUI(ai_plays_white=False)
    game.run()
