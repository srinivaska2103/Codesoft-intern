import math
#Initialize the board
board = [' ' for _ in range(9)]  # 3x3 grid (indexes 0–8)

def print_board():
    print()
    for i in range(3):
        print(' | '.join(board[i*3:(i+1)*3]))
        if i < 2:
            print('---------')
    print()


def check_winner(player):
    win_conditions = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

# Check if board is full
def is_full():
    return ' ' not in board

# Minimax algorithm
def minimax(is_maximizing):
    if check_winner('O'):
        return 1
    elif check_winner('X'):
        return -1
    elif is_full():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(False)
                board[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(True)
                board[i] = ' '
                best_score = min(score, best_score)
        return best_score


def best_move():
    best_score = -math.inf
    move = None
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    return move

# Main game loop
def play_game():
    print("Welcome to Tic-Tac-Toe!")
    print_board()

    while True:
        # Human move
        while True:
            try:
                move = int(input("Enter your move (1-9): ")) - 1
                if move < 0 or move > 8 or board[move] != ' ':
                    print("Invalid move, try again.")
                else:
                    board[move] = 'X'
                    break
            except ValueError:
                print("Please enter a valid number (1-9).")

        print_board()

     
        if check_winner('X'):
            print("You win! 🎉")
            break

        if is_full():
            print("It's a tie! 🤝")
            break

       
        print("AI is thinking... 🤖")
        move = best_move()
        board[move] = 'O'
        print_board()

        if check_winner('O'):
            print("AI wins! 🧠💥")
            break

        if is_full():
            print("It's a tie! 🤝")
            break


# Run the game
if __name__ == "__main__":
    play_game()
