# Imports
import time
import random
import numpy as np
import pickle as pkl
from collections import defaultdict


class GameState:
    """ Class for representing a state of the tic-tac-toe game. """
    _player_symbols = {0: " ", 1: "X", 2: "O"}

    def __init__(self, board=None, prev_player=2):
        """
        Initializes a GameState.
        If the given board parameter is None, initialize an empty board.

        Parameters
        ----------
        board: numpy.ndarray, shape=(3, 3) or None
            The cells of the tic-tac-toe board, with values either 0 (unoccupied) or a player number (1 or 2)
        prev_player: int
            1 or 2, the number of the player who is NOT about to make a move
        """
        if board is None:
            self.board = np.zeros((6, 7))
        else:
            self.board = board
        self.prev_player = prev_player

    def get_next_player(self):
        """
        Return the number of the next player, the one who is about to make a move.

        Returns
        -------
        int
        """
        if self.prev_player == 1:
            return 2
        else:
            return 1

    def has_ended(self):
        """
        Determine if the game has ended.

        If a row, column, or diagonal has been entirely filled by a player, return their player number
        If there is a draw (all cells are filled, no winner), return -1
        If the game is still going, return 0

        Returns
        -------
        int
        """
        board = self.board
        if 0 not in board:
            return -1
        for x in range(len(board) - 3):
            for y in range(len(board[0])):
                if np.array_equal(board[x:x + 4, y], np.array([1, 1, 1, 1])):
                    return 1
                if np.array_equal(board[x:x + 4, y], np.array([2, 2, 2, 2])):
                    return 2
        for y in range(len(board[0]) - 3):
            for x in range(len(board)):
                if np.array_equal(board[x, y:y + 4], np.array([1, 1, 1, 1])):
                    return 1
                if np.array_equal(board[x, y:y + 4], np.array([2, 2, 2, 2])):
                    return 2
        for x in range(len(board) - 3):
            for y in range(len(board[0]) - 3):
                if np.array_equal([board[x, y], board[x + 1, y + 1], board[x + 2, y + 2], board[x + 3, y + 3]],
                                  np.array([1, 1, 1, 1])):
                    return 1
                if np.array_equal([board[x, y], board[x + 1, y + 1], board[x + 2, y + 2], board[x + 3, y + 3]],
                                  np.array([2, 2, 2, 2])):
                    return 2
        for x in range(3, len(board)):
            for y in range(len(board[0]) - 3):
                if np.array_equal([board[x, y], board[x - 1, y + 1], board[x - 2, y + 2], board[x - 3, y + 3]],
                                  np.array([1, 1, 1, 1])):
                    return 1
                if np.array_equal([board[x, y], board[x - 1, y + 1], board[x - 2, y + 2], board[x - 3, y + 3]],
                                  np.array([2, 2, 2, 2])):
                    return 2
        return 0

    def get_possible_moves(self):
        """
        Return a list of all possible tic-tac-toe moves that the next player can make
        from this current state.

        Each move can be defined as a tuple of two integers, the row and column of the cell to choose,
        which are equivalent to the index of the cell in the board.

        i.e. For the following board,

         1 | 0 | 2
        -----------
         0 | 1 | 2
        -----------
         1 | 2 | 0

        Return [(0, 1), (1, 0), (2, 2)] or some variant.

        Returns
        -------
        List[tuple[int]]
            List of tuple pairs corresponding to possible moves.
        """
        moves = []
        for x in range(len(self.board[0])):
            if 0 in self.board[:, x]:
                moves.append(x)
        return moves

    def get_new_state(self, move):
        """
        Create a new GameState for the move played, with new board/player values.
        (Remember to COPY the board if you wish to change it.)

        Parameters
        ----------
        move: tuple[int]
            A tuple (r, c) of the move played by the next_player.

        Returns
        -------
        GameState
            The new GameState.
        """
        new_board = np.copy(self.board)
        row = 0
        while new_board[row, move] != 0:
            row += 1
        #             if row >= 6:
        #                 print("Invalid move, try again.")
        #                 return 1
        if self.prev_player == 1:
            new_board[row, move] = 2
            new_prev_player = 2
        else:
            new_board[row, move] = 1
            new_prev_player = 1
        return GameState(board=new_board, prev_player=new_prev_player)

    def __repr__(self):
        return self.board, self.prev_player

    def __str__(self):
        # Prettify board
        board_display = ("\n" + "-" * 27 + "\n").join(
            [" " + " | ".join([GameState._player_symbols[x] for x in row]) + " " for row in self.board[::-1]])
        string = " 0   1   2   3   4   5   6\n" + board_display + "\nNext player: " + str(self.get_next_player()) + "\n"
        return string


class Node:
    """ Nodes that build up the tree to search through. """

    def __init__(self, gamestate):
        """
        Initializes a Node with a GameState and empty dictionary of children.

        Parameters
        ----------
        gamestate: GameState
            The GameState associated with this node
        """
        self.gamestate = gamestate
        self.children = {}

    def get_children(self):
        """
        Expands the tree by getting all possible moves from self.gamestate,
        making new Nodes associated with each of these,
        and adding them to the self.children dictionary mapping move (tuple) to Node.

        Returns
        -------
        None
        """
        for move in self.gamestate.get_possible_moves():
            new_node = Node(self.gamestate.get_new_state(move))
            self.children[move] = new_node


class MonteCarlo:
    """ Class that handles Monte Carlo searches over the game tree. """

    def __init__(self, initial_time=10, calc_time=2, c=1.4, max_expansions=1, max_moves=None):
        """
        Initializes a MonteCarlo instance, which involves initializing the following instance variables:
         - search parameters, as passed to the constructor
         - self.history, a list containing the root node (Node corresponding to the initial, empty GameState)
         - self.wins, self.plays, empty defaultdicts mapping Nodes to ints that store win/total play records

        Then run an initial search.

        Parameters
        ----------
        initial_time: float
            Seconds for which to run the initial search
        calc_time: float
            Seconds for which to "think" before the computer makes a move
        c: float
            Exploration parameter for calculation of UCT
        max_expansions: int or None
            Maximum number of nodes to expand before entering pure simulation phase, or None if unlimited
        max_moves: int or None
            Maximum number of moves to try per simulation, or None if unlimited
        """
        self.initial_time = initial_time
        self.calc_time = calc_time
        self.c = c
        if max_expansions == None:
            self.max_expansions = float("inf")
        else:
            self.max_expansions = max_expansions
        if max_moves == None:
            self.max_moves = float("inf")
        else:
            self.max_moves = max_moves
        gs = GameState()
        node = Node(gs)
        self.history = [node]
        self.wins = defaultdict()
        self.plays = defaultdict()
        self.wins[node] = 0
        self.plays[node] = 0
        self.run_search(initial=True)

    def update(self, move):
        """
        When a move is made, get the next GameState and add its node to the game history.
        Return None if the move is invalid.

        Parameters
        ----------
        move: tuple[int]
            The move last made

        Returns
        -------
        Node
            The GameState that follows the move, now the last element of self.history
        """
        last_node = self.history[-1]
        if move in last_node.gamestate.get_possible_moves():
            new_gs = last_node.gamestate.get_new_state(move)
            node = Node(new_gs)
            self.history.append(node)
            return node
        else:
            return None

    def win_prob(self, node):
        """
        Calculate the win probability associated with the given Node.

        Parameters
        ----------
        node: Node

        Returns
        -------
        float
            The node's wins divided by the node's plays, or 0 if there have been no plays.
        """
        if node not in self.plays:
            self.plays[node] = 0
        if node not in self.wins:
            self.wins[node] = 0
        if self.plays[node] == 0:
            return 0
        return self.wins[node] / self.plays[node]

    def uct(self, node, parent):
        """
        Calculates a node's UCT.

        Parameters
        ----------
        node: Node
        parent: Node
            The node for which "node" is a direct child.

        Returns
        -------
        float
        """
        # Calculate exploitation (win ratio)
        exploitation = self.win_prob(node)

        # Calculate exploration (how unexplored it is)
        if self.plays[node] == 0:
            exploration = self.c * np.sqrt(np.log(self.plays[parent]) / 1e-10)
        exploration = self.c * np.sqrt(np.log(self.plays[parent]) / self.plays[node])

        return exploitation + exploration

    def search(self):
        """
        Perform one round of MCTS.

        1. Initialize a temporary history
        2. While the current node is not a terminal node (has_ended() returns 0), and we haven't yet reached the move limit:
            - Selection: choose moves by maximum UCT value for as long as we are
                         in explored territory, adding to the temporary history
            - Expansion: expand the current node, choosing a random move and adding to
                         the temporary history until the maximum number of expansions is reached
            - Simulation: choose random moves until the end of the game is reached
                          (do not add to the temporary history)
        3. Backpropagation: backpropagate through the temporary history and increment the number of plays for each node,
                            as well as the number of wins if its prev_player matches the game's overall winner or
                            if the game is a draw (by a lesser amount)
                            (the move that leads to this node is more likely to be picked by the previous player)
        """
        temp_history = [self.history[-1]]
        move_count = 0
        cur = temp_history[-1]
        expansions = 0
        while temp_history[-1].gamestate.has_ended() == 0 and move_count < self.max_moves:
            if len(cur.children) != 0:
                max_child = None
                max_uct = -99999
                for c in cur.children:
                    uct_val = self.uct(cur.children[c], cur)
                    if uct_val > max_uct:
                        max_uct = uct_val
                        max_child = cur.children[c]
                cur = max_child
                temp_history.append(cur)
            elif expansions < self.max_expansions:
                expansions += 1
                cur.get_children()
                move = random.choice(list(cur.children.keys()))
                cur = cur.children[move]
                temp_history.append(cur)
            else:
                move = random.choice(list(cur.children.keys()))
        winner = temp_history[-1].gamestate.has_ended()
        for node in temp_history:
            if node.gamestate.prev_player == winner:
                if node not in self.wins:
                    self.wins[node] = 1
                else:
                    self.wins[node] += 1
            elif winner == -1:
                if node not in self.wins:
                    self.wins[node] = 0.5
                else:
                    self.wins[node] += 0.5
            elif node not in self.wins:
                self.wins[node] = 0
            if node not in self.plays:
                self.plays[node] = 1
            else:
                self.plays[node] += 1

    def run_search(self, max_time=None, initial=False):
        """
        Run search for the given amount of time.

        Parameters
        ----------
        max_time: float
            Number of seconds for which to run the search
        initial: bool
            Defines what max_time should default to if it is given as None
            Choose self.initial_time if initial is True, else self.calc_time

        Returns
        -------
        None
        """
        if max_time is None:
            max_time = self.initial_time if initial else self.calc_time
        if initial: print("Running initial search...")

        t0 = time.time()
        sims = 0
        while time.time() - t0 < max_time:
            self.search()
            sims += 1

        print("Ran simulations:", sims)

    def make_computer_move(self):
        """
        Run search for calc_time seconds, choose the best (highest win probability) move from the
        set of possible moves, update the history, and return the status of the new GameState.

        Returns
        -------
        int
            Status of the game, either the winner number, -1 if the game is drawn, or 0 otherwise
        """
        t0 = time.time()
        while time.time() - t0 < self.calc_time:
            self.search()
        child_nodes = self.history[-1].children
        next_move = self.history[-1].gamestate.get_possible_moves()[0]
        for move in child_nodes:
            if self.win_prob(child_nodes[move]) > self.win_prob(child_nodes[next_move]):
                next_move = move
        new_node = self.update(next_move)
        return new_node.gamestate.has_ended()

    def make_player_move(self):
        """
        Get player input (repeat as long as input is invalid), and make a move.

        Returns
        -------
        int
            Status of the game, either the winner number, -1 if the game is drawn, or 0 otherwise
        """
        col = input("Which column would you like to put a piece in?\t")
        new_node = self.update(int(col))
        while new_node is None:
            col = input("Invalid input. Please input the column you want to put a piece in.\t")
            new_node = self.update(col)
        return new_node.gamestate.has_ended()


def play_game(player1=True, player2=False):
    """
    Plays a game, displaying the board each turn.

    Parameters
    ----------
    player1: bool
        Whether or not the first player (X) should be human-controlled.
    player2: bool
        Whether or not the second player (O) should be human-controlled.
    """

    game = MonteCarlo(max_expansions=None)
    print(game.history[-1].gamestate)

    while True:
        # Player 1 move
        if player1:
            x = game.make_player_move()
        else:
            x = game.make_computer_move()
        print(game.history[-1].gamestate)
        if x != 0: break

        # Player 2 move
        if player2:
            x = game.make_player_move()
        else:
            x = game.make_computer_move()
        print(game.history[-1].gamestate)
        if x != 0: break

    if game.history[-1].gamestate.has_ended() == -1:
        print("Game ended in a draw.")
    else:
        print(game.history[-1].gamestate.prev_player, " won.")


def main():
    play_game()


if __name__ == "__main__":
    main()