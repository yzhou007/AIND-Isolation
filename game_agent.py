"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def custom_score(game, player):
    """Outputs a score equal to the difference in the number of moves
    available to the two players, while penalizing the moves for the
    maximizing player that are in the corner and rewarding the moves for the
    minimizing player that are in the corner. These penalties/rewards are
    elevated near end game through a game state factor. Submitted.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")  
    
    return heuristic_function1(game, player)

def heuristic_function1(game, player):
    """It varies the weights to the number of moves of my own 
       and opponent players depending on how far into the game.
       When the occupied spaces are smaller than 0.3 of the total board spaces, 
       I output the score equal to “# of my_moves - 1.5 * # of opponent moves”. 
       When the occupied spaces are larger than 0.3 of the total board spaces, 
       I output the score equal to “# of my_moves - 3 * # of opponent moves”. 
    """
    
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    blank_spaces = game.get_blank_spaces()
    occupied_spaces = game.height * game.width - len(blank_spaces)
    
    if occupied_spaces < 0.3 * game.height * game.width:
         return float(len(own_moves) - 1.5 * len(opp_moves))
        
    return float(len(own_moves) - 3.0 * len(opp_moves))

def heuristic_function2(game, player):
    """ It uses 1 and 2 for the weights of the number of moves of my own 
        and opponent players.  In addition, it uses the strategy that 
        keeps my own player stay away from the 4 corners while let the 
        opponent player get trapped by the 4 corners, 
        when it gets closer to the end of the game. 
    """
    
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    
    corners = [(0, 0), (game.height-1, 0), (game.height-1, game.width-1), (0, game.width-1)]
    blank_spaces = game.get_blank_spaces()
    occupied_spaces = game.height * game.width - len(blank_spaces)
    
    own_corners, opp_corners = [], []          
    if occupied_spaces < 0.3 * game.height * game.width:
        if corners in blank_spaces:
            own_corners = [move for move in own_moves if move in corners]
            opp_corners = [move for move in opp_moves if move in corners]
            
    return float(len(own_moves) - 2 * len(opp_moves) - 5000*len(own_corners) + 50000*len(opp_corners))

    
def heuristic_function3(game, player):
    """It uses 1 and 2 for the weights of the number of moves of my own 
       and opponent players.  In addition, it adopts the strategy that 
       let my own player occupy the center position as quickly as possible 
       and keep the opponent player away from the center. 
    """
    
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    blank_spaces = game.get_blank_spaces()
    occupied_spaces = game.height * game.width - len(blank_spaces)
    center = ((game.height - 1) / 2, (game.width - 1) / 2)
    
    own_center = []
    opp_center = []
    if occupied_spaces < 0.3 * game.height * game.width:
        if center in blank_spaces:
            own_center = [move for move in own_moves if move == center]
            opp_center = [move for move in opp_moves if move == center]
           
    return float(len(own_moves) - 2 * len(opp_moves) + 5000 * len(own_center) - 50000 *len(opp_center))
    

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return -1, -1
        
        best_move = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                max_depth = game.height * game.width
                for depth in range(1, max_depth):
                    # call methods based on their string names 
                    _, best_move = getattr(self, self.method)(game, depth)
            else:
                # call methods based on their string names 
                _, best_move = getattr(self, self.method)(game, self.search_depth)
                
        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
            
        # When the depth is equal to zero, return the score value from the custom_socre() function
        if depth == 0:
            return self.score(game, self), (-1, -1)
        
        legal_moves = game.get_legal_moves()
        if not legal_moves: # When there is no available move, return a utility value 
            return game.utility(self), (-1, -1)
        
        best_move = None
        if maximizing_player: # my own player layer, get the maximum value
            best_value = float("-inf")
            for move in legal_moves:
                next_game = game.forecast_move(move)
                value, _ = self.minimax(next_game, depth - 1, False)
                if best_value < value:
                    best_value = value
                    best_move = move
        else:
            best_value = float("inf") 
            for move in legal_moves: # opponent player layer, get the minimum value 
                next_game = game.forecast_move(move)
                value, _ = self.minimax(next_game, depth - 1, True)
                if best_value > value:
                    best_value = value
                    best_move = move
        return best_value, best_move
    
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        # When the depth is equal to zero, return the score value from the custom_socre() function
        if depth == 0:
            return self.score(game, self), (-1, -1)
        
        legal_moves = game.get_legal_moves()
        if not legal_moves: # When there is no available move, return a utility value 
            return game.utility(self), (-1, -1)
        
        best_move = None
        if maximizing_player: 
            best_value = float("-inf")
            for move in legal_moves: # my own player layer, get the maximum value
                next_game = game.forecast_move(move)
                value, _ = self.alphabeta(next_game, depth - 1, alpha, beta, False)
                if best_value < value:
                    best_value = value
                    best_move = move
                if best_value >= beta: # pruning unecessary branches
                    return best_value, best_move
                alpha = max(alpha, best_value)
        else :
            best_value = float("inf")  
            for move in legal_moves: # opponent player layer, get the minimum value 
                next_game = game.forecast_move(move)
                value, _ = self.alphabeta(next_game, depth - 1, alpha, beta, True)
                if best_value > value:
                    best_value = value
                    best_move = move
                if best_value <= alpha: # pruning unecessary branches
                    return best_value, best_move
                beta = min(beta, best_value)
        return best_value, best_move
