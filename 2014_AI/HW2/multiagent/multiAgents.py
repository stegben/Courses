# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
	
    "*** YOUR CODE HERE ***"
	# score setting
    foodEaten = 100
    ghostEaten = 300
    
	
	# more information
    newGhostPosition = [ngs.getPosition() for ngs in newGhostStates]
    distGhost = [util.manhattanDistance(newPos , ngp) for ngp in newGhostPosition]
    distFood  = [util.manhattanDistance(newPos , ofp) for ofp in oldFood.asList()]
	
	# scoring start!
    if successorGameState.isWin():
        return float("inf") 
    # score = -util.manhattanDistance(newPos , newGhostStates.getPacmanPosition())
    score = 0
    if newPos in oldFood.asList():
        score += foodEaten
    for df in distFood:
        score += 5/(df+0.001)
    for ghost in newGhostStates:
        dist = util.manhattanDistance(newPos , ghost.getPosition())
        if ghost.scaredTimer>0:
            score += 300 / (dist+0.001)
            score += ghostEaten if newPos==ghost.getPosition() else 0
        else:
            score -= 100 / (dist+0.001) 
	
    score += successorGameState.getScore() 
    return score

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def MinMax(gameState , depth , AgentIndex ):
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
		# for pacman, find max
        if AgentIndex == 0:
            return max( [-float("inf")] + 
						[MinMax(gameState.generateSuccessor(AgentIndex, action)
							    ,depth 
							    ,AgentIndex + 1  ) 
					for action in gameState.getLegalActions(AgentIndex)] ) 
		# for last ghost, find min, index come back to 0
        elif AgentIndex == gameState.getNumAgents()-1:
            return min( [float("inf")] + 
						[MinMax(gameState.generateSuccessor(AgentIndex, action)
							    ,depth - 1
							    , 0  ) 
					for action in gameState.getLegalActions(AgentIndex)] )
		# for other ghosts, find min
        else:
            return min( [float("inf")] + 
						[MinMax(gameState.generateSuccessor(AgentIndex, action)
							    ,depth 
							    , AgentIndex + 1  ) 
					for action in gameState.getLegalActions(AgentIndex)] )
	
    legalMoves = gameState.getLegalActions()
    scores = [MinMax(gameState.generateSuccessor(0,action) , self.depth , 1) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    if not bestIndices: return Directions.STOP
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def AlphaBeta(gameState , depth , AgentIndex ,alpha , beta ):
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
		# for pacman, find max
        if AgentIndex == 0:
            m = -float("inf")
            for action in gameState.getLegalActions(AgentIndex):
                m = max(m , AlphaBeta(gameState.generateSuccessor(AgentIndex, action)
										,depth 
										,AgentIndex + 1  
										,alpha
										,beta)   )
                if m>beta: return m
                alpha = max(m,alpha)
            return m
		# for last ghost, find min, index come back to 0
        elif AgentIndex == gameState.getNumAgents()-1:
            m = float("inf")
            for action in gameState.getLegalActions(AgentIndex):
                m = min(m , AlphaBeta(gameState.generateSuccessor(AgentIndex, action)
										,depth -1
										,0
										,alpha
										,beta)   )
                if m<alpha: return m
                beta = min(m,beta)
            return m

		# for other ghosts, find min
        else:
            m = float("inf")
            for action in gameState.getLegalActions(AgentIndex):
                m = min(m , AlphaBeta(gameState.generateSuccessor(AgentIndex, action)
										,depth
										,AgentIndex + 1
										,alpha
										,beta)   )
                if m<alpha: return m
                beta = min(m,beta)
            return m
	
    legalMoves = gameState.getLegalActions()
    bestaction = Directions.STOP
    score = -(float("inf"))
    alpha = -(float("inf"))
    beta = float("inf")
    for action in legalMoves:
        nextState = gameState.generateSuccessor(0, action)
        prevscore = score
        score = max(score, AlphaBeta(nextState, self.depth , 1 , alpha, beta))
        if score > prevscore:
            bestaction = action
        if score >= beta:
            return bestaction
        alpha = max(alpha, score)
    return bestaction
	

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    def ExpectiMax(gameState , depth , AgentIndex):
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
		# for pacman, find max
        if AgentIndex == 0:
            return max( [-float("inf")] + 
						[ExpectiMax(gameState.generateSuccessor(AgentIndex, action)
							    ,depth 
							    ,AgentIndex + 1  ) 
					for action in gameState.getLegalActions(AgentIndex)] ) 
		# for last ghost, calculate Expected value, index come back to 0
        elif AgentIndex == gameState.getNumAgents()-1:
            posibleList = [ExpectiMax(gameState.generateSuccessor(AgentIndex, action)
							    ,depth - 1
							    , 0  ) 
					for action in gameState.getLegalActions(AgentIndex)]
            return sum(posibleList) / float(len(posibleList)) 
		# for other ghosts, calculate Expected value
        else:
            posibleList = [ExpectiMax(gameState.generateSuccessor(AgentIndex, action)
							    ,depth
							    ,AgentIndex + 1 ) 
					for action in gameState.getLegalActions(AgentIndex)]
            return sum(posibleList) / float(len(posibleList)) 
    legalMoves = gameState.getLegalActions()
    scores = [ExpectiMax(gameState.generateSuccessor(0,action) , self.depth , 1) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    if not bestIndices: return Directions.STOP
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
	
	For Pacman to win and survive the game, the following factor should be concerned:
	
	1. number of food rest
	this number should be
	
	2. distant to all foods
	the smaller the better.
	also consider that the nearest should weighted more
	so I use 1/distance as a feature
	
	3. distance to ghost
	the larger the better. But ghost only matters when it's nearby
	so I set two conditions: if the ghost is near, then add a distance feature,
	if not, then simply ignore it.
	
	4. distance to scared ghost
	the smaller the better.
	
	
	finally, by adjusting each parameter, 
	one can get a average of win rate of 9%, scoring 1300(include win and lost)
	
    """
    "*** YOUR CODE HERE ***"	
    # Useful information you can extract from a GameState (pacman.py)
    curPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    #scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #distGhost = [util.manhattanDistance(curPos , gs.getPosition()) for gs in ghostStates]
    distFood  = [util.manhattanDistance(curPos , ofp) for ofp in foodPos.asList()]

	# score setting
    foodRestRatio = 23
    foodDistRatio = 8
    ghostDistRatio = 100
    ghostRange = 9
    scaredGhostDistRatio = 300
    scaredGhostChasedDistRatio = 500
	
	# scoring start!
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -(float("inf")) 		

    score = 0 - len(distFood) * foodRestRatio
	
    for df in distFood:
        score += foodDistRatio/(df+0.001)
    for ghost in ghostStates:
        dist = util.manhattanDistance(curPos , ghost.getPosition())
        if ghost.scaredTimer>0 and dist>ghost.scaredTimer/2:
            score += scaredGhostDistRatio / (dist+0.001)
        elif ghost.scaredTimer>0 and dist<=ghost.scaredTimer/2:
            score += scaredGhostChasedDistRatio / (dist+0.001)
        elif dist>ghostRange:
            score -= ghostDistRatio / (dist+0.001) 
	
    score += currentGameState.getScore() 
    return score

# Abbreviation
better = betterEvaluationFunction


