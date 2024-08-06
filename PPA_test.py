import numpy as np
import matplotlib.pyplot as plt
import math as math

def sgn(number):
    # Returns -1 if number is negative, 1 otherwise
    if number >= 0:
        return 1
    else:
        return -1
    
def pt_to_pt_distance(pt1,pt2):
    # Returns the Euclidean distance between two points
    distance = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
    return distance

path1 = [[0.0, 0.0], [0.571194595265405, -0.4277145118491421], [1.1417537280142898, -0.8531042347260006],
         [1.7098876452457967, -1.2696346390611464], [2.2705328851607995, -1.6588899151216996],
          [2.8121159420106827, -1.9791445882187304], [3.314589274316711, -2.159795566252656], 
          [3.7538316863009027, -2.1224619985315876], [4.112485112342358, -1.8323249172947023],
            [4.383456805594431, -1.3292669972090994], [4.557386228943757, -0.6928302521681386], 
            [4.617455513800438, 0.00274597627737883], [4.55408382321606, 0.6984486966257434], 
            [4.376054025556597, 1.3330664239172116], [4.096280073621794, 1.827159263675668], 
            [3.719737492364894, 2.097949296701878], [3.25277928312066, 2.108933125822431],
              [2.7154386886417314, 1.9004760368018616], [2.1347012144725985, 1.552342808106984],
                [1.5324590525923942, 1.134035376721349], [0.9214084611203568, 0.6867933269918683], [0.30732366808208345, 0.22955002391894264], [-0.3075127599907512, -0.2301742560363831], [-0.9218413719658775, -0.6882173194028102], [-1.5334674079795052, -1.1373288016589413], [-2.1365993767877467, -1.5584414896876835], [-2.7180981380280307, -1.9086314914221845], [-3.2552809639439704, -2.1153141204181285], [-3.721102967810494, -2.0979137913841046], [-4.096907306768644, -1.8206318841755131], [-4.377088212533404, -1.324440752295139], [-4.555249804461285, -0.6910016662308593], [-4.617336323713965, 0.003734984720118972], [-4.555948690867849, 0.7001491248072772], [-4.382109193278264, 1.3376838311365633], [-4.111620918085742, 1.8386823176628544], [-3.7524648889185794, 2.1224985058331005], [-3.3123191098095615, 2.153588702898333], [-2.80975246649598, 1.9712114570096653], [-2.268856462266256, 1.652958931009528], [-1.709001159778989, 1.2664395490411673], [-1.1413833971013372, 0.8517589252820573], [-0.5710732645795573, 0.4272721367616211], [0, 0], [0.571194595265405, -0.4277145118491421]]
currentPos = [0,0]
currentHeading = 5.75959
lastFoundIndex = 0
lookAheadDis = 0.8
linearVel = 100

# Set this to true if you use rotations
using_rotation = False

def pure_pursuit_step(path, currentPos, currentHeading, lookAheadDis, LFindex):
  '''
  Pure pursuit algorithm for a robot to follow a path.
  
  Inputs:
  path - list of points that the robot should follow.
  currentPos - list of the current x and y position of the robot.
  currentHeading - the current heading of the robot in degrees.  
  lookAheadDis - the distance the robot should look ahead to find the goal point.
  LFindex - the index of the last found point in the path.
  
  Outputs:
  goalPt - the point the robot should aim to reach.
  lastFoundIndex - the index of the last found point in the path.
  turnVel - the turning velocity of the robot.
  '''

  # extract currentX and currentY
  currentX = currentPos[0]
  currentY = currentPos[1]

  # use for loop to search intersections
  lastFoundIndex = LFindex
  intersectFound = False
  startingIndex = lastFoundIndex

  for i in range(startingIndex, len(path)-1):

    # beginning of line-circle intersection code
    x1 = path[i][0] - currentX
    y1 = path[i][1] - currentY
    x2 = path[i+1][0] - currentX
    y2 = path[i+1][1] - currentY
    dx = x2 - x1
    dy = y2 - y1
    dr = math.sqrt (dx**2 + dy**2)
    D = x1*y2 - x2*y1
    discriminant = (lookAheadDis**2) * (dr**2) - D**2

    if discriminant >= 0:
      # Solutions exist
      sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
      sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
      sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
      sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

      sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
      sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]

      # end of line-circle intersection code
      min_x = min(path[i][0], path[i+1][0])
      min_y = min(path[i][1], path[i+1][1])
      max_x = max(path[i][0], path[i+1][0])
      max_y = max(path[i][1], path[i+1][1])

      # if one or both of the solutions are in range
      if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) or ((min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):

        foundIntersection = True

        # if both solutions are in range, check which one is better
        if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) and ((min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):
          # make the decision by compare the distance between the intersections and the next point in path
          if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
            goalPt = sol_pt1
          else:
            goalPt = sol_pt2
        
        # if not both solutions are in range, take the one that's in range
        else:
          # if solution pt1 is in range, set that as goal point
          if (min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y):
            goalPt = sol_pt1
          else:
            goalPt = sol_pt2
          
        # only exit loop if the solution pt found is closer to the next pt in path than the current pos
        if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
          # update lastFoundIndex and exit
          lastFoundIndex = i
          break
        else:
          # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
          lastFoundIndex = i+1
        
      # if no solutions are in range
      else:
        foundIntersection = False
        # no new intersection found, potentially deviated from the path
        # follow path[lastFoundIndex]
        goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

  # obtained goal point, now compute turn vel
  # initialize proportional controller constant
  Kp = 3

  # calculate absTargetAngle with the atan2 function
  # absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/np.pi
  absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0])
  if absTargetAngle < 0: absTargetAngle += 2*np.pi

  # compute turn error by finding the minimum angle
  turnError = absTargetAngle - currentHeading
  if turnError > np.pi or turnError < -np.pi :
    turnError = -1 * sgn(turnError) * (2*np.pi - abs(turnError))
  
  # apply proportional controller
  turnVel = Kp*turnError
  
  return goalPt, lastFoundIndex, turnVel

goal_point, lastIndex, omega = pure_pursuit_step(path1, currentPos, currentHeading, lookAheadDis, lastFoundIndex)

print("end of code")